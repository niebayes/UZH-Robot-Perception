#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>  // C++17: std::optional
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "google_suite.h"
#include "manif/manif.h"
#include "opencv2/opencv.hpp"

using RigidTransformation = Eigen::Matrix<double, 3, 4>;

// @brief Load 6dof poses from text file and store them into a 2d vector.
static std::vector<std::vector<double>> LoadPoses(
    const std::string& file_name) {
  std::vector<std::vector<double>> poses;

  std::ifstream fin(file_name, std::ios::in);
  if (fin.is_open()) {
    std::string pose;
    while (std::getline(fin, pose)) {
      double w_x, w_y, w_z, t_x, t_y, t_z;
      std::istringstream iss(pose);
      if (iss.good() && iss >> w_x >> w_y >> w_z >> t_x >> t_y >> t_z) {
        poses.push_back(std::vector<double>{w_x, w_y, w_z, t_x, t_y, t_z});
      }
    }
    fin.close();
    LOG(INFO) << "Loaded " << poses.size() << " poses";
  } else {
    LOG(ERROR) << "Fail loading file " << file_name;
  }
  return poses;
}

//@brief Load camera calibration matrix K and store it into a vector.
static std::vector<double> LoadK(const std::string& file_name) {
  std::vector<double> K;

  std::ifstream fin(file_name, std::ios::in);
  if (fin.is_open()) {
    std::string row;
    while (std::getline(fin, row)) {
      double coeff_1, coeff_2, coeff_3;
      std::istringstream iss(row);
      if (iss.good() && iss >> coeff_1 >> coeff_2 >> coeff_3) {
        K.push_back(coeff_1);
        K.push_back(coeff_2);
        K.push_back(coeff_3);
      }
    }
    fin.close();
  } else {
    LOG(ERROR) << "Fail loading file " << file_name;
  }
  return K;
}

//@brief Load lens distortion coefficients D and store them into a vector.
static std::vector<double> LoadD(const std::string& file_name) {
  std::vector<double> D;

  std::ifstream fin(file_name, std::ios::in);
  if (fin.is_open()) {
    std::string coeffs;
    if (std::getline(fin, coeffs)) {
      double k1, k2;
      std::istringstream iss(coeffs);
      if (iss.good() && iss >> k1 >> k2) {
        D.push_back(k1);
        D.push_back(k2);
      }
    }
    fin.close();
  } else {
    LOG(ERROR) << "Fail loading file " << file_name;
  }
  return D;
}

//@brief Imitate matlab's meshgrid.
template <typename Derived>
static void Meshgrid(const int width, const int height,
                     Eigen::MatrixBase<Derived>* X,
                     Eigen::MatrixBase<Derived>* Y) {
  const Eigen::VectorXi x = Eigen::VectorXi::LinSpaced(width, 0, width - 1),
                        y = Eigen::VectorXi::LinSpaced(height, 0, height - 1);
  *X = x.transpose().replicate(height, 1);
  *Y = y.replicate(1, width);
}

//@brief Toy function for Rodrigues formula transforming rotation vector to
// corresponding rotation matrix
//! When using const Eigen::Ref binds const object, qualify the object with a
//! const. Otherwise, errors induced.
static void Rodrigues(const Eigen::Ref<const Eigen::Vector3d>& rotation_vector,
                      Eigen::Matrix3d* rotation_matrix) {
  //* Simpler way using eigen's API.
  // Eigen::MatrixXd rotation_matrix_tmp =
  //     (Eigen::AngleAxisd(rotation_vector.norm(),
  //     rotation_vector.normalized()))
  //         .matrix();
  // std::cout << rotation_matrix_tmp << '\n';

  const double theta = rotation_vector.norm();
  const Eigen::Vector3d omega = rotation_vector.normalized();
  const Eigen::Matrix3d omega_hat =
      (Eigen::Matrix3d() << 0.0, -omega.z(), omega.y(), omega.z(), 0.0,
       -omega.x(), -omega.y(), omega.x(), 0.0)
          .finished();
  *rotation_matrix = Eigen::Matrix3d::Identity() + std::sin(theta) * omega_hat +
                     (1 - std::cos(theta)) * omega_hat * omega_hat;
}

//@brief Construct a rigid transformation matrix from the pose vector
static void PoseVectorToTransformationMatrix(const std::vector<double>& pose,
                                             RigidTransformation* T) {
  const Eigen::Vector3d rotation_vector{pose[0], pose[1], pose[2]},
      translation{pose[3], pose[4], pose[5]};
  Eigen::Matrix3d rotation_matrix;
  Rodrigues(rotation_vector, &rotation_matrix);
  //! When templating the T, the methods below are ill-defined and induce
  //! errors. There's no need to templating here.
  T->leftCols(3) = rotation_matrix;
  T->rightCols(1) = translation;
}

//@brief Distortion coefficients' indices used to vividly extract values.
// The advantage also includes the good flexibility when transfering to another
// distortion model which may have another set of distortion coefficients.
namespace {
enum DistorionCoefficientIndices : int { INDEX_RADIAL_K1, INDEX_RADIAL_K2 };
}

//@brief Distort normalized image points to get the image points expressed in
// pixel coords.
// Distortion model: x_d = x_u * (1 + k1 * r2 + k2 * r2 * r2)
static void DistortPoints(
    const Eigen::Ref<const Eigen::Matrix2Xd>& normalized_image_points,
    Eigen::Matrix2Xd* distorted_image_points,
    const Eigen::Ref<const Eigen::Vector2d>& D) {
  // Unpack D to get the distortion coefficients
  const double k1 = D(INDEX_RADIAL_K1);
  const double k2 = D(INDEX_RADIAL_K2);

  const Eigen::VectorXd r2 = normalized_image_points.colwise().squaredNorm();
  const Eigen::VectorXd distortion_factor =
      (k1 * r2 + k2 * r2.cwiseProduct(r2)).unaryExpr([](double x) {
        return ++x;
      });
  *distorted_image_points =
      normalized_image_points * distortion_factor.asDiagonal();
}

//@brief Project modes used in ProjectPoints.
// If PROJECT_WITH_DISTORTION, the function protects 3D scene points considering
// distortion and otherwise not.
namespace {
enum ProjectModes : int { PROJECT_WITH_DISTORTION, PROJECT_WITHOUT_DISTORTION };
}

//@brief Project 3D scene points according to calibration matrix K and optional
// user provided distortion coefficients D
//! The std::optional is a feature of C++17. Alternatively you can avoid this by
//! overloading ProjectPoints function.
//? This paradigm is nor working.
// template <typename Derived>
// const Eigen::Ref<const Eigen::MatrixBase<Derived>>& object_points
static void ProjectPoints(
    const Eigen::Ref<const Eigen::Matrix3Xd>& object_points,
    Eigen::Matrix2Xd* image_points, const Eigen::Ref<const Eigen::Matrix3d>& K,
    int project_mode, std::optional<Eigen::Vector2d> D_opt = std::nullopt) {
  Eigen::Vector2d D = Eigen::Vector2d::Zero();
  if (D_opt) {
    D = D_opt.value();
  }
  const Eigen::Matrix2Xd normalized_image_points =
      object_points.colwise().hnormalized();
  DistortPoints(normalized_image_points, image_points, D);
  *image_points = (K * image_points->colwise().homogeneous()).topRows(2);
}

//@brief Imitate matlab's scatter.
static void Scatter(cv::InputOutputArray image,
                    const Eigen::Ref<const Eigen::VectorXi>& x,
                    const Eigen::Ref<const Eigen::VectorXi>& y, int radius,
                    const cv::Scalar& color, int thickness = 1) {
  if (x.size() <= 0 || y.size() <= 0) {
    LOG(ERROR) << "Invalid input vectors";
    return;
  } else if (x.size() != y.size()) {
    LOG(ERROR)
        << "YOU_MIXED_DIFFERENT_SIZED_VECTORS";  // Mimic eigen's behavior
    return;
  }

  const int kNumPoints = x.size();
  for (int i = 0; i < kNumPoints; ++i) {
    cv::circle(image, {x(i), y(i)}, radius, color, thickness);
  }
  LOG(INFO) << "Created a scatter plot with " << kNumPoints
            << " points rendered";
}

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath{"data/ex1/"};

  // Construct a 2D mesh grid where each vertix corrsponds to a inner corner.
  // Given settings, cf. data/ex1/images/img_0001.jpg
  const cv::Size checkerboard(9, 6);
  const int kNumCorners = checkerboard.width * checkerboard.height;
  const double kCellSize = 0.04;  // Unit: meters

  Eigen::MatrixXi X, Y;
  Meshgrid(checkerboard.width, checkerboard.height, &X, &Y);
  X.resize(1, kNumCorners);
  Y.resize(1, kNumCorners);

  // 3D corners expressed in world coord.
  //! In eigen, the dynamic size matrix is actually a matrix
  //! with rows and/or cols set to -1; and eigen use just the
  //! rows and cols for checking. Hence, before manipulating
  //! the rows/cols/coefficients of a dynamic size matrix, be
  //! aware of initializing it first.
  Eigen::Matrix3Xd p_W_corners(3, kNumCorners);
  p_W_corners.row(0) = X.cast<double>();
  p_W_corners.row(1) = Y.cast<double>();
  p_W_corners.row(2).setZero();
  p_W_corners *= kCellSize;

  // Rigid transformation from world coord. to camera coord.
  const std::vector<std::vector<double>> poses =
      LoadPoses(kFilePath + "poses.txt");
  RigidTransformation T_C_W;
  PoseVectorToTransformationMatrix(poses[0], &T_C_W);
  const Eigen::Matrix3Xd p_C_corners =
      T_C_W * p_W_corners.colwise().homogeneous();

  // Project 3D corners to image plane
  // Load K and D
  //@warning This is cool, but also dangerous! Because the Eigen object will NOT
  // create its own memory. It will operate on the memory provided by "data". In
  // other words, working with the Eigen object when the "data" object is out of
  // scope will result in a segmentation fault (or memory access violation)
  //! Hence you shall not use const qualifier to K_tmp, otherwise errors
  //! induced.
  std::vector<double> K_tmp = LoadK(kFilePath + "K.txt");
  std::vector<double> D_tmp = LoadD(kFilePath + "D.txt");
  //@warning Eigen use column-major storage order! Hence when constructing a
  // eigen Matrix use the data pointer of an external array-like object, be sure
  // transposing it!
  const Eigen::Matrix3d K =
      (Eigen::Map<Eigen::Matrix3d>(K_tmp.data())).transpose();
  const Eigen::Vector2d D(D_tmp.data());

  Eigen::Matrix2Xd image_points;
  ProjectPoints(p_C_corners, &image_points, K, PROJECT_WITH_DISTORTION, D);

  // Superimpose points on the image
  const int kImageIndex = 1;
  const std::string kImageName{
      cv::format((kFilePath + "images/img_%04d.jpg").c_str(), kImageIndex)};
  cv::Mat image = cv::imread(kImageName, cv::IMREAD_COLOR);
  const Eigen::VectorXi& x = image_points.row(0).cast<int>();
  const Eigen::VectorXi& y = image_points.row(1).cast<int>();
  Scatter(image, x, y, 3, {0, 0, 255}, cv::FILLED);
  cv::imshow("", image);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}