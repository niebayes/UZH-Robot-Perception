#include <cmath>
#include <fstream>
#include <iostream>
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

static std::vector<std::vector<double>> LoadPoses(
    const std::string& file_name) {
  std::vector<std::vector<double>> poses;

  std::ifstream fin(file_name, std::ios::in);
  std::string pose;
  while (std::getline(fin, pose)) {
    double w_x, w_y, w_z, t_x, t_y, t_z;
    std::istringstream iss(pose);
    if (iss.good() && iss >> w_x >> w_y >> w_z >> t_x >> t_y >> t_z) {
      poses.push_back(std::vector<double>{w_x, w_y, w_z, t_x, t_y, t_z});
    }
  }
  LOG(INFO) << "Load " << poses.size() << " poses";
  return poses;
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
static void DistortPoints(
    const Eigen::Ref<const Eigen::Matrix2Xd>& normalized_image_points,
    Eigen::Matrix2Xd* distorted_image_points,
    const Eigen::Ref<const Eigen::VectorXd>& D) {
  // Unpack D to get the distortion coefficients
  const double k1 = D(INDEX_RADIAL_K1);
  const double k2 = D(INDEX_RADIAL_K2);

  const Eigen::VectorXd r2 = normalized_image_points.colwise().squaredNorm();
}

//@brief Project modes used in ProjectPoints.
// If PROJECT_WITH_DISTORTION, the function protects 3D scene points considering
// distortion and otherwise not.
namespace {
enum ProjectModes : int { PROJECT_WITH_DISTORTION, PROJECT_WITHOUT_DISTORTION };
}

//@brief Project 3D scene points according to calibration matrix K and optional
// user provided distortion coefficients D
template <typename Derived, typename... Args>
static void ProjectPoints(
    const Eigen::Ref<const Eigen::MatrixBase<Derived>>& object_points,
    Eigen::Matrix2Xd* image_points, const Eigen::Ref<const Eigen::Matrix3d>& K,
    int project_mode, Args... D) {
  // w
}

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath{"data/ex1/images/"};
  const int kImageIndex = 1;
  const std::string kFileName{
      cv::format((kFilePath + "img_%04d.jpg").c_str(), kImageIndex)};
  cv::Mat image = cv::imread(kFileName, cv::IMREAD_GRAYSCALE);
  const std::vector<std::vector<double>> poses =
      LoadPoses("data/ex1/poses.txt");

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
  Eigen::MatrixXd p_W_corners(3, kNumCorners);
  p_W_corners.row(0) = X.cast<double>();
  p_W_corners.row(1) = Y.cast<double>();
  p_W_corners.row(2).setZero();
  p_W_corners *= kCellSize;

  // Rigid transformation from world coord. to camera coord.
  RigidTransformation T_C_W;
  PoseVectorToTransformationMatrix(poses[0], &T_C_W);
  Eigen::MatrixXd p_C_corners = T_C_W * p_W_corners.colwise().homogeneous();

  return EXIT_SUCCESS;
}