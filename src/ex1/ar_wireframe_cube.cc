#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>  // C++17: std::optional
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "google_suite.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"
// #include "unsupported/Eigen/CXX11/Tensor"

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
// TODO Generalize this function to arbitrarily accomodating [low, hight] range
// values. E.g. use OpenCV's cv::Range
template <typename Derived>
static void Meshgrid(const int width, const int height,
                     Eigen::MatrixBase<Derived>* X,
                     Eigen::MatrixBase<Derived>* Y) {
  const Eigen::VectorXi x = Eigen::VectorXi::LinSpaced(width, 0, width - 1),
                        y = Eigen::VectorXi::LinSpaced(height, 0, height - 1);
  *X = x.transpose().replicate(height, 1);
  *Y = y.replicate(1, width);
}

//@brief Imitate matlab's meshgrid operating on 3D grid though.
// You could also use eigen's Tensor module which is not supported yet though.
//@ref
// http://eigen.tuxfamily.org/dox-devel/unsupported/group__CXX11__Tensor__Module.html
//@warning If using fixed-size eigen objects, care has to be taken on the
// alignment issues.
//@ref https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html
//? Why "template argument deduction failed"?
// template <typename Derived>
static void Meshgrid3D(const cv::Range& x_range, const cv::Range& y_range,
                       const cv::Range& z_range,
                       std::vector<Eigen::MatrixXi>* X,
                       std::vector<Eigen::MatrixXi>* Y,
                       std::vector<Eigen::MatrixXi>* Z) {
  //  std::vector<typename Eigen::MatrixBase<Derived>>* X,
  //  std::vector<typename Eigen::MatrixBase<Derived>>* Y,
  //  std::vector<typename Eigen::MatrixBase<Derived>>* Z) {
  const int width = x_range.size() + 1, height = y_range.size() + 1,
            depth = z_range.size() + 1;
  const Eigen::VectorXi x = Eigen::VectorXi::LinSpaced(width, x_range.start,
                                                       x_range.end),
                        y = Eigen::VectorXi::LinSpaced(height, y_range.start,
                                                       y_range.end),
                        z = Eigen::VectorXi::LinSpaced(depth, z_range.start,
                                                       z_range.end);
  // const Eigen::MatrixBase<Derived>
  const Eigen::MatrixXi X_any_depth = x.transpose().replicate(height, 1),
                        Y_any_depth = y.replicate(1, width);
  for (int d = 0; d < depth; ++d) {
    X->push_back(X_any_depth);
    Y->push_back(Y_any_depth);
    Eigen::MatrixXi Z_d_depth(height, width);
    Z_d_depth.fill(z(d));
    Z->push_back(Z_d_depth);
  }
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
// TODO Change normalized_image_points from Matrix2Xd to Matrix3Xd to accomodate
// the convection that normalized image points are expressed at Z = 1 where the
// coordinates are 3D vectors.
template <typename Derived>
static void DistortPoints(
    const Eigen::MatrixBase<Derived>& normalized_image_points,
    Eigen::MatrixBase<Derived>* distorted_image_points,
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
//? This paradigm is not working.
// template <typename Derived>
// const Eigen::Ref<const Eigen::MatrixBase<Derived>>& object_points
static void ProjectPoints(
    const Eigen::Ref<const Eigen::Matrix3Xd>& object_points,
    Eigen::Matrix2Xd* image_points, const Eigen::Ref<const Eigen::Matrix3d>& K,
    int project_mode, std::optional<Eigen::Vector2d> D_opt = std::nullopt) {
  Eigen::Vector2d D = Eigen::Vector2d::Zero();
  if (project_mode == PROJECT_WITH_DISTORTION && D_opt) {
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

//@brief Interpolation methods used in UndistortImage.
namespace {
enum InterpolationMethods : int { NEAREST_NEIGHBOR, BILINEAR };
}

//@brief Undistort image according to distortion function \Tau specified with
// distortion coefficients D
static cv::Mat UndistortImage(const cv::Mat& distorted_image,
                              const Eigen::Ref<const Eigen::Matrix3d>& K,
                              const Eigen::Ref<const Eigen::Vector2d>& D,
                              int interpolation_method) {
  if (distorted_image.channels() > 1) {
    LOG(ERROR) << "Only support grayscale image at this moment";
  }

  const cv::Size& image_size = distorted_image.size();
  cv::Mat undistorted_image = cv::Mat::zeros(image_size, CV_8UC1);
  Eigen::MatrixXd distorted_image_eigen(image_size.height, image_size.width);
  cv::cv2eigen(distorted_image, distorted_image_eigen);

  // Use backward warping
  // TODO Optimize the interpolation processing; use vectorized techniques. For
  // bilinear interpolation, consider using "shift" to wrap single interpolating
  // process into a matrix-like one.
  for (int u = 0; u < image_size.width; ++u) {
    for (int v = 0; v < image_size.height; ++v) {
      // x = (u, v) is the pixel in undistorted image

      // Find the corresponding distorted image if applied the disortion
      // coefficients K
      // First, find the normalized image coordinates
      Eigen::Vector2d normalized_image_point =
          (K.inverse() * Eigen::Vector2d{u, v}.homogeneous()).hnormalized();

      // Apply the distortion
      Eigen::Vector2d distorted_image_point;
      DistortPoints(normalized_image_point, &distorted_image_point, D);

      // Convert back to pixel coordinates
      distorted_image_point.noalias() =
          (K * distorted_image_point.homogeneous()).hnormalized();

      // Apply interpolation
      // up: distorted x coordinate; vp: distorted y coordinate.
      const double up = distorted_image_point.x(),
                   vp = distorted_image_point.y();
      // up_0: squeeze up to the closest pixel nearest to up along the upper
      // left direction; vp_0, in the same principle.
      const double up_0 = std::floor(up), vp_0 = std::floor(vp);
      uchar intensity = 0;
      switch (interpolation_method) {
        case NEAREST_NEIGHBOR:
          // Nearest-neighbor interpolation
          //@ref https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
          //! The correct way do this may be using std::round. However, we use
          //! std::floor here for the sake of simplicity and consistency.
          if (up_0 >= 0 && up_0 < image_size.width && vp_0 >= 0 &&
              vp_0 < image_size.height) {
            // TODO Elegantly resolve narrowing issue here.
            intensity = distorted_image.at<uchar>({up_0, vp_0});
          }
          break;

        case BILINEAR:
          // Bilinear interpolation
          // Use bilinear interpolation to counter against edge artifacts.
          //! We apply the unit square paradigm considering the simplicity.
          //@ref
          // https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
          if (up_0 + 1 >= 0 && up_0 + 1 < image_size.width && vp_0 + 1 >= 0 &&
              vp_0 + 1 < image_size.height) {
            const double x = up - up_0, y = vp - vp_0;
            // TODO Elegantly resolve narrowing issue here.
            const Eigen::Matrix2d four_corners =
                (Eigen::Matrix<uchar, 2, 2>()
                     << distorted_image.at<uchar>({up_0, vp_0}),
                 distorted_image.at<uchar>({up_0, vp_0 + 1}),
                 distorted_image.at<uchar>({up_0 + 1, vp_0}),
                 distorted_image.at<uchar>({up_0 + 1, vp_0 + 1}))
                    .finished()
                    .cast<double>();
            intensity = cv::saturate_cast<uchar>(
                Eigen::Vector2d{1 - x, x}.transpose() * four_corners *
                Eigen::Vector2d{1 - y, y});
          }
          break;

        default:
          LOG(ERROR) << "Invalid interpolation method";
          break;
      }
      undistorted_image.at<uchar>({u, v}) = intensity;
    }
  }
  const std::string log_info{std::string{"Undistorted an image with "}.append(
      interpolation_method ? "bilinear interpolation"
                           : "nearest-neighbor interpolation")};
  LOG(INFO) << log_info;
  return undistorted_image;
}

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath{"data/ex1/"};

  // Part I:
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
  ProjectPoints(p_C_corners, &image_points, K, PROJECT_WITHOUT_DISTORTION);

  // Superimpose points on the image
  const int kImageIndex = 1;
  const std::string kImageName{cv::format(
      (kFilePath + "images_undistorted/img_%04d.jpg").c_str(), kImageIndex)};
  cv::Mat image = cv::imread(kImageName, cv::IMREAD_COLOR);
  const Eigen::VectorXi& x = image_points.row(0).cast<int>();
  const Eigen::VectorXi& y = image_points.row(1).cast<int>();
  Scatter(image, x, y, 3, {0, 0, 255}, cv::FILLED);
  cv::imshow("Scatter plot with corners reprojected", image);
  // cv::waitKey(0);

  // Draw a customized cube on the undistorted image
  // TODO Modularize the codes below.
  const Eigen::Vector3i cube{2, 2, 2};
  std::vector<Eigen::MatrixXi> cube_X, cube_Y, cube_Z;
  Meshgrid3D(cv::Range(0, cube.x() - 1), cv::Range(0, cube.y() - 1),
             cv::Range(-cube.z() + 1, 0), &cube_X, &cube_Y, &cube_Z);
  const int kNumVerticesPerDepth = cube.x() * cube.y();
  const int depth = cube.z();
  Eigen::Matrix3Xd p_W_cube(3, kNumVerticesPerDepth * depth);
  for (int d = 0; d < depth; ++d) {
    cube_X[d].resize(1, kNumVerticesPerDepth);
    cube_Y[d].resize(1, kNumVerticesPerDepth);
    cube_Z[d].resize(1, kNumVerticesPerDepth);
    p_W_cube.row(0).segment(d * kNumVerticesPerDepth, kNumVerticesPerDepth) =
        cube_X[d].cast<double>();
    p_W_cube.row(1).segment(d * kNumVerticesPerDepth, kNumVerticesPerDepth) =
        cube_Y[d].cast<double>();
    p_W_cube.row(2).segment(d * kNumVerticesPerDepth, kNumVerticesPerDepth) =
        cube_Z[d].cast<double>();
  }
  const double kOffsetX = 3 * kCellSize, kOffsetY = 1 * kCellSize,
               kScaling = 2 * kCellSize;
  p_W_cube.noalias() = (kScaling * p_W_cube).colwise() +
                       Eigen::Vector3d{kOffsetX, kOffsetY, 0.0};

  // Draw the cube frame by frame to create a video
  cv::Mat sample_image =
      cv::imread(kFilePath + "images/img_0001.jpg", cv::IMREAD_COLOR);
  const cv::Size frame_size = sample_image.size();
  cv::VideoWriter video(kFilePath + "cube_video.avi",
                        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0,
                        frame_size, true);
  if (video.isOpened()) {
    for (int pose_index = 0;; ++pose_index) {
      cv::Mat frame =
          cv::imread(cv::format((kFilePath + "images/img_%04d.jpg").c_str(),
                                pose_index + 1),
                     cv::IMREAD_COLOR);
      if (frame.empty()) {
        LOG(INFO) << "Wrote " << pose_index << " images to the video";
        break;
      }
      const std::vector<double>& camera_pose = poses[pose_index];
      RigidTransformation T_C_W_cube;
      PoseVectorToTransformationMatrix(camera_pose, &T_C_W_cube);
      const Eigen::Matrix3Xd p_C_cube =
          T_C_W_cube * p_W_cube.colwise().homogeneous();
      Eigen::Matrix2Xd cube_image_points;
      ProjectPoints(p_C_cube, &cube_image_points, K, PROJECT_WITH_DISTORTION,
                    D);
      //! The Meshgrid3D returns points with column-major order, not a cyclic
      //! order. Hence, you need to swap the corresponding columns to get a
      //! cyclic order in order to bootstrap the cube drawing.
      cube_image_points.leftCols(4).col(2).swap(
          cube_image_points.leftCols(4).col(3));
      cube_image_points.rightCols(4).col(2).swap(
          cube_image_points.rightCols(4).col(3));

      //! Transfer from eigen to OpenCV to utilize OpenCV's drawing functions.
      cv::Mat cube_base, cube_top;
      cv::eigen2cv(Eigen::Matrix<double, 2, 4>(
                       cube_image_points.leftCols(kNumVerticesPerDepth).data()),
                   cube_base);
      cv::eigen2cv(
          Eigen::Matrix<double, 2, 4>(
              cube_image_points.rightCols(kNumVerticesPerDepth).data()),
          cube_top);

      //! The conversion below is necessary due to the assertions in
      //! cv::polylines.
      cube_base.reshape(1).convertTo(cube_base, CV_32S);
      cube_top.reshape(1).convertTo(cube_top, CV_32S);
      cv::polylines(frame, cube_base.t(), true, {0, 0, 255}, 3);
      cv::polylines(frame, cube_top.t(), true, {0, 0, 255}, 3);
      for (int i = 0; i < kNumVerticesPerDepth; ++i) {
        cv::line(frame, {cube_base.col(i)}, {cube_top.col(i)}, {0, 0, 255}, 3);
      }
      video << frame;
    }
    video.release();
    LOG(INFO) << "Successfully wrote a video file: cube_video.avi";
  } else {
    LOG(ERROR) << "Failed writing the video file: cube_video.avi";
  }

  // Part II: undistort an image
  cv::Mat distorted_image =
      cv::imread(kFilePath + "images/img_0001.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat undistorted_image_nearest_neighbor =
      UndistortImage(distorted_image, K, D, NEAREST_NEIGHBOR);
  cv::Mat undistorted_image_bilinear =
      UndistortImage(distorted_image, K, D, BILINEAR);
  cv::Mat comparison(distorted_image.rows, 2 * distorted_image.cols, CV_8UC1);
  cv::hconcat(undistorted_image_nearest_neighbor, undistorted_image_bilinear,
              comparison);
  cv::imshow(
      "Comparison between nearest-neighbor interpolation and bilinear "
      "interpolation in undistortion",
      comparison);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}