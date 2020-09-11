#include <string>

#include "Eigen/Dense"
#include "google_suite.h"
#include "image_formation.h"
#include "io.h"
#include "matlab_port.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"
#include "transform.h"

//@brief Construct a rigid transformation matrix from the pose vector
static void PoseVectorToTransformationMatrix(const Eigen::VectorXd& pose,
                                             Eigen::Matrix<double, 3, 4>* T) {
  const Eigen::Vector3d rotation_vector{pose[0], pose[1], pose[2]},
      translation{pose[3], pose[4], pose[5]};
  Eigen::Matrix3d rotation_matrix;
  uzh::Rodrigues(rotation_vector, &rotation_matrix);
  T->leftCols(3) = rotation_matrix;
  T->rightCols(1) = translation;
}

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string file_path{"data/01_ar_wireframe_cube/"};

  // Part I:
  // Construct a 2D mesh grid where each vertix corrsponds to a inner corner.
  // Given settings, cf. data/ex1/images/img_0001.jpg
  const cv::Size checkerboard(9, 6);
  const int kNumCorners = checkerboard.width * checkerboard.height;
  const double kCellSize = 0.04;  // Unit: meters

  Eigen::MatrixXi X, Y;
  uzh::meshgrid(checkerboard.width, checkerboard.height, &X, &Y);
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
  const Eigen::MatrixXd poses =
      uzh::armaLoad<Eigen::MatrixXd>(file_path + "poses.txt").transpose();
  Eigen::Matrix<double, 3, 4> T_C_W;
  PoseVectorToTransformationMatrix(poses.col(0), &T_C_W);
  const Eigen::Matrix3Xd p_C_corners =
      T_C_W * p_W_corners.colwise().homogeneous();

  // Project 3D corners to image plane
  // Load K and D
  const Eigen::Matrix3d K = uzh::armaLoad<Eigen::Matrix3d>(file_path + "K.txt");
  const Eigen::Vector2d D =
      uzh::armaLoad<Eigen::RowVector2d>(file_path + "D.txt").transpose();

  Eigen::Matrix2Xd image_points;
  uzh::ProjectPoints(p_C_corners, &image_points, K,
                     uzh::PROJECT_WITHOUT_DISTORTION);

  // Superimpose points on the image
  const int kImageIndex = 1;
  const std::string kImageName{cv::format(
      (file_path + "images_undistorted/img_%04d.jpg").c_str(), kImageIndex)};
  cv::Mat image = cv::imread(kImageName, cv::IMREAD_COLOR);
  const Eigen::VectorXi& x = image_points.row(0).cast<int>();
  const Eigen::VectorXi& y = image_points.row(1).cast<int>();
  uzh::scatter(image, x, y, 3, {0, 0, 255}, cv::FILLED);
  cv::imshow("Scatter plot with corners reprojected", image);
  cv::waitKey(0);

  // Draw a customized cube on the undistorted image
  const Eigen::Vector3i cube{2, 2, 2};
  std::vector<Eigen::MatrixXi> cube_X, cube_Y, cube_Z;
  uzh::meshgrid(cv::Range(0, cube.x() - 1), cv::Range(0, cube.y() - 1),
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
  p_W_cube = ((kScaling * p_W_cube).colwise() +
              Eigen::Vector3d{kOffsetX, kOffsetY, 0.0})
                 .eval();

  // Draw the cube frame by frame to create a video
  cv::Mat sample_image =
      cv::imread(file_path + "images/img_0001.jpg", cv::IMREAD_COLOR);
  const cv::Size frame_size = sample_image.size();
  cv::VideoWriter video(file_path + "cube_video.avi",
                        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0,
                        frame_size, true);
  if (video.isOpened()) {
    for (int pose_index = 0;; ++pose_index) {
      cv::Mat frame =
          cv::imread(cv::format((file_path + "images/img_%04d.jpg").c_str(),
                                pose_index + 1),
                     cv::IMREAD_COLOR);
      if (frame.empty()) {
        LOG(INFO) << "Wrote " << pose_index << " images to the video";
        break;
      } else {
        LOG(INFO) << "Writing the " << pose_index + 1 << " image.";
      }

      const Eigen::VectorXd& camera_pose = poses.col(pose_index);
      Eigen::Matrix<double, 3, 4> T_C_W_cube;
      PoseVectorToTransformationMatrix(camera_pose, &T_C_W_cube);
      const Eigen::Matrix3Xd p_C_cube =
          T_C_W_cube * p_W_cube.colwise().homogeneous();
      Eigen::Matrix2Xd cube_image_points;
      uzh::ProjectPoints(p_C_cube, &cube_image_points, K,
                         uzh::PROJECT_WITH_DISTORTION, D);
      //! The uzh::meshgrid returns points with column-major order, not a cyclic
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
      cv::imread(file_path + "images/img_0001.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat undistorted_image_nearest_neighbor =
      uzh::UndistortImage(distorted_image, K, D, uzh::NEAREST_NEIGHBOR);
  cv::Mat undistorted_image_bilinear =
      uzh::UndistortImage(distorted_image, K, D, uzh::BILINEAR);
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