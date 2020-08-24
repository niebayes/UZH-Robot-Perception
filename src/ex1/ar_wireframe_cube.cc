#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "common_utils.h"
#include "google_suite.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

// TODO(bayes) Rewrite this to incorporate new written functions.

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
  Matrix34d T_C_W;
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
      Matrix34d T_C_W_cube;
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