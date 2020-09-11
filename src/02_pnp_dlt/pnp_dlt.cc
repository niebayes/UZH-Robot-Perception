#include <string>

#include "Eigen/Dense"
#include "armadillo"
#include "dlt.h"
#include "google_suite.h"
#include "io.h"
#include "matlab_port.h"
#include "opencv2/opencv.hpp"

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  // Load data files
  const std::string kFilePath{"data/02_pnp_dlt/"};

  const Eigen::Matrix3d K = uzh::armaLoad<Eigen::Matrix3d>(kFilePath + "K.txt");
  const Eigen::MatrixXd observations =
      uzh::armaLoad<Eigen::MatrixXd>(kFilePath + "detected_corners.txt");
  // The p_W_corners.txt contains coordinates of 3D reference points expressed
  // in centimeters which is better to be transformed to canonical unit meter.
  const Eigen::Matrix3Xd p_W_corners =
      0.01 * uzh::armaLoad<Eigen::MatrixX3d>(kFilePath + "p_W_corners.txt")
                 .transpose();

  // Draw the cube frame by frame to create a video
  cv::Mat sample_image = cv::imread(
      kFilePath + "images_undistorted/img_0001.jpg", cv::IMREAD_COLOR);
  const cv::Size frame_size = sample_image.size();
  cv::VideoWriter video(kFilePath + "reprojected.avi",
                        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 15,
                        frame_size, true);
  if (video.isOpened()) {
    for (int image_index = 1;; ++image_index) {
      cv::Mat frame = cv::imread(
          cv::format((kFilePath + "images_undistorted/img_%04d.jpg").c_str(),
                     image_index),
          cv::IMREAD_COLOR);
      if (frame.empty()) {
        LOG(INFO) << "Wrote " << image_index - 1 << " images to the video";
        break;
      }

      // Run DLT
      Eigen::RowVectorXd row_vec = observations.row(image_index - 1);
      const Eigen::Matrix2Xd image_points =
          Eigen::Map<Eigen::Matrix<double, 2, 12>>(row_vec.data());
      uzh::CameraMatrixDLT M_dlt =
          uzh::EstimatePoseDLT(image_points, p_W_corners, K);
      M_dlt.DecomposeDLT();

      // Compare reprojected points and the obvervations.
      const Eigen::Matrix<double, 3, 4> M = M_dlt.getM();
      Eigen::Matrix2Xd reprojected_points;
      uzh::ReprojectPoints(p_W_corners, &reprojected_points, K, M,
                           uzh::PROJECT_WITHOUT_DISTORTION);

      // Compute reprojection error
      double reprojection_error =
          uzh::GetReprojectionError(image_points, reprojected_points);
      cv::putText(frame,
                  cv::format("Reprojection error: %.4f", reprojection_error),
                  {5, 40}, cv::FONT_HERSHEY_PLAIN, 2, {0, 255, 0}, 3);

      // Draw them on current frame.
      const Eigen::VectorXi& reproj_x = reprojected_points.row(0).cast<int>();
      const Eigen::VectorXi& reproj_y = reprojected_points.row(1).cast<int>();
      uzh::scatter(frame, reproj_x, reproj_y, 4, {0, 0, 255});

      const Eigen::VectorXi& observed_x = image_points.row(0).cast<int>();
      const Eigen::VectorXi& observed_y = image_points.row(1).cast<int>();
      uzh::scatter(frame, observed_x, observed_y, 4, {0, 255, 0});

      // Show the frame.
      cv::imshow("Reprojected points vs. Ground truth", frame);
      const char key = cv::waitKey(50);
      if (key == 32) cv::waitKey(0);  // 'Space' key -> pause.

      // Write to the video.
      video << frame;
    }
    video.release();
    LOG(INFO) << "Successfully wrote a video file: reprojected.avi";
  } else {
    LOG(ERROR) << "Failed writing the video file: reprojected.avi";
  }

  return EXIT_SUCCESS;
}
