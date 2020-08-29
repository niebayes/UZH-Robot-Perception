#include <string>
#include <tuple>

#include "Eigen/Dense"
#include "armadillo"
#include "google_suite.h"
#include "io.h"
#include "matlab_port.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "stereo.h"
#include "transfer.h"

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  // Load data
  const std::string kFilePath{"data/ex5/"};
  const std::string kLeftImageName{kFilePath + "left/%06d.png"};
  const std::string kRightImageName{kFilePath + "right/%06d.png"};

  cv::Mat left_image =
      cv::imread(cv::format(kLeftImageName.c_str(), 0), cv::IMREAD_GRAYSCALE);
  cv::Mat right_image =
      cv::imread(cv::format(kRightImageName.c_str(), 0), cv::IMREAD_GRAYSCALE);
  arma::mat K = uzh::LoadArma<double>(kFilePath + "K.txt");
  const arma::mat poses = uzh::LoadArma<double>(kFilePath + "poses.txt");

  // Downsample images to speed up computation, as well as K which is expressed
  // in pixels.
  const arma::mat left_img =
      uzh::cv2arma<double>(uzh::imresize(left_image, 0.5)).t();
  const arma::mat right_img =
      uzh::cv2arma<double>(uzh::imresize(right_image, 0.5)).t();
  K.head_rows(2) /= 2.0;

  // Given settings
  const double kBaseLine = 0.54;
  const int kPatchRadius = 5;
  const double kMinDisparity = 5;
  const double kMaxDisparity = 50;
  const arma::vec2 kXLimits{7, 20};
  const arma::vec2 kYLimits{-6, 10};
  const arma::vec2 kZLimits{-5, 5};

  // Part I: calculate pixel disparity
  const arma::mat disparity_map = GetDisparity(
      left_img, right_img, kPatchRadius, kMinDisparity, kMaxDisparity);

  return EXIT_SUCCESS;
}