#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "armadillo"
#include "transfer.h"
// #include "common.h"
// #include "feature.h"
#include "google_suite.h"
// #include "interpolation.h"
// #include "io.h"
#include "matlab_port.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

//@brief Return an image given file name, data depth and rescale factor.
//@param file_name String denoting the file name including the relative file
// path.
//@param ddepth Data depth of the elements in the image. It's specified as CV_8U
// or CV_64F etc.
//@param rescale_factor The returned image is rescaled according to this factor.
//@return An cv::Mat object.
cv::Mat GetRescaledImage(const std::string& file_name,
                         const double rescale_factor = 1.0) {
  cv::Mat img = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
  cv::resize(img, img, {}, rescale_factor, rescale_factor, cv::INTER_AREA);
  return uzh::im2double(img);
}

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath{"data/ex4/"};
  cv::Mat img_1_show = cv::imread(kFilePath + "img_1.jpg", cv::IMREAD_COLOR);
  cv::Mat img_2_show = cv::imread(kFilePath + "img_1.jpg", cv::IMREAD_COLOR);
  // cv::Mat img_1, img_2;
  // cv::cvtColor(img_1_show, img_1, cv::COLOR_BGR2GRAY, 1);
  // cv::cvtColor(img_2_show, img_2, cv::COLOR_BGR2GRAY, 1);

  // Decimate the images for speed.
  // The original images are [3024 x 4032 x 3] color images.
  const double kRescaleFactor = 0.3;
  cv::Mat left_image =
      GetRescaledImage(kFilePath + "img_1.jpg", kRescaleFactor);
  cv::Mat right_image =
      GetRescaledImage(kFilePath + "img_2.jpg", kRescaleFactor);

  // Degrees to which the right image is rotated to test SIFT rotation
  // invariance. Positive -> counter-clockwise and negative -> clockwise.
  const double degree = 0;
  if (degree != 0) {
    right_image = uzh::imrotate(right_image, degree);
  }

  return EXIT_SUCCESS;
}