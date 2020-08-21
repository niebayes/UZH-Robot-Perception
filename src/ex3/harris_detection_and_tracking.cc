#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "feature.h"
#include "google_suite.h"
#include "opencv2/opencv.hpp"

int main(int /*argv*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath = "data/ex3/";
  cv::Mat image = cv::imread(kFilePath + "KITTI/000000.png",
                             cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
  cv::Mat harris_response, shi_tomasi_response;
  const int kPatchSize = 9;
  const double kHarrisKappa = 0.06;

  // Part I: compute response.
  // The shi_tomasi_response is computed as comparison.
  HarrisResponse(image, harris_response, kPatchSize, kHarrisKappa);
  ShiTomasiResponse(image, shi_tomasi_response, 9);

  // Part II: select keypoints

  return EXIT_SUCCESS;
}
