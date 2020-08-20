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
  cv::Mat image =
      cv::imread(kFilePath + "KITTI/000000.png", cv::IMREAD_GRAYSCALE);
  cv::Mat harris_response;
  const int kPatchSize = 9;
  const double harris_kappa = 0.06;
  HarrisResponse(image, harris_response, kPatchSize, harris_kappa);
  return EXIT_SUCCESS;
}
