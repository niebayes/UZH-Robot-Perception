#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "google_suite.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

int main(int /*argv*/, char** argv) {
  const std::string kFilePath = "data/ex3/";
  cv::Mat image =
      cv::imread(kFilePath + "KITTI/000000.png", cv::IMREAD_GRAYSCALE);
  return EXIT_SUCCESS;
}