#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "armadillo"
#include "common.h"
#include "feature.h"
#include "google_suite.h"
#include "interpolation.h"
#include "io.h"
#include "matlab_port.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath{"data/ex4/"};
  cv::Mat img_1_show = cv::imread(kFilePath + "img_1.jpg", cv::IMREAD_COLOR);
  cv::Mat img_2_show = cv::imread(kFilePath + "img_1.jpg", cv::IMREAD_COLOR);
  cv::Mat img_1, img_2;
  cv::cvtColor(img_1_show, img_1, cv::COLOR_BGR2GRAY, 1);
  cv::cvtColor(img_2_show, img_2, cv::COLOR_BGR2GRAY, 1);

  return EXIT_SUCCESS;
}