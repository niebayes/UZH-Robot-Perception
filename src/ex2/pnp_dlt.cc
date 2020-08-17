#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "common_utils.h"
#include "google_suite.h"
#include "opencv2/opencv.hpp"
// #include "unsupported/Eigen/KroneckerProduct"

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath{"data/ex2/"};

  // Load data
  Eigen::Matrix3d K;
  Eigen::Matrix2Xd observations;
  Eigen::Matrix3Xd p_W_corners;
  // LoadK(kFilePath + "K.txt", &K);
  // LoadObservations(kFilePath + "detected_corners.txt", &observations);
  LoadObjectPoints(kFilePath + "p_W_corners.txt", &p_W_corners);

  // Run DLT
  // auto res = EstimateCameraPoseDLT(observations, p_W_corners, true);

  return EXIT_SUCCESS;
}
