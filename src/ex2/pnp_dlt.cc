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
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"
// #include "unsupported/Eigen/KroneckerProduct"

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath{"data/ex2/"};

  return EXIT_SUCCESS;
}
