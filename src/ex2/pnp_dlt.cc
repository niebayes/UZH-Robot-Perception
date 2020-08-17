#include <cmath>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "google_suite.h"
#include "opencv2/opencv.hpp"
#include "unsupported/Eigen/KroneckerProduct"

void EstimateCameraPoseWithDLT() {
  // matlab's kron

  // matlab's svd
}

// is_valid_rotation_mat

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  return EXIT_SUCCESS;
}