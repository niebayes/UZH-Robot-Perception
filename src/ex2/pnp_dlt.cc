#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "google_suite.h"
#include "opencv2/opencv.hpp"
#include "unsupported/Eigen/KroneckerProduct"

void EstimateCameraPoseWithDLT() {
  //@note You can also use Kronecker Product to obtain the Q matrix.
  // More about usages of Kronecker product,
  //@ref
  // https://blogs.sas.com/content/iml/2020/07/27/8-ways-kronecker-product.html

  // matlab's svd
}

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  return EXIT_SUCCESS;
}
