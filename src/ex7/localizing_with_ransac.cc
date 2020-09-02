#include <string>
#include <tuple>

#include "Eigen/Dense"
#include "armadillo"
#include "ceres/ceres.h"
#include "google_suite.h"
#include "io.h"
#include "matlab_port.h"
#include "opencv2/opencv.hpp"
#include "ransac.h"
#include "transfer.h"

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  return EXIT_SUCCESS;
}