#include "two_view_geometry.h"

#include <string>
#include <tuple>

#include "arma_traits.h"
#include "armadillo"
#include "google_suite.h"
#include "io.h"
#include "matlab_port.h"
#include "opencv2/opencv.hpp"
#include "transfer.h"
#include "two_view_geometry.h"

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  return EXIT_SUCCESS;
}