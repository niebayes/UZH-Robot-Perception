#include <string>
#include <tuple>
#include <vector>

#include "feature/matching.h"
#include "google_suite.h"
#include "io.h"
#include "klt.h"
#include "matlab_port.h"
#include "opencv2/opencv.hpp"
#include "transfer.h"

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath{"data/ex8/"};

  return EXIT_SUCCESS;
}