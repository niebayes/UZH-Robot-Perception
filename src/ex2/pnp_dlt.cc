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

  // Load data
  // auto cv_corners = Load<double>(kFilePath + "p_W_corners.txt");
  // Eigen::MatrixXd p_W_corners(cv_corners.rows, cv_corners.cols);
  // cv::cv2eigen(cv_corners.matrix().t(), p_W_corners);
  // std::cout << p_W_corners << '\n';

  // auto cv_K = Load<double>(kFilePath + "K.txt");
  // Eigen::MatrixXd K(cv_K.rows, cv_K.cols);
  // cv::cv2eigen(cv_K.matrix(), K);
  // std::cout << K << '\n';

  // auto cv_observations = Load<double>(kFilePath + "detected_corners.txt");
  // Eigen::MatrixXd observations(cv_observations.rows, cv_observations.cols);
  // cv::cv2eigen(cv_observations.matrix().t(), observations);
  // std::cout << observations << '\n';
  std::cout << Load<Eigen::MatrixXd>(kFilePath + "detected_corners.txt")
            << '\n';

  // Run DLT
  // auto res = EstimateCameraPoseDLT(observations, p_W_corners, true);

  return EXIT_SUCCESS;
}
