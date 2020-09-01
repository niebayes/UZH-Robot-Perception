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

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  // Load data
  const std::string kFilePath{"data/ex6/"};
  const cv::Mat img1 = cv::imread(kFilePath + "0001.jpg", cv::IMREAD_COLOR);
  const cv::Mat img2 = cv::imread(kFilePath + "0002.jpg", cv::IMREAD_COLOR);
  const arma::mat K = uzh::LoadArma<double>(kFilePath + "K.txt");
  const arma::mat p1 = uzh::LoadArma<double>(kFilePath + "matches0001.txt");
  const arma::mat p2 = uzh::LoadArma<double>(kFilePath + "matches0002.txt");

  // Transform to homogeneous 2D coordinates.
  const arma::mat p1_h = uzh::homogeneous<double>(p1);
  const arma::mat p2_h = uzh::homogeneous<double>(p2);

  // Estimate essential matrix E.
  // Assume K1 = K2 = K.
  const arma::mat E = uzh::EstimateEssentialMatrix(p1_h, p2_h, K, K);

  // Decompose E get Rs and u;
  arma::field<arma::mat> Rs; 
  arma::vec u;
  std::tie(Rs, u) = uzh::DecomposeEssentialMatrix(E);

  // Disambiguate combinations of R and t.
  arma::mat R; 
  arma::mat t;
  std::tie(R, t) = uzh::DisambiguateRelativePoses(Rs, u, p1_h, p2_h, K, K);

  // Triangulate a point cloud from the views.
  const arma::mat M1 = K * arma::eye<arma::mat>(3, 4);
  const arma::mat M2 = K * arma::join_horiz(R, t);
  const arma::mat P = uzh::LinearTriangulation(p1_h, p2_h, M1, M2);

  // Show the result.
  P.print("P:\n");

  return EXIT_SUCCESS;
}