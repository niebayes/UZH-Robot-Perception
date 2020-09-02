#include <random>
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

  const std::string kFilePath{"data/ex7/"};

  // Random seed.
  // arma::arma_rng::set_seed(42);
  arma::arma_rng::set_seed_random();

  // Create data for ParabolaRANSAC.
  const int kNumInliers = 20, kNumOutliers = 10;
  const double kNoiseRatio = 0.1;
  const arma::vec3 poly_coeffs = arma::randu<arma::vec>(3);
  // The extremum at the quadratic curve is at x = -b / 2a.
  const double x_extremum = -poly_coeffs(1) / (2 * poly_coeffs(0));
  const double y_lowest =
      arma::as_scalar(arma::polyval(poly_coeffs, arma::vec{x_extremum}));
  // Bound x in the range of length 1.0
  const double x_span = 1.0;
  const double x_start = x_extremum - 0.5;  // x_end = x_extremeum + 0.5
  const double y_highest =
      arma::as_scalar(arma::polyval(poly_coeffs, arma::vec{x_start}));
  const double y_span = y_highest - y_lowest;
  // Random sample x coordinates.
  const arma::rowvec x = arma::randu<arma::rowvec>(kNumInliers) + x_start;
  // Compute corresponding y coordinates.
  arma::rowvec y = arma::polyval(poly_coeffs, x);
  // The data is perturbed by a factor of 0.1 at most along y direction.
  const double max_noise = kNoiseRatio * y_span;
  y += (arma::randu<arma::rowvec>(y.size()) - 0.5) * 2.0 * max_noise;
  arma::mat data(2, kNumInliers + kNumOutliers);
  data.row(0) =
      arma::join_horiz(x, arma::randu<arma::rowvec>(kNumOutliers) + x_start);
  data.row(1) = arma::join_horiz(
      y, arma::randu<arma::rowvec>(kNumOutliers) * y_span + y_lowest);

  // Fit a model with RANSAC.
  arma::mat best_poly_coeffs;
  arma::urowvec max_inlier_counts;
  data = uzh::LoadArma<double>(kFilePath + "data.txt");
  data.print("data");
  std::tie(best_poly_coeffs, max_inlier_counts) =
      uzh::ParabolaRANSAC(data, max_noise);

  best_poly_coeffs.print("coeffs");
  max_inlier_counts.print("max_inlier_cnt");

  return EXIT_SUCCESS;
}