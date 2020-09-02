#include <functional>
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
#include "pcl/visualization/pcl_plotter.h"
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

  // Part I: fit a parabola with RANSAC.
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
  // The data is perturbed by a factor of at most 0.1 along y direction.
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
  std::tie(best_poly_coeffs, max_inlier_counts) =
      uzh::ParabolaRANSAC(data, max_noise);

  // Compare various fits with RANSAC.
  pcl::visualization::PCLPlotter::Ptr plotter(
      new pcl::visualization::PCLPlotter);
  plotter->setTitle("Fitted parabolas");
  plotter->setShowLegend(true);
  plotter->setXRange(x_start, x_start + x_span);
  plotter->setYRange(y_lowest - max_noise, y_highest + max_noise);

  // Plot scattered data points.
  plotter->addPlotData(arma::conv_to<std::vector<double>>::from(data.row(0)),
                       arma::conv_to<std::vector<double>>::from(data.row(1)),
                       "data points", vtkChart::POINTS);
  // Plot ground truth.
  const arma::vec x_plot =
      arma::linspace<arma::vec>(x_start, x_start + x_span, 100);
  plotter->addPlotData(arma::conv_to<std::vector<double>>::from(x_plot),
                       arma::conv_to<std::vector<double>>::from(
                           arma::polyval(poly_coeffs, x_plot)),
                       "ground truth");
  // Plot full fit with all the data.
  const arma::vec full_fit = arma::polyfit(data.row(0), data.row(1), 2);
  plotter->addPlotData(
      arma::conv_to<std::vector<double>>::from(x_plot),
      arma::conv_to<std::vector<double>>::from(arma::polyval(full_fit, x_plot)),
      "full data fit");
  // Plot RANSAC fitted at each iteration except the last one.
  for (int i = 0; i < best_poly_coeffs.n_cols - 1; ++i) {
    const arma::vec coeffs = best_poly_coeffs.col(i);
    plotter->addPlotData(
        arma::conv_to<std::vector<double>>::from(x_plot),
        arma::conv_to<std::vector<double>>::from(arma::polyval(coeffs, x_plot)),
        "");
  }
  // Plot RANSAC fitted result.
  const arma::vec best_coeffs = best_poly_coeffs.tail_cols(1);
  plotter->addPlotData(arma::conv_to<std::vector<double>>::from(x_plot),
                       arma::conv_to<std::vector<double>>::from(
                           arma::polyval(best_coeffs, x_plot)),
                       "RANSAC result");
  plotter->plot();

  // Plot the maximum inlier counts at eack iteration.
  pcl::visualization::PCLPlotter::Ptr inlier_cnt_plotter(
      new pcl::visualization::PCLPlotter);
  inlier_cnt_plotter->setTitle("Maximum inliers count");
  inlier_cnt_plotter->setXTitle("iteration");
  inlier_cnt_plotter->setYTitle("Number of inliers");
  const arma::vec iterations = arma::linspace<arma::vec>(
      1, max_inlier_counts.size(), max_inlier_counts.size());
  inlier_cnt_plotter->addPlotData(
      arma::conv_to<std::vector<double>>::from(iterations),
      arma::conv_to<std::vector<double>>::from(max_inlier_counts),
      "max inliers");
  inlier_cnt_plotter->plot();

  // Compare the RMS error of the full fit and that of RANSAC fit.
  const arma::vec x_rms = arma::linspace<arma::vec>(0.0, 1.0, 100);
  const double full_fit_rms =
      std::sqrt(uzh::arma2eigen(arma::polyval(poly_coeffs, x_rms) -
                                arma::polyval(full_fit, x_rms))
                    .squaredNorm() /
                best_poly_coeffs.n_cols);
  const double ransac_fit_rms =
      std::sqrt(uzh::arma2eigen(arma::polyval(poly_coeffs, x_rms) -
                                arma::polyval(best_coeffs, x_rms))
                    .squaredNorm() /
                best_poly_coeffs.n_cols);
  std::cout << "Full fit RMS: " << full_fit_rms << '\n';
  std::cout << "RANSAC fit RMS: " << ransac_fit_rms << '\n';

  // Part II: localizing with RANSAC.

  return EXIT_SUCCESS;
}