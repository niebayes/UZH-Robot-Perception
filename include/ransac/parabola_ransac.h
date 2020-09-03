#ifndef UZH_RANSAC_PARABOLA_RANSAC_H_
#define UZH_RANSAC_PARABOLA_RANSAC_H_

#include <tuple>

#include "armadillo"
#include "glog/logging.h"
#include "matlab_port/datasample.h"

namespace uzh {

//@brief Fit a parabola with outlier rejection performed by RANSAC.
//@param data [2 x n] matrix where each column contains a 2D data point (x, y).
//@param max_noise Inliers in the data are contaminated with noise along y
// direction with the maximum extent max_noise. This is the indicator through
// which a data point is selected as an inlier or outlier.
//@param num_iterations RANSAC proceeds until num_iterations iterations are
// reached.
//@returns
// best_polynome_coefficients -- [3 x k] matrix containing all polynome
// coefficients evaluated by polyfit best ever so far at each iteration
// i = 1, 2, ..., k.
// max_inlier_counts -- [1 x k] row vector containing the inlier counts
// corresponding to the polynome coeffcients best ever so far at each iteration
// i = 1, 2, ..., k.
std::tuple<arma::mat /* best_polynome_coefficients */,
           arma::urowvec /* max_inlier_counts */>
ParabolaRANSAC(const arma::mat& data, const double max_noise,
               const int num_iterations = 100) {
  if (data.empty()) LOG(ERROR) << "Empty data.";
  if (max_noise < 0 || num_iterations < 0)
    LOG(ERROR) << "max_noise and num_iterations shall not be negative.";

  const bool refine_with_inliers = true;

  // Matrices to be returned.
  arma::mat best_polynome_coefficients(3, num_iterations);
  arma::urowvec max_inlier_counts(num_iterations);

  arma::vec3 best_poly_coeffs(arma::fill::zeros);
  int max_inlier_cnt = 0;
  for (int i = 0; i < num_iterations; ++i) {
    // Randomly draw 3 samples without replacement.
    arma::mat samples;
    std::tie(samples, std::ignore) = uzh::datasample<double>(data, 3, 1, false);
    // Fit a model with these samples and get the coefficients of the model.
    arma::vec poly_coeffs = arma::polyfit(samples.row(0), samples.row(1), 2);
    // Compute the residuals.
    const arma::rowvec residuals =
        arma::abs(arma::polyval(poly_coeffs, data.row(0)) - data.row(1));
    // Obtain the inliers' indices.
    // Plus 1e-5 to account for the case where max_noise = 0.
    const arma::uvec inlier_indices = arma::find(residuals < max_noise + 1e-5);
    // Count inliers.
    const int inlier_cnt = inlier_indices.size();

    if (inlier_cnt > max_inlier_cnt) {
      max_inlier_cnt = inlier_cnt;
      // Refine the model if more inliers are obtained.
      if (refine_with_inliers) {
        poly_coeffs = arma::polyfit(data(arma::uvec{0}, inlier_indices),
                                    data(arma::uvec{1}, inlier_indices), 2);
      }
      best_poly_coeffs = poly_coeffs;
    }
    // Record the best model coefficients and inlier counts so far.
    best_polynome_coefficients.col(i) = best_poly_coeffs;
    max_inlier_counts(i) = max_inlier_cnt;
  }

  return {best_polynome_coefficients, max_inlier_counts};
}

}  // namespace uzh

#endif  // UZH_RANSAC_PARABOLA_RANSAC_H_