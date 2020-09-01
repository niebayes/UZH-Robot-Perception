#ifndef UZH_TWO_VIEW_GEOMETRY_ESTIMATE_ESSENTIAL_MATRIX_H_
#define UZH_TWO_VIEW_GEOMETRY_ESTIMATE_ESSENTIAL_MATRIX_H_

#include "armadillo"
#include "glog/logging.h"
#include "two_view_geometry/fundamental_normalized_8point.h"

namespace uzh {

//@brief Estimate the essential matrix given point correspondences and
// calibration matrix K.
//@param p1 [3 x n] matrix where each column contains the homogeneous
// coordinates of image points p1 on the left camera.
//@param p2 [3 x n] matrix where each column contains the homogeneous
// coordinates of image points p2 on the right camera.
//@param K1 [3 x 3] calibration matrix for the left camera.
//@param K2 [3 x 3] calibration matrix for the right camera.
//@return E -- [3 x 3] essential matrix.
arma::mat /* E */
EstimateEssentialMatrix(const arma::mat& p1, const arma::mat& p2,
                        const arma::mat& K1, const arma::mat& K2) {
  if (p1.n_cols != p2.n_cols)
    LOG(ERROR) << "Number of points of p1 and p2 must be consistent.";
  if (p1.n_rows != 3 || p2.n_rows != 3)
    LOG(ERROR) << "Points must be represented as homogeneous coordinates.";
  if (p1.n_cols < 8)
    LOG(ERROR) << "Insufficient number of point correspondences to call "
                  "FundamentalNormalized8point function.";
  if (K1.n_rows != 3 || K1.n_cols != 3 || K2.n_rows != 3 || K2.n_cols != 3)
    LOG(ERROR) << "Invalid calibration matrix.";

  // Obtain fundamental matrix F.
  const arma::mat F = FundamentalNormalized8Point(p1, p2);

  // Incorporate calibration matrices K's and F to obtain E.
  return K2.t().i() * F * K1.i();
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_ESTIMATE_ESSENTIAL_MATRIX_H_