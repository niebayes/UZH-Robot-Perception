#ifndef UZH_TWO_VIEW_GEOMETRY_FUNDAMENTAL_NORMALIZED_8POINT_H_
#define UZH_TWO_VIEW_GEOMETRY_FUNDAMENTAL_NORMALIZED_8POINT_H_

#include <tuple>

#include "armadillo"
#include "glog/logging.h"
#include "two_view_geometry/fundamental_8point.h"
#include "two_view_geometry/normalize_points.h"

namespace uzh {

//@brief Find fundamental matrix F from 2D point correspondences using 8 point
// algorithm with normalized coordinates.
//@param p1 [3 x n] matrix where each column contains the homogeneous
// coordinates of image points p1 on the left camera.
//@param p2 [3 x n] matrix where each column contains the homogeneous
// coordinates of image points p2 on the right camera.
//@return F -- [3 x 3] fundamental matrix encapsulating the two view geometry.
//! F is found by accumulating point correspondences (>= 8) each contributing an
//! independent linear equation involving p1, p2 and F.
//! In the end of the function, a posteriori enforcement is applied to enforce
//! the singularity constraint: det(F) = 0, since F is not full rank
arma::mat /* F */
FundamentalNormalized8Point(const arma::mat& p1, const arma::mat& p2) {
  if (p1.n_cols != p2.n_cols)
    LOG(ERROR) << "Number of points of p1 and p2 must be consistent.";
  if (p1.n_rows != 3 || p2.n_rows != 3)
    LOG(ERROR) << "Points must be represented as homogeneous coordinates.";
  if (p1.n_cols < 8)
    LOG(ERROR) << "Insufficient number of point correspondences.";

  // Normalize the two sets of points.
  arma::mat normalized_p1, normalized_p2, T1, T2;
  std::tie(normalized_p1, T1) = NormalizePoints(p1);
  std::tie(normalized_p2, T2) = NormalizePoints(p2);

  // Obtain F_tilde.
  const arma::mat F_tilde = Fundamental8Point(normalized_p1, normalized_p2);

  // Return F = T2' * F_tilde * T1
  return T2.t() * F_tilde * T1;
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_FUNDAMENTAL_NORMALIZED_8POINT_H_