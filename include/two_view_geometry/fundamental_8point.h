#ifndef UZH_TWO_VIEW_GEOMETRY_FUNDAMENTAL_8POINT_H_
#define UZH_TWO_VIEW_GEOMETRY_FUNDAMENTAL_8POINT_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Find fundamental matrix F from 2D point correspondences using 8 point
// algorithm.
//@param p1 [3 x n] matrix where each column contains the homogeneous
// coordinates of image points p1 on the left camera.
//@param p2 [3 x n] matrix where each column contains the homogeneous
// coordinates of image points p2 on the right camera.
//@return F -- [3 x 3] fundamental matrix encapsulating the two view geometry.
//! F is found by accumulating point correspondences (>= 8) each contributing an
//! independent linear equation involving p1, p2 and F.
//! In the end of the function, a posteriori enforcement is applied to enforce
//! the singularity constraint: det(F) = 0, since F is not full rank.
//@ref https://qr.ae/pNYJw3
arma::mat /* F */
Fundamental8Point(const arma::mat& p1, const arma::mat& p2) {
  if (p1.n_cols != p2.n_cols)
    LOG(ERROR) << "Number of points of p1 and p2 must be consistent.";
  if (p1.n_rows != 3 || p2.n_rows != 3)
    LOG(ERROR) << "Points must be represented as homogeneous coordinates.";
  if (p1.n_cols < 8)
    LOG(ERROR) << "Insufficient number of point correspondences.";

  const int kNumCorrespondences = p1.n_cols;
  arma::mat Q(kNumCorrespondences, 9, arma::fill::zeros);

  // Build a system of linear equations QF = 0.
  // Incrementally populate Q.
  for (int i = 0; i < kNumCorrespondences; ++i) {
    Q.row(i) = arma::kron(p1.col(i), p2.col(i)).t();
  }
  // Solve QF = 0 with SVD where F is the vectorized F.
  arma::vec s;
  arma::mat U, V;
  arma::svd(U, s, V, Q);
  arma::mat F = arma::reshape(V.tail_cols(1), 3, 3);

  // Enforce the singularity constraint. This constraint ensures that all the
  // epipolar lines in the image intersect at a single point, the epipole.
  if (arma::det(F) != 0) {
    // If not singular, project F to F_tilde with the last singular value being
    // set to zero.
    arma::vec s_tilde;
    arma::mat U_tilde, V_tilde;
    arma::svd(U_tilde, s_tilde, V_tilde, F);
    s_tilde.tail(1) = 0;
    const arma::mat F_tilde = U_tilde * arma::diagmat(s_tilde) * V_tilde.t();
    F = F_tilde;
  }

  return F;
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_FUNDAMENTAL_8POINT_H_