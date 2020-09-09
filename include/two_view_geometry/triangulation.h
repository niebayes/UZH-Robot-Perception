#ifndef UZH_TWO_VIEW_GEOMETRY_TRIANGULATION_H
#define UZH_TWO_VIEW_GEOMETRY_TRIANGULATION_H

#include "armadillo"
#include "glog/logging.h"
#include "transform/hat.h"

namespace uzh {

//@brief Linear triangulation where the 3D scene points P's are computed from
// DLT (Direct Linear Transform) by minimizing the algebraic error.
//@param p1 [3 x n] matrix where each column contains the homogeneous
// coordinates for an image point p_i on the left camera.
//@param p2 [3 x n] matrix where each column contains the homogeneous
// coordinates for an image point p_j on the right camera. The image points p_i
// and p_j are the correspondent projections of a same 3D scene point P.
//@param M1 [3 x 4] matrix representing the projection matrix K[I|0] for the
// left camera, where the I is a [3 x 3] identity matrix and 0 is a [3 x 1]
// vector.
//@param M2 [3 x 4] matrix representing the projection matrix K[R|t] where it is
// assumed the calibration matrices K1 = K2 = K for the left and right camera
// respectively.
//@return P -- [4 x n] matrix where each column contains the homogeneous
// coordinates for a 3D scene point P.
//! The triangulated 3D scene points are in the left camera frame.
arma::mat /* P */
LinearTriangulation(const arma::mat& p1, const arma::mat& p2,
                    const arma::mat& M1, const arma::mat& M2) {
  if (p1.n_cols != p2.n_cols)
    LOG(ERROR) << "Number of points of p1 and p2 must be consistent.";
  if (p1.n_rows != 3 || p2.n_rows != 3)
    LOG(ERROR) << "Points must be represented as homogeneous coordinates.";
  if (M1.n_rows != 3 || M1.n_cols != 4 || M2.n_rows != 3 || M2.n_cols != 4)
    LOG(ERROR) << "Invalid projection matrix.";

  const int kNumPoints = p1.n_cols;

  // Construct matrix P to be populated.
  arma::mat P(4, kNumPoints, arma::fill::zeros);

  for (int i = 0; i < kNumPoints; ++i) {
    // For each pair of point correspondences, find the best P in the least
    // square sense by solving AP = 0 where A is a [6 x 4] matrix obtained by
    // stacking two systems of equations each contributed by an image point p_i
    // and its corresponding projection matrix M_i.
    arma::mat A(6, 4, arma::fill::zeros);
    A.head_rows(3) = uzh::hat<double>(p1.col(i)) * M1;
    A.tail_rows(3) = uzh::hat<double>(p2.col(i)) * M2;
    arma::vec s;
    arma::mat U, V;
    arma::svd(U, s, V, A);

    // Populate P.
    P.col(i) = V.tail_cols(1);
  }

  // Dehomogenize, i.e. divide all elements by the last element
  // along each column for every point in P.
  P /= arma::repmat(P.tail_rows(1), 4, 1);

  return P;
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_TRIANGULATION_H