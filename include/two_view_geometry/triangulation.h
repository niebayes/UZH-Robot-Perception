#ifndef UZH_TWO_VIEW_GEOMETRY_TRIANGULATION_H
#define UZH_TWO_VIEW_GEOMETRY_TRIANGULATION_H

#include "armadillo"
#include "glog/logging.h"

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
// coordinates
// for a 3D scene point P.
arma::mat /* P */
LinearTriangulation(const arma::mat& p1, const arma::mat& p2,
                    const arma::mat& M1, const arma::mat& M2) {
  // assert
}

//@brief Nonlinear triangulation where the 3D scene points P's are computed from
// non-linear least squares by minimizing the geometric error, i.e. the SSRE
// (Sum of Squared Reprojection Error).
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
// coordinates
// for a 3D scene point P.
//! In practice, this function takes as initial estimate the result of
//! LinearTriangulation.
arma::mat /* P */
NonlinearTriangulation(const arma::mat& p1, const arma::mat& p2,
                       const arma::mat& M1, const arma::mat& M2) {
  // assert
  const arma::mat& P_init = LinearTriangulation(p1, p2, M1, M2);
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_TRIANGULATION_H