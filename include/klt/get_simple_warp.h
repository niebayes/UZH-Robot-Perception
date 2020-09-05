#ifndef UZH_KLT_GET_SIMPLE_WARP_H_
#define UZH_KLT_GET_SIMPLE_WARP_H_

#include <cmath>

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Return a affine warp matrix given translation, rotation angle and
// scale.
//@param dx Displacement in x direction.
//@param dy Displacement in y direction.
//@param theta Rotation angle, specified with degrees.
//@param s Scale factor.
//@return W -- [2 x 3] affine warp matrix such that the warped 2D point is given
// by p_w = W * p where p and p_w are 2D homogeneous coordinates.
//! Note, W is used with inverse warping through out the KLT pipeline we
//! implemented.
arma::mat GetSimpleWarp(const double dx, const double dy, const double theta,
                        const double s) {
  if (s <= 0)
    LOG(ERROR) << "Scale factor should not be less than or equal to zero.";

  // Translation vector.
  const arma::vec2 t{dx, dy};
  // Rotation matrix.
  const double radians = theta * arma::datum::pi / 180.0;
  const arma::mat22 R{{std::cos(radians), -std::sin(radians)},
                      {std::sin(radians), std::cos(radians)}};
  // Affine warp matrix
  const arma::mat W = s * arma::join_horiz(R, t);

  return W;
}

}  // namespace uzh

#endif  // UZH_KLT_GET_SIMPLE_WARP_H_