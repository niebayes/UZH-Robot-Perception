#ifndef UZH_KLT_WARP_IMAGE_H_
#define UZH_KLT_WARP_IMAGE_H_

#include <cmath>

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Warp an image given affine warp matrix.
//@param W_inv Affine warp matrix denoting the inverse warping.
//@param I_r Reference image to be warped.
//@return I_w Warped image.
arma::umat /* I_w */
WarpImage(const arma::umat& I_r, const arma::mat& W_inv) {
  if (I_r.empty() || W_inv.empty()) LOG(ERROR) << "Empty input.";
  if (W_inv.n_rows != 2 || W_inv.n_cols != 3)
    LOG(ERROR) << "Invalid affine warp matrix.";

  arma::umat I_w(arma::size(I_r), arma::fill::zeros);

  //! By convection, x -> col, y -> row.
  for (int x = 0; x < I_w.n_cols; ++x) {
    for (int y = 0; y < I_w.n_rows; ++y) {
      // (x, y) is the warped point, while p is the unwarped point.
      // W is the inverse warping such that W * dst = src.
      const arma::vec2 p = W_inv * arma::Col<int>{x, y, 1};
      // Boundary check.
      if (p(0) < I_r.n_cols && p(1) < I_r.n_rows && p(0) >= 0 && p(1) >= 0) {
        // Simply apply nearest neighbor interpolation to avoid non-integer
        // coordinates.
        I_w(y, x) = I_r(std::floor(p(1)), std::floor(p(0)));
      }
      // If not within the boundary of I_r, the corresponding pixels in I_w is
      // set to zero as what we initialized.
    }
  }

  return I_w;
}

}  // namespace uzh

#endif  // UZH_KLT_WARP_IMAGE_H_