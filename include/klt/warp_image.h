#ifndef UZH_KLT_WARP_IMAGE_H_
#define UZH_KLT_WARP_IMAGE_H_

#include <cmath>

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Warp an image given affine warp matrix.
//@param W Affine warp matrix denoting the inverse warping.
//@param I_r Reference image to be warped.
//@return I_w Warped image.
arma::umat WarpImage(const arma::umat& I_r, const arma::mat& W) {
  if (I_r.empty() || W.empty()) LOG(ERROR) << "Empty input.";
  if (W.n_rows != 2 || W.n_cols != 3)
    LOG(ERROR) << "Invalid affine warp matrix.";

  arma::umat I_w(arma::size(I_r), arma::fill::zeros);

  //! By convection, x -> col, y -> row.
  for (int x = 0; x < I_w.n_cols; ++x) {
    for (int y = 0; y < I_w.n_rows; ++y) {
      // Warped 2D point.
      const arma::vec2 p_w = W * arma::Col<int>{x, y, 1};
      // Boundary check.
      if (p_w(0) < I_r.n_cols && p_w(1) < I_r.n_rows && p_w(0) >= 0 &&
          p_w(1) >= 0) {
        // Simply apply nearest neighbor interpolation to avoid non-integer
        // coordinates.
        I_w(y, x) = I_r(std::floor(p_w(1)), std::floor(p_w(0)));
      }
      // If not within the boundary of I_r, the corresponding pixels in I_w is
      // set to zero as what we initialized.
    }
  }

  return I_w;
}

}  // namespace uzh

#endif  // UZH_KLT_WARP_IMAGE_H_