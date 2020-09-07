#ifndef UZH_KLT_GET_WARPED_PATCH_H_
#define UZH_KLT_GET_WARPED_PATCH_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Warp a patch of an image.
//@param I Image to track point / patch in.
//@param W_inv [2 x 3] affine warp matrix denoting inverse warping.
//@param x_T [2 x 1] column vector denoting the center of the patch, specified
// as (x, y).
//@param r_T Patch radius, positive odd integer number.
//@return warped_patch -- [2*r_T+1 x 2*r_T+1] patch warped by W.
arma::umat /* warped_patch */
GetWarpedPatch(const arma::umat& I, const arma::mat& W_inv,
               const arma::vec2 x_T, const int r_T) {
  if (I.empty() || W_inv.empty()) LOG(ERROR) << "Empty input.";
  if (W_inv.n_rows != 2 || W_inv.n_cols != 3)
    LOG(ERROR) << "Affine warp matrix should be a [2 x 3] matrix.";
  if (r_T <= 0) LOG(ERROR) << "Invalid patch radius.";
  // if (r_T % 2 == 0) LOG(WARNING) << "Odd-size patch radius.";

  const int patch_size = 2 * r_T + 1;
  arma::umat warped_patch(patch_size, patch_size, arma::fill::zeros);

  for (int x = -r_T; x <= r_T; ++x) {
    for (int y = -r_T; y <= r_T; ++y) {
      // (x, y) is the warped point, while p is the unwarped point.
      // And warped(y, x) = unwarped(v, u). The u, v will be replaced with the
      // interpolated value later on.
      // W is the inverse warping such that W * dst = src.
      const arma::vec2 p =
          W_inv * arma::Col<double>{static_cast<double>(x),
                                    static_cast<double>(y), 1.0} +
          x_T;
      // p.t().print("p");
      // Boundary check.
      if (p(0) + 1 < I.n_cols && p(1) + 1 < I.n_rows && p(0) >= 0 &&
          p(1) >= 0) {
        // Apply bilinear interpolation.
        const int u = std::floor(p(0)), v = std::floor(p(1));
        const double a = p(0) - u, b = p(1) - v;
        warped_patch(y + r_T, x + r_T) = arma::as_scalar(
            arma::rowvec2{1 - a, a} *
            arma::umat22{I(v, u), I(v, u + 1), I(v + 1, u), I(v + 1, u + 1)} *
            arma::vec2{1 - b, b});
      }
    }
  }

  return warped_patch;
}

}  // namespace uzh

#endif  // UZH_KLT_GET_WARPED_PATCH_H_