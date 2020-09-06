#ifndef UZH_KLT_TRACK_KLT_ROBUST_H_
#define UZH_KLT_TRACK_KLT_ROBUST_H_

#include <tuple>

#include "armadillo"
#include "glog/logging.h"
#include "klt/track_klt.h"

namespace uzh {

//@brief Robustly do KLT tracking via thresholding on bidirectional error.
//@param I_r Reference image.
//@param I Warped image to track point / patch in.
//@param x_T [2 x 1] column vector denotes the point to track or the center
// of the patch to track. It's specified as (x, y).
//@param r_T Radius of the patch to track.
//@param num_iterations Maximum number of iterations of the Gauss-Newton
// iterative algorithm.
//@param lambda Threshold for bidirectional error.
//@return
// delta_t -- [2 x 1] column vector representing the translational
// delta by which the keypoint has moved from the reference image to the warped
// image.
// is_kept -- True if the KLT tracking has passed the bidirectional error test.
std::tuple<arma::vec2 /* delta_t */, bool> TrackKLTRobust(
    const arma::umat& I_r, const arma::umat& I, const arma::uvec2& x_T,
    const int r_T, const int num_iterations, const double lambda) {
  if (I_r.empty() || I.empty() || x_T.empty()) LOG(ERROR) << "Empty input.";
  if (arma::size(I_r) != arma::size(I))
    LOG(ERROR) << "size(I_r) must be consistent with size(I).";
  if (!arma::all(x_T) || r_T < 0 || r_T % 2 == 0 || num_iterations < 1 ||
      lambda < 0)
    LOG(ERROR) << "Invalid input.";

  arma::mat W, W_inv;
  std::tie(W, std::ignore) = uzh::TrackKLT(I_r, I, x_T, r_T, num_iterations);
  const arma::vec2 delta_t = W.tail_cols(1);
  std::tie(W_inv, std::ignore) =
      uzh::TrackKLT(I_r, I, x_T + arma::conv_to<arma::uvec2>::from(delta_t),
                    r_T, num_iterations);
  const arma::vec2 delta_t_inv = W_inv.tail_cols(1);
  const bool is_kept = arma::norm(delta_t + delta_t_inv) < lambda;

  return {delta_t, is_kept};
}

}  // namespace uzh

#endif  // UZH_KLT_TRACK_KLT_ROBUST_H_