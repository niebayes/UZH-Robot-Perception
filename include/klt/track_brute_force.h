#ifndef UZH_KLT_TRACK_BRUTE_FORCE_H_
#define UZH_KLT_TRACK_BRUTE_FORCE_H_

#include <tuple>

#include "armadillo"
#include "glog/logging.h"
#include "klt/get_simple_warp.h"
#include "klt/warp_image.h"

namespace uzh {

//@brief Find a warp W which minimizes the sum of square difference (SSD)
// between the template and a warped patch from the image in which the template
// is supposed to be tracked. The finding is implemented as a brute-force
// manner.
//@param I_r Reference image.
//@param I Warped image to track point / patch in.
//@param x_T [2 x 1] column vector denotes the point to track or the center of
// the patch to track. It's specified as (x, y).
//@param r_T Radius of the patch to track.
//@param r_D Radius of the region to search d within.
//@return
// d -- [2 x 1] column vector denoting the displacement vector wrt. the
// reference image. I.e. it specifies the forward warping from the reference
// image I_r to the warped image I. ssds -- [2*r_D+1 x 2*r_D+1] matrix where
// each entry stores the corresponding SSD measurement between the template
// patch and the candidate patch extracted around x_T after warping specified
// with W.
//! This function assumes the warping is a translation-only one.
std::tuple<arma::vec2 /* d */, arma::mat /* ssds */> TrackBruteForce(
    const arma::umat& I_r, const arma::umat& I, const arma::uvec2& x_T,
    const int r_T, const int r_D) {
  if (I_r.empty() || I.empty() || x_T.empty()) LOG(ERROR) << "Empty input.";
  if (arma::size(I_r) != arma::size(I))
    LOG(ERROR) << "size(I_r) should be consistent with size(I).";
  if (!arma::all(x_T) || r_T < 0 || r_T % 2 == 0 || r_D < 0)
    LOG(ERROR) << "Invalid input.";

  const int search_size = 2 * r_D + 1;
  arma::mat ssds(search_size, search_size, arma::fill::zeros);

  // Template to track.
  const arma::umat template_patch =
      uzh::GetWarpedPatch(I_r, uzh::GetSimpleWarp(0, 0, 0, 1), x_T, r_T);

  for (int dx = -r_D; dx <= r_D; ++dx) {
    for (int dy = -r_D; dy <= r_D; ++dy) {
      const arma::umat candidate_patch =
          uzh::GetWarpedPatch(I, uzh::GetSimpleWarp(dx, dy, 0, 1), x_T, r_T);
      const double ssd =
          arma::accu(arma::square(template_patch - candidate_patch));
      ssds(dx + r_D, dy + r_D) = ssd;
    }
  }

  const arma::uword min_idx = ssds.index_min();
  const arma::uvec sub = arma::ind2sub(arma::size(ssds), min_idx);
  const arma::vec2 d = sub - arma::uvec2{r_D, r_D};

  return {d, ssds};
}

}  // namespace uzh

#endif  // UZH_KLT_TRACK_BRUTE_FORCE_H_