#ifndef UZH_KLT_TRACK_KLT_H_
#define UZH_KLT_TRACK_KLT_H_

#include <tuple>
#include <vector>

#include "arma_traits/arma_homogeneous.h"
#include "armadillo"
#include "glog/logging.h"
#include "klt/get_simple_warp.h"
#include "klt/get_warped_patch.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

namespace uzh {

//@brief Apply Lucas-Kanade tracker to find a general affine warp W between a
// template patch and a warped patch using Gauss-Newton iterative algorithm.
//@param I_r Reference image.
//@param I Warped image to track point / patch in.
//@param x_T [2 x 1] column vector denotes the point to track or the center of
// the patch to track. It's specified as (x, y).
//@param r_T Radius of the patch to track.
//@param num_iterations Maximum number of iterations of the Gauss-Newton
// iterative algorithm.
//@return
// W -- [2 x 3] general affine warp matrix which specifies the forward warping
// from the reference image I_r to the warped image I.
// param_history -- [6 x (k + 1)] matrix where each column contains the 6
// parameters for the general affine warp matrix and k denotes the number of
// iterations used by the Gauss-Newton algorithm to converge. The plus one is
// accounting for the initial identity estimate, aka. no translation, rotation
// and identity scale.
//! To simplify the expression, vector calculus is applied with almost all the
//! involving matrices are vectorized.
std::tuple<arma::mat /* W */, arma::mat /* param_history */> TrackKLT(
    const arma::umat& I_r, const arma::umat& I, const arma::uvec2& x_T,
    const int r_T, const int num_iterations) {
  if (I_r.empty() || I.empty() || x_T.empty()) LOG(ERROR) << "Empty input.";
  if (arma::size(I_r) != arma::size(I))
    LOG(ERROR) << "size(I_r) must be consistent with size(I).";
  if (!arma::all(x_T) || r_T < 0 || r_T % 2 == 0 || num_iterations < 1)
    LOG(ERROR) << "Invalid input.";

  arma::mat param_history(6, num_iterations + 1);
  // Initial estimate of W, an identity affine matrix.
  arma::mat W = uzh::GetSimpleWarp(0, 0, 0, 1);
  // The frist slice of history is the initial estimate.
  param_history.col(0) = arma::vectorise(W);
  // Template patch.
  const arma::umat template_patch = uzh::GetWarpedPatch(I_r, W, x_T, r_T);
  // Apply vectorization trick.
  const arma::uvec i_r = arma::vectorise(template_patch);

  // Quickly get all the dw/dp items by utilizing kronecker product tricks.
  //@note Kronecker product tricks: horizontal concatenation and sequential
  // replication.
  //@ref
  // https://blogs.sas.com/content/iml/2020/07/27/8-ways-kronecker-product.html
  // A x B, where A is a row vector of ones, horizontal concatenation of
  // length(A) copies of B.
  // A x B, where B is a row vector of ones, length(B) copies of the first
  // column of A, followed by length(B) copies of the second column of A, and so
  // forth.
  const int patch_size = 2 * r_T + 1;
  const arma::urowvec xs = arma::linspace<arma::urowvec>(-r_T, r_T, patch_size);
  const arma::urowvec ys = xs;
  const arma::urowvec ones(patch_size, arma::fill::ones);
  const arma::umat xy1 = uzh::homogeneous<arma::uword>(
      arma::join_horiz(arma::kron(xs, ones).t(), arma::kron(ones, ys).t()), 1);
  // dwdp is a [2*patch_size x 6] matrix collecting all the dwdp items evaluated
  // at all pixels through out the patch.
  const arma::umat dwdp = arma::kron(xy1, arma::eye<arma::umat>(2, 2));

  for (int i = 0; i < num_iterations; ++i) {
    
  }
}

}  // namespace uzh

#endif  // UZH_KLT_TRACK_KLT_H_