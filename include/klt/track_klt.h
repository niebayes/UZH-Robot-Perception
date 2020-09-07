#ifndef UZH_KLT_TRACK_KLT_H_
#define UZH_KLT_TRACK_KLT_H_

#include <tuple>
#include <vector>

#include "arma_traits/arma_homogeneous.h"
#include "armadillo"
#include "glog/logging.h"
#include "klt/get_simple_warp.h"
#include "klt/get_warped_patch.h"
#include "matlab_port/conv2.h"
#include "matlab_port/imagesc.h"
#include "matlab_port/imresize.h"
#include "matlab_port/subplot.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "transfer/arma2img.h"

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
//@param visualize If true, the process of Gauss-Newton method will be
// visualized. Toggle this when debugging.
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
    const arma::umat& I_r, const arma::umat& I, const arma::vec2& x_T,
    const int r_T, const int num_iterations, const bool visualize = false) {
  if (I_r.empty() || I.empty() || x_T.empty()) LOG(ERROR) << "Empty input.";
  if (arma::size(I_r) != arma::size(I))
    LOG(ERROR) << "size(I_r) must be consistent with size(I).";
  if (!arma::all(x_T) || r_T < 0 || r_T % 2 == 0 || num_iterations < 1)
    LOG(ERROR) << "Invalid input.";

  arma::mat param_history(6, num_iterations + 1);
  // Initial estimate of W, an identity affine matrix.
  arma::mat W = uzh::GetSimpleWarp(0.0, 0.0, 0.0, 1.0);
  // The frist slice of history is the initial estimate.
  param_history.col(0) = arma::vectorise(W);
  // Template patch.
  const arma::umat I_r_patch = uzh::GetWarpedPatch(I_r, W, x_T, r_T);
  // Apply vectorization trick.
  const arma::vec i_r =
      arma::conv_to<arma::vec>::from(arma::vectorise(I_r_patch));

  // Quickly get all the dw/dp items by utilizing kronecker product tricks.
  //@note Kronecker product tricks: horizontal concatenation and sequential
  // replication.
  //@ref
  // https://blogs.sas.com/content/iml/2020/07/27/8-ways-kronecker-product.html
  // A x B, where A is a row vector of ones, denotes horizontal concatenation of
  // length(A) copies of B.
  // A x B, where B is a row vector of ones, denotes length(B) copies of the
  // first column of A, followed by length(B) copies of the second column of A,
  // and so forth.
  const int patch_size = 2 * r_T + 1;
  const arma::rowvec xs = arma::linspace<arma::rowvec>(
      -static_cast<double>(r_T), static_cast<double>(r_T), patch_size);
  const arma::rowvec ys = xs;
  const arma::rowvec ones(patch_size, arma::fill::ones);
  const arma::mat xy1 = uzh::homogeneous<double>(
      arma::join_horiz(arma::kron(xs, ones).t(), arma::kron(ones, ys).t()), 1);
  // dwdp is a [2*patch_size x 6] matrix collecting all the dwdp items evaluated
  // at all pixels through out the patch.
  const arma::mat dwdp = arma::kron(xy1, arma::eye<arma::mat>(2, 2));

  for (int k = 0; k < num_iterations; ++k) {
    //! Warping is very sensitive to noise as well as small shift within few
    //! pixels. Hence, to get the most accurate gradient, instead of padding by
    //! replication or zero-padding, it's better to in advance extract a bigger
    //! patch to get the best padding wrt. the computation of gradient.
    const arma::umat I_w_patch_big = uzh::GetWarpedPatch(I, W, x_T, r_T + 1);
    const arma::umat I_w_patch =
        I_w_patch_big(1, 1, arma::size(patch_size, patch_size));
    const arma::vec i =
        arma::conv_to<arma::vec>::from(arma::vectorise(I_w_patch));

    // Obtain di/dp.
    // First, compute the x and y gradients with Sobel operators.
    const arma::mat I_tmp = arma::conv_to<arma::mat>::from(I_w_patch_big);
    const arma::mat I_w_patch_grad_x = uzh::conv2(
        {1, 0, -1}, {1}, I_tmp(1, 0, arma::size(patch_size, patch_size + 2)),
        uzh::VALID);
    const arma::mat I_w_patch_grad_y = uzh::conv2(
        {1}, {1, 0, -1}, I_tmp(0, 1, arma::size(patch_size + 2, patch_size)),
        uzh::VALID);
    // Second, get di/dw.
    const arma::vec ix = arma::vectorise(I_w_patch_grad_x),
                    iy = arma::vectorise(I_w_patch_grad_y);
    const arma::mat didw = arma::join_horiz(ix, iy);
    // Multiply didw with dwdp row-blockwisely to get didp.
    arma::mat didp(patch_size * patch_size, 6, arma::fill::zeros);
    for (int j = 0; j < didp.n_rows; ++j) {
      didp.row(j) = didw.row(j) * dwdp.rows(2 * j, 2 * j + 1);
    }

    // Obtain Hessian.
    const arma::mat H = didp.t() * didp;

    // Get delta p.
    //! Use solve instead of inverse to tackle singular issues.
    const arma::vec delta_p = arma::solve(H, didp.t() * (i_r - i));

    // Update W.
    W += arma::reshape(delta_p, 2, 3);

    // Record W.
    param_history.col(k + 1) = arma::vectorise(W);

    // Intentionally hide this option.
    if (visualize) {
      // Auxiliary items for better visualization effect.
      const int num_plots_per_row = 6;
      //! Upsample by a factor of s making visualization better.
      const double s = 4;
      const cv::Mat plot_place_holder = uzh::arma2img(
          arma::umat(s * patch_size, s * patch_size, arma::fill::zeros));

      // Plot reference patch, wraped patch and their difference.
      std::vector<cv::Mat> patches(num_plots_per_row, plot_place_holder);
      patches[2] = uzh::imresize(uzh::arma2img(I_r_patch), s);
      patches[3] = uzh::imresize(uzh::arma2img(I_w_patch), s);
      patches[4] = uzh::imresize(uzh::arma2img(I_r_patch - I_w_patch), s);
      const cv::Mat patches_plot =
          uzh::imagesc(uzh::MakeCanvas(patches, s * patch_size, 1), false);

      // Plot gradients of warped patch.
      std::vector<cv::Mat> gradients(num_plots_per_row, plot_place_holder);
      gradients[2] = uzh::imresize(
          uzh::arma2img(arma::conv_to<arma::umat>::from(I_w_patch_grad_x)), s);
      gradients[3] = uzh::imresize(
          uzh::arma2img(arma::conv_to<arma::umat>::from(I_w_patch_grad_y)), s);
      const cv::Mat gradients_plot =
          uzh::imagesc(uzh::MakeCanvas(gradients, s * patch_size, 1), false);

      // Plot steepest descent patches for visualizing the process of
      // Gauss-Newton method.
      // Each patch corresponds to a parameter of W. In particular, the last two
      // patches corresponds to the two translation parameters.
      std::vector<cv::Mat> descent_patches(num_plots_per_row,
                                           plot_place_holder);
      for (int patch_idx = 0; patch_idx < 6; ++patch_idx) {
        descent_patches[patch_idx] = uzh::imresize(
            uzh::arma2img(arma::conv_to<arma::umat>::from(
                arma::reshape(didp.col(patch_idx), patch_size, patch_size))),
            s);
      }
      const cv::Mat descent_patches_plot = uzh::imagesc(
          uzh::MakeCanvas(descent_patches, s * patch_size, 1), false);

      // Display all plots.
      cv::Mat display;
      cv::Mat all_plots[3] = {patches_plot, gradients_plot,
                              descent_patches_plot};
      cv::vconcat(all_plots, 3, display);
      cv::imshow(
          "Top to bottom: patches, gradient patches, steepest descent patches",
          display);
      const char key = cv::waitKey(50);
      if (key == 32) cv::waitKey(0);  // 'Space' key -> pause.
    }

    // Check if converged.
    const double converge_threshold = 1e-3;
    if (arma::norm(delta_p) < converge_threshold) {
      // Only keep records till now.
      param_history = param_history.head_cols(k + 2);
      LOG(INFO) << "Gradient descent converged in " << k + 1 << " iterations.";
      break;
    }
    if (k == 49) {
      LOG(INFO) << "Gradient descent failed to converge.";
    }
  }

  return {W, param_history};
}

}  // namespace uzh

#endif  // UZH_KLT_TRACK_KLT_H_