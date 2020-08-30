#ifndef UZH_STEREO_GET_DISPARITY
#define UZH_STEREO_GET_DISPARITY

#include <tuple>  // std::tie

#include "armadillo"
#include "glog/logging.h"
#include "matlab_port/pdist2.h"

namespace stereo {

// TODO(bayes) Parallelise the function.
//@ref https://stackoverflow.com/a/45773308/14007680
//@ref
// https://software.intel.com/content/www/us/en/develop/tools/compilers/c-compilers.html
//@ref
// https://devblogs.microsoft.com/cppblog/using-c17-parallel-algorithms-for-better-performance/

//@brief Find the disparity for pixels in left image based on measuring the
// patch-wise similarity between the patches in either images.
//@param left_img Left image in which all defined pixels are goint to be
// assigned a disparity value. The term "defined pixels" refer to those that
// conform to certain rules.
//@param right_img Right image in which, for each defined pixel in the left
// image, a strip of patches are used to compute the patch-wise similarities
// based on SSD measure.
//@param patch_radius Radius of the patch used to compute the patch-wise SSD.
// The size of the patch is computed as patch_size = 2 * patch_radius + 1.
//@param min_disparity Value specifies the lower bound below which the disparity
// is rejected. This also attenuate the effect of noise.
//@param max_disparity Value specifies the upper bound exceed which the
// disparity is also rejected. This also bounds the search space and thus the
// computation effort.
//! In this function we treat the disparity as a shift between pixels in either
//! images which is sign-free. So the the coordinates of the pixel p_r in the
//! right image is computed as p_r = p_l - [d; 0; 0] where p_l is the
//! coordinates of the corresponding pixel in the left image.
// For an outline of the method this function adopted to find disparity,
//@ref
// https://en.wikipedia.org/wiki/Binocular_disparity#Computing_disparity_using_digital_stereo_images
arma::mat GetDisparity(const arma::umat& left_img, const arma::umat& right_img,
                       const int patch_radius, const double min_disparity,
                       const double max_disparity,
                       const bool reject_outliers = true,
                       const bool refine_subpixel = true) {
  // Assure the sizes of the left_img and the right_img are consistent and not
  // empty.
  if (left_img.empty() || right_img.empty() ||
      left_img.n_rows != right_img.n_rows ||
      left_img.n_cols != right_img.n_cols) {
    LOG(FATAL) << "Empty input image or inconsistent image sizes.";
  }

  // Determine the search space.
  const int img_rows = left_img.n_rows;
  const int img_cols = left_img.n_cols;
  const int patch_size = 2 * patch_radius + 1;
  // Only the patch_size matters.
  const arma::umat patch(patch_size, patch_size, arma::fill::none);
  // Double type to account for non discretized disparity range.
  const int search_range = max_disparity - min_disparity + 1;
  // The (patch_size - 1) accounts for the first search region from right to
  // left.
  const arma::umat strip(patch_size, search_range + patch_size - 1,
                         arma::fill::none);

  // Construct disparity map to be assigned.
  arma::mat disparity_map(img_rows, img_cols, arma::fill::zeros);

  // Debug switch.
  bool debug_ssds = false;

  // For each defined pixel in the left image, match the strip of pixels in the
  // right image. The undefined pixels are those on borders which involve pixels
  // out of the image boundary, and those start from which the search range may
  // exceed the image boundary.
  //! row and col denotes the upper left corner of the corresponding patch.
  for (int row = 0; row < img_rows - patch_size + 1; ++row) {
    for (int col = max_disparity; col < img_cols - patch_size + 1; ++col) {
      // For each (col + patch_radius, row + patch_radius) pixel in the left
      // image, construct a [patch_size x patch_size] patch and match it against
      // the whole strip of patches in the right image along the epipolar line.
      const arma::umat left_patch = left_img(row, col, arma::size(patch));
      //! Note the search region always on the left of the corresponding pixel
      //! in the right image, despite the regions used to compute the few "right
      //! most" pixels.
      const arma::umat right_strip =
          right_img(row, col - max_disparity, arma::size(strip));

      // For use of uzh::pdist2 to speed up computation, vectorize the strip of
      // patches and stack them to a big matrix against which the vectorized
      // left_patch is matched.
      const arma::umat left_patch_vec = arma::vectorise(left_patch);
      arma::umat right_strip_stack(left_patch_vec.n_rows, search_range,
                                   arma::fill::zeros);
      //! Note stacked in this way, the index i denotes the so-called "negative
      //! disparity" such that disparity = max_disparity - i.
      for (int i = 0; i < search_range; ++i) {
        right_strip_stack.col(i) =
            arma::vectorise(right_strip(0, i, arma::size(patch)));
      }

      // Compute patch-wise SSD.
      arma::mat ssds;
      std::tie(ssds, std::ignore) =
          uzh::pdist2(arma::conv_to<arma::mat>::from(left_patch_vec),
                      arma::conv_to<arma::mat>::from(right_strip_stack),
                      uzh::SQUARED_EUCLIDEAN);

      // Optional, debug ssds by ploting
      if (debug_ssds) {
        std::vector<cv::Mat> plots(2);
        plots[0] = uzh::imagesc(left_patch, false);
        plots[1] = uzh::imagesc(right_strip, false);
        cv::imshow("Left patch and right strip (Hold any key to move)",
                   uzh::MakeCanvas(plots, left_img.n_rows, 1));
        char key = cv::waitKey(0);
        if (key == 27) {  // 'ESC' key -> exit.
          cv::destroyAllWindows();
          debug_ssds = false;
        }
      }

      // Find the disparity.
      const arma::uword negative_disparity = arma::index_min(ssds.as_row());
      double disparity =
          max_disparity - static_cast<double>(negative_disparity);

      // Optional, reject outliers to disambiguous matching.
      if (reject_outliers) {
        // Potential disparity will be rejected if there're more than 2
        // disparity candidates have the SSD values smaller than 1.5 * min_ssd,
        // or if it's the max_disparity or min_disparity where the true
        // disparity may outside the search range.
        if (arma::size(arma::find(ssds <= 1.5 * ssds.min())).n_rows > 2 ||
            negative_disparity == 0 || negative_disparity == ssds.n_cols - 1) {
          // If outlier, the disparity is left to zero.
          continue;
        }

        // Optional, refine discretized disparity to subpixel accuracy.
        // This refinement is only applied to inliers.
        if (refine_subpixel) {
          // A simple refinement technique is applied: a second-degree
          // polynomial fit is applied on the neighbors of the original
          // disparity, and the disparity at which the SSD is the smallest along
          // the quadratic curve is selected as the final disparity.
          const double neg_disp = static_cast<double>(negative_disparity);
          const arma::vec x{neg_disp - 1, neg_disp, neg_disp + 1};
          const arma::vec p =
              arma::polyfit(x, ssds(arma::conv_to<arma::uvec>::from(x)), 2);
          // - p(1) / (2 * p(0)) denotes the axis of symmetry at which the
          // smallest SSD is obtained.
          disparity_map(row + patch_radius, col + patch_radius) =
              max_disparity + p(1) / (2 * p(0));
        } else {
          // The shift by patch_radius is to move the upper left corner to the
          // anchor point.
          disparity_map(row + patch_radius, col + patch_radius) = disparity;
        }
      } else
        disparity_map(row + patch_radius, col + patch_radius) = disparity;
    }  // col
  }    // row

  return disparity_map;
}

}  // namespace stereo

#endif  // UZH_STEREO_GET_DISPARITY