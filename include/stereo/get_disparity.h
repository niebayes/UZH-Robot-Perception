#ifndef UZH_STEREO_GET_DISPARITY
#define UZH_STEREO_GET_DISPARITY

#include <tuple>

#include "armadillo"
#include "glog/logging.h"
#include "matlab_port/pdist2.h"

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
                       const double max_disparity) {
  // Assure the sizes of the left_img and the right_img are consistent and not
  // empty.
  if (left_img.empty() || right_img.empty() ||
      left_img.n_rows != right_img.n_rows ||
      left_img.n_cols != right_img.n_cols) {
    LOG(FATAL) << "Empty input image or inconsistent image size.";
  }

  // Determine the search space.
  const int img_rows = left_img.n_rows;
  const int img_cols = left_img.n_cols;
  const int patch_size = 2 * patch_radius + 1;
  // Only the patch_size matters.
  const arma::umat patch(patch_size, patch_size, arma::fill::none);
  // Double type to account for non discretized disparity range.
  const double search_range = max_disparity - min_disparity + 1;
  // The (patch_size - 1) accounts for the first search region from right to
  // left.
  const arma::umat strip(patch_size, search_range + patch_size - 1,
                         arma::fill::none);

  // Construct disparity map to be assigned.
  arma::mat disp_map(img_rows, img_cols, arma::fill::zeros);

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
      for (int i = 0; i < static_cast<int>(search_range); ++i) {
        right_strip_stack.col(i) =
            arma::vectorise(right_strip(0, i, arma::size(patch)));
      }

      // Compute patch-wise SSD.
      arma::mat ssds;
      std::cout << arma::conv_to<arma::mat>::from(left_patch_vec) << '\n';
      // std::tie(ssds, std::ignore) =
      //     uzh::pdist2(arma::conv_to<arma::mat>::from(left_patch_vec),
      //                 arma::conv_to<arma::mat>::from(right_strip_stack),
      //                 uzh::SQUARED_EUCLIDEAN);
    }
  }

  return disp_map;
}

#endif  // UZH_STEREO_GET_DISPARITY