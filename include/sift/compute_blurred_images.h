#ifndef UZH_SIFT_COMPUTE_BLURRED_IMAGES_H_
#define UZH_SIFT_COMPUTE_BLURRED_IMAGES_H_

#include <cmath>

#include "armadillo"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "transfer/cv2arma.h"

namespace uzh {

//@brief Compute blurred images for all images in the image pyramid.
//! Images in a certain octave are blurred with Gaussians of different sigmas.
//! Images in different octaves are blurred with the same set of Gaussians.
//@param image_pyramid Image pyramid containing set of original images to be
// blurred.
//@param num_scales Number of scales per octave from which the number of images
// per octave is computed.
//@param sigma Base sigma from which the set of sigmas used to get different
// Gaussians are generated.
arma::field<arma::cube> ComputeBlurredImages(
    const arma::field<cv::Mat>& image_pyramid, const int num_scales,
    const double base_sigma) {
  const int kNumOctaves = image_pyramid.size();
  // This formula is by observing that the each scale is formed from three DoG
  // images each of which is obtained from two Gaussian blurred images. Hence
  // the plus 3.
  const int kImagesPerOctave = num_scales + 3;
  arma::field<arma::cube> blurred_images(kNumOctaves);

  // Populate each octave with kImagesPerOctave images.
  for (int o = 0; o < kNumOctaves; ++o) {
    arma::cube octave = arma::zeros<arma::cube>(
        image_pyramid(o).rows, image_pyramid(o).cols, kImagesPerOctave);
    // Gaussian blur images in an octave with increasing sigmas.
    for (int i = 0; i < kImagesPerOctave; ++i) {
      // Such that s = [-1, 0, ..., num_scales + 1], 6 indices in total.
      // FIXME This range could also be changed,
      // e.g. s = [0, ..., num_scales + 2].
      cv::Mat blurred_image;
      const int s = i - 1;
      const double sigma = std::pow(2, s / (double)num_scales) * base_sigma;
      cv::GaussianBlur(image_pyramid(o), blurred_image, {}, sigma, sigma,
                       cv::BORDER_ISOLATED);
      octave.slice(i) = uzh::cv2arma<double>(blurred_image).t();
    }
    blurred_images(o) = octave;
  }

  return blurred_images;
}

}  // namespace uzh

#endif  // UZH_SIFT_COMPUTE_BLURRED_IMAGES_H_