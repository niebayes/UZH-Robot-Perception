#ifndef UZH_SIFT_COMPUTE_IMAGE_PYRAMID_H_
#define UZH_SIFT_COMPUTE_IMAGE_PYRAMID_H_

#include "armadillo"
#include "glog/logging.h"
#include "matlab_port/imresize.h"
#include "opencv2/core.hpp"

//@brief Compute image pyramid by recursively decimating the original image.
//! This function generates original images of all octaves. That is
//! the resolutions of the original image in octave o and the original
//! image in octave o+1 differ by a factor of 2, i.e. the lower octave is
//! downsampled by a factor of 2.
//@param Image to be downsampled.
//@param num_octaves Number of octaves contained in this pyramid. An octave is
// nothing but a level of the pyramid.
//@return A image pyramid containing five images with different resolutions.
arma::field<cv::Mat> ComputeImagePyramid(const cv::Mat& image,
                                         const int num_octaves) {
  if (image.empty()) LOG(ERROR) << "Empty input image.";
  if (num_octaves <= 1) LOG(ERROR) << "Invalid num_octaves.";
  arma::field<cv::Mat> image_pyramid(num_octaves);
  image_pyramid(0) = image;
  for (int o = 1; o < num_octaves; ++o) {
    // Downsample by a factor of 2.
    image_pyramid(o) = uzh::imresize(image_pyramid(o - 1), 0.5);
  }
  return image_pyramid;
}

#endif  // UZH_SIFT_COMPUTE_IMAGE_PYRAMID_H_