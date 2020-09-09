#ifndef UZH_STEREO_GET_IMAGE_STEREO_H_
#define UZH_STEREO_GET_IMAGE_STEREO_H_

#include "armadillo"
#include "matlab_port/imresize.h"
#include "opencv2/imgcodecs.hpp"
#include "transfer/cv2arma.h"

namespace uzh {

//@brief Helper function to read image.
arma::umat GetImageStereo(const std::string& image_name) {
  cv::Mat image = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
  // Downsample by a factor of 2 to speed up computation.
  image.convertTo(image, CV_64F);
  const arma::umat img = arma::conv_to<arma::umat>::from(
      uzh::cv2arma<double>(uzh::imresize(image, 0.5)).t());
  return img;
}

}  // namespace uzh

#endif  // UZH_STEREO_GET_IMAGE_STEREO_H_