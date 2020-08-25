#ifndef UZH_MATLAB_PORT_IM2DOUBLE_H
#define UZH_MATLAB_PORT_IM2DOUBLE_H

#include "opencv2/core.hpp"

namespace uzh {
//@brief Imitate matlab's im2double. Convert an uint8 image to double image
// where the elements are in range [0, 1].
// The uint8 elements are first converted to double type and then rescaled to
// range [0, 1] by being divided by 255.0.
cv::Mat_<double> im2double(const cv::Mat& image) {
  cv::Mat_<double> double_image;
  image.convertTo(double_image, CV_64FC1, 1.0 / 255.0);
  return double_image;
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_IM2DOUBLE_H