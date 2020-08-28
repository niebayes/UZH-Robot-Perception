#ifndef UZH_MATLAB_PORT_IMRESIZE_H_
#define UZH_MATLAB_PORT_IMRESIZE_H_

#include <cmath>

#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

//@note OpenCV's resize function's INTER_AREA.
//@ref
// https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3

namespace uzh {

//@brief Imitate matlab's imresize. Resize an image given the resize factor.
//@param image Image to be resized.
//@param resize_factor By which the rows and cols are stretched if greater than
// 1 or squished if less than 1.
//@param The resized image.
cv::Mat imresize(const cv::Mat& image, const double resize_factor) {
  if (image.empty()) LOG(ERROR) << "Empty input image.";

  //! Note, to accomodate matlab's result, we ceil the rows and cols.
  const int rows = std::ceil(image.rows * resize_factor);
  const int cols = std::ceil(image.cols * resize_factor);
  cv::Mat resized_image = cv::Mat::zeros(rows, cols, CV_64F);

  int interpolation_method;
  if (resize_factor == 1)
    return image;
  else if (resize_factor > 1)
    interpolation_method = cv::INTER_CUBIC;
  else if (resize_factor < 1 && resize_factor > 0)
    interpolation_method = cv::INTER_AREA;
  else
    LOG(ERROR) << "Invalid resize factor";

  //! This will get the result slightly different to that of matlab.
  // cv::resize(image, resized_image, {}, resize_factor, resize_factor,
  //            interpolation_method);
  cv::resize(image, resized_image, resized_image.size(), 0, 0,
             interpolation_method);
  return resized_image;
}

//@brief Imitate matlab's imresize. Resize an image given the desired rows and
// cols of the resized image.
//@param image Image to be resized.
//@param rows Number of rows of the resized image.
//@param cols Number of cols of the resized image.
//@return The resized image.
cv::Mat imresize(const cv::Mat& image, const int rows, const int cols) {
  cv::Mat resized_image;
  int interpolation_method;
  if (rows <= 0 || cols <= 0 || image.empty()) {
    LOG(ERROR) << "Invalid parameters";
  }
  if (rows == image.rows && cols == image.cols)
    return image;
  else if (rows > image.rows || cols > image.cols)
    interpolation_method = cv::INTER_CUBIC;
  else if (rows < image.rows || cols < image.cols)
    interpolation_method = cv::INTER_AREA;
  cv::resize(image, resized_image, {rows, cols}, 0.0, 0.0,
             interpolation_method);
  return resized_image;
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_IMRESIZE_H_