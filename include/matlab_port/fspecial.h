#ifndef UZH_MATLAB_PORT_FSPECIAL_H_
#define UZH_MATLAB_PORT_FSPECIAL_H_

#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace uzh {

enum FilterType : int { GAUSSIAN };

cv::Mat fspecial(const int filter_type, const int aperture_size,
                 const double sigma, const int ddepth = CV_64F) {
  cv::Mat filter;
  if (filter_type == GAUSSIAN) {
    cv::Mat gaussian_vec = cv::getGaussianKernel(aperture_size, sigma, ddepth);
    filter = gaussian_vec * gaussian_vec.t();
  } else {
    LOG(ERROR) << "Other filter types are not implemented yet.";
  }
  return filter;
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_FSPECIAL_H_