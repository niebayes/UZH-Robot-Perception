#ifndef UZH_MATLAB_PORT_IMRESIZE_H_
#define UZH_MATLAB_PORT_IMRESIZE_H_

#include "opencv2/core.hpp"

//@note OpenCV's resize function's INTER_AREA.
//@ref
// https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3

namespace uzh {
cv::Mat imresize(const cv::Mat& image, const double scale) {}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_IMRESIZE_H_