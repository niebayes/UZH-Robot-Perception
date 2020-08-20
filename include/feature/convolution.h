#ifndef UZH_FEATURE_CONV2_H_
#define UZH_FEATURE_CONV2_H_

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

//@brief Imitate matlab's conv2. Convolve the image with the given kernel.
// TODO(bayes) Remove OpenCV dependency.
void Conv2D(cv::InputArray src, cv::OutputArray dst, int ddepth,
            cv::InputArray kernel, cv::Point anchor = cv::Point(-1, -1),
            double delta = 0.0, int border_type = 4) {
  cv::filter2D(src, dst, ddepth, kernel, anchor, delta, border_type);
}

//@brief Overloaded of conv2d. Convolve the image first with the kernel_x along
// the rows and then convolve the result obtained with the kernel_y along the
// columns.
// TODO(bayes) Remove OpenCV dependency.
// TODO(bayes) Implement.
// void Conv2D() { cv::sepFilter2D(); }

//@brief Imitate matlab's padarray. Pad the images with the given border type.
// TODO(bayes) Remove OpenCV dependency.
// TODO(bayes) Implement.
void PadArray() {}

#endif  // UZH_FEATURE_CONV2_H_