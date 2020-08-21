#ifndef UZH_FEATURE_CONV2_H_
#define UZH_FEATURE_CONV2_H_

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

namespace external {
enum ConvolutionType {
  /* Return the full convolution, including border */
  CONVOLUTION_FULL,

  /* Return only the part that corresponds to the original image */
  CONVOLUTION_SAME,

  /* Return only the submatrix containing elements that were not influenced by
   * the border
   */
  CONVOLUTION_VALID
};

void conv2(const Mat& img, const Mat& kernel, ConvolutionType type, Mat& dest) {
  Mat source = img;
  if (CONVOLUTION_FULL == type) {
    source = Mat();
    const int additionalRows = kernel.rows - 1,
              additionalCols = kernel.cols - 1;
    copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2,
                   (additionalCols + 1) / 2, additionalCols / 2,
                   BORDER_CONSTANT, Scalar(0));
  }

  Point anchor(kernel.cols - kernel.cols / 2 - 1,
               kernel.rows - kernel.rows / 2 - 1);
  int borderMode = BORDER_CONSTANT;
  // filter2D(source, dest, img.depth(), /*flip(kernel)*/, anchor, 0,
  // borderMode);

  if (CONVOLUTION_VALID == type) {
    dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2)
               .rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
  }
}
}  // namespace external

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