#ifndef UZH_FEATURE_PAR_ARRAY_H_
#define UZH_FEATURE_PAR_ARRAY_H_

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace uzh {

//@brief Imitate matlab's padarray. Pad the input image using the given pad_size
// and pad_value.
//@param image Input image to be padded.
//@param pad_size Vector contains size of padded values corresponding to
// different directions.
//@param pad_value Vector contains values to be padded in certain directions
// according to the pad_size. By default, the values are zeros in each dimenion.
// The size(pad_size) must be equal to size(pad_value)
//
// E.g. PadArray(A, [1, 2, 3, 4], 1) pads 1 ones to top, 2 ones to
// bottom, 3 ones to left and 4 ones to right, where the 1, 2, 3, 4 denotes the
// width of the padded borders in different directions.
void PadArray(cv::Mat& image, const cv::Scalar_<int> pad_size,
              const double pad_value = 0) {
  cv::copyMakeBorder(image, image, pad_size(0), pad_size(1), pad_size(2),
                     pad_size(3), cv::BORDER_CONSTANT, {pad_value});
}

}  // namespace uzh

#endif  // UZH_FEATURE_PAR_ARRAY_H_