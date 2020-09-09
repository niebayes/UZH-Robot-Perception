#ifndef UZH_SIFT_GET_IMAGE_SIFT_H_
#define UZH_SIFT_GET_IMAGE_SIFT_H_

#include "matlab_port/im2double.h"
#include "matlab_port/imresize.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

namespace uzh {

//@brief Return an image given file name, data depth and rescale factor.
//@param file_name String denoting the file name including the relative file
// path.
//@param rescale_factor The returned image is rescaled according to this factor.
//@return The rescaled image.
cv::Mat GetImageSIFT(const std::string& file_name,
                     const double rescale_factor = 1.0) {
  return uzh::im2double(uzh::imresize(
      cv::imread(file_name, cv::IMREAD_GRAYSCALE), rescale_factor));
}

}  // namespace uzh

#endif  // UZH_SIFT_GET_IMAGE_SIFT_H_