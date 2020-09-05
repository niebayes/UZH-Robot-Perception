#ifndef UZH_TRANSFER_ARMA2IMG_H_
#define UZH_TRANSFER_ARMA2IMG_H_

#include "armadillo"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "transfer/arma2cv.h"

namespace uzh {

//@brief Transfer from arma::umat to cv::Mat particularly for 8-bit image.
cv::Mat arma2img(const arma::umat& M) {
  if (M.empty()) LOG(ERROR) << "Empty input.";

  cv::Mat M_cv = uzh::arma2cv<int>(arma::conv_to<arma::Mat<int>>::from(M));
  cv::Mat I; 
  M_cv.convertTo(I, CV_8UC1);
  return I;
}

}  // namespace uzh

#endif  // UZH_TRANSFER_ARMA2IMG_H_