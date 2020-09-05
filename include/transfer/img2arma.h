#ifndef UZH_TRANSFER_IMG2ARMA_H_
#define UZH_TRANSFER_IMG2ARMA_H_

#include "armadillo"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "transfer/cv2arma.h"

namespace uzh {

//@brief Transfer from cv::Mat to arma::umat particularly for 8-bit image.
arma::umat img2arma(const cv::Mat& I) {
  if (I.empty()) LOG(ERROR) << "Empty input.";

  const arma::Mat<uchar> M = uzh::cv2arma<uchar>(I).t();
  return arma::conv_to<arma::umat>::from(M);
}

}  // namespace uzh

#endif  // UZH_TRANSFER_IMG2ARMA_H_