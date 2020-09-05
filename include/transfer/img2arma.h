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

  cv::Mat I_copy = I.clone();
  const arma::umat I_arma = uzh::cv2arma<arma::uword>(I_copy).t();
  return I_arma;
}

}  // namespace uzh

#endif  // UZH_TRANSFER_IMG2ARMA_H_