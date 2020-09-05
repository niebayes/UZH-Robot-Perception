#ifndef UZH_TRANSFER_IMG2ARMA_H_
#define UZH_TRANSFER_IMG2ARMA_H_

#include "armadillo"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "transfer/cv2arma.h"

namespace uzh {

//@brief Transfer from cv::Mat to arma::umat particularly for 8-bit image.
// arma::umat img2arma(const cv::Mat& I) {
//   if (I.empty()) LOG(ERROR) << "Empty input.";

//   cv::Mat I_copy = I.clone();
//   const arma::umat I_arma = uzh::cv2arma<arma::uword>(I_copy).t();
//   return I_arma;
// }
arma::umat img2arma(const cv::Mat &C, bool copy = true) {
  /*
   OpenCV (cv::Mat) is Row-major order and Armadillo is Column-major order.
   If copy=true, arma::inplace_trans(A) should be used to keep
   the Row-major order from cv::Mat.
   */
  // return arma::Mat<V>(cvMatIn.data, cvMatIn.rows, cvMatIn.cols, false,
  // false);
  return arma::umat(reinterpret_cast<arma::uword *>(C.data),
                    static_cast<arma::uword>(C.cols),
                    static_cast<arma::uword>(C.rows),
                    /*copy_aux_mem*/ copy,
                    /*strict*/ false);
}

}  // namespace uzh

#endif  // UZH_TRANSFER_IMG2ARMA_H_