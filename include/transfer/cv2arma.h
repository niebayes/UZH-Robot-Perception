#ifndef UZH_TRANSFER_CV2ARMA_H_
#define UZH_TRANSFER_CV2ARMA_H_

#include "armadillo"
#include "opencv2/core.hpp"

namespace uzh {

//! The returned arma::Mat<V> is not transposed to accomodate the row-major
//! convection of OpenCV.
template <class V>
arma::Mat<V> cv2arma(const cv::Mat &C, bool copy = true) {
  /*
   OpenCV (cv::Mat) is Row-major order and Armadillo is Column-major order.
   If copy=true, arma::inplace_trans(A) should be used to keep
   the Row-major order from cv::Mat.
   */
  // return arma::Mat<V>(cvMatIn.data, cvMatIn.rows, cvMatIn.cols, false,
  // false);
  return arma::Mat<V>(reinterpret_cast<V *>(C.data),
                      static_cast<arma::uword>(C.cols),
                      static_cast<arma::uword>(C.rows),
                      /*copy_aux_mem*/ copy,
                      /*strict*/ false);
}

}  // namespace uzh

#endif  // UZH_TRANSFER_CV2ARMA_H_