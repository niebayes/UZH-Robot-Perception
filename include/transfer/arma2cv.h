#ifndef UZH_TRANSFER_ARMA2CV_H_
#define UZH_TRANSFER_ARMA2CV_H_

#include "Eigen/Core"
#include "opencv2/core.hpp"

namespace uzh {

template <class V>
cv::Mat_<V> arma2cv(const arma::Mat<V> &A) {
  cv::Mat_<V> C;
  cv::transpose(
      cv::Mat_<V>(static_cast<int>(A.n_cols), static_cast<int>(A.n_rows),
                  const_cast<V *>(A.memptr())),
      C);
  return C;
};

}  // namespace uzh

#endif  // UZH_TRANSFER_ARMA2CV_H_