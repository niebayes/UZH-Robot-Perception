#ifndef UZH_TRANSFER_ARMA2CV_H_
#define UZH_TRANSFER_ARMA2CV_H_

#include "Eigen/Core"
#include "opencv2/core.hpp"

template <class V>
void arma2cv(const arma::Mat<V> &A, cv::Mat_<V> &C) {
  cv::transpose(
      cv::Mat_<V>(static_cast<int>(A.n_cols), static_cast<int>(A.n_rows),
                  const_cast<V *>(A.memptr())),
      C);
};

#endif  // UZH_TRANSFER_ARMA2CV_H_