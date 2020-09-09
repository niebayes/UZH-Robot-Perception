#ifndef UZH_TRANSFER_CV2EIGEN_H_
#define UZH_TRANSFER_CV2EIGEN_H_

#include "Eigen/Core"
#include "opencv2/core.hpp"
#include "opencv2/core/eigen.hpp"

namespace uzh {

// MatrixX<float> x; instead of Eigen::MatrixXf x;
template <typename V>
using MatrixX = typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>;

// MatrixXrm<float> x; instead of MatrixXf_rm x;
template <typename V>
using MatrixXrm =
    typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// void cv2eigen(const Mat& src, Eigen::Matrix<_Tp, _rows, _cols, _options,
// _maxRows, _maxCols>& dst)

template <typename V>
MatrixX<V> cv2eigen(cv::Mat &C) {
  Eigen::Map<MatrixXrm<V>> E(C.ptr<V>(), C.rows, C.cols);
  return E;
}

}  // namespace uzh

#endif  // UZH_TRANSFER_CV2EIGEN_H_