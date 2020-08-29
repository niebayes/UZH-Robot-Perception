#ifndef UZH_TRANSFER_ARMA2EIGEN_H_
#define UZH_TRANSFER_ARMA2EIGEN_H_

#include "Eigen/Core"
#include "armadillo"
#include "transfer/arma2cv.h"
#include "transfer/cv2eigen.h"

namespace uzh {

// Eigen::MatrixXd arma2eigen(arma::mat &A) {
//   return Eigen::Map<Eigen::MatrixXd>(A.memptr(), A.n_rows, A.n_cols);
// }

Eigen::MatrixXd arma2eigen(const arma::mat &A) {
  cv::Mat cv_mat = uzh::arma2cv<double>(A);
  return uzh::cv2eigen<double>(cv_mat);
}

}  // namespace uzh

#endif  // UZH_TRANSFER_ARMA2EIGEN_H_