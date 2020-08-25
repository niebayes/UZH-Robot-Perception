#ifndef UZH_TRANSFER_ARMA2EIGEN_H_
#define UZH_TRANSFER_ARMA2EIGEN_H_

#include "Eigen/Core"
#include "armadillo"

namespace uzh {

Eigen::MatrixXd arma2eigen(arma::mat &A) {
  return Eigen::Map<Eigen::MatrixXd>(A.memptr(), A.n_rows, A.n_cols);
}

}  // namespace uzh

#endif  // UZH_TRANSFER_ARMA2EIGEN_H_