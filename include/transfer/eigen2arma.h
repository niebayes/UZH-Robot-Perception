#ifndef UZH_TRANSFER_EIGEN2ARMA_H_
#define UZH_TRANSFER_EIGEN2ARMA_H_

#include "Eigen/Core"
#include "armadillo"

namespace uzh {

arma::mat eigen2arma(Eigen::MatrixXd &E, bool copy = true) {
  return arma::mat(E.data(), E.rows(), E.cols(), /*copy_aux_mem*/ copy,
                   /*strict*/ false);
}

}  // namespace uzh

#endif  // UZH_TRANSFER_EIGEN2ARMA_H_