#ifndef UZH_MATLAB_PORT_H_
#define UZH_MATLAB_PORT_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

// TODO(bayes)
arma::mat /* filtered_image */
conv2(const arma::mat& A, const arma::mat& B) {
  // arma::conv2();
}

arma::mat /* filtered_image */
conv2(const arma::rowvec& u, const arma::vec& v, const arma::mat& A) {
  //
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_H_