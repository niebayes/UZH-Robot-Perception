#ifndef UZH_MATLAB_PORT_CROSS_H_
#define UZH_MATLAB_PORT_CROSS_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Return the corresponding skew-symmetric matrix of the input 3D vector.
arma::mat /* U */
cross(const arma::vec& u) {
  // assert
}

//@brief Imitate matlab's cross. Return the cross product of pairs of vectors
// stored in matrix A and B respectively.
arma::mat /* C */
cross(const arma::mat& A, const arma::mat& B) {
  // assert
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_CROSS_H_