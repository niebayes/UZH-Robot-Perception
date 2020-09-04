#ifndef UZH_MATLAB_PORT_NNZ_H_
#define UZH_MATLAB_PORT_NNZ_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Return number of non-zero elements.
//@param X Matrix or vector.
template <typename T>
int nnz(const arma::Mat<T>& X) {
  if (X.empty()) {
    LOG(WARNING) << "nnz works with empty matrix.";
    return 0;
  }

  return arma::size(arma::nonzeros(X)).n_rows;
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_NNZ_H_