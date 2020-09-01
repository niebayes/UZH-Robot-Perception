#ifndef UZH_ARMA_TRATIS_ARMA_HOMOGENEOUS_H_
#define UZH_ARMA_TRATIS_ARMA_HOMOGENEOUS_H_

#include "armadillo"

namespace uzh {

//@brief Homogenize each column or row of matrix or vector.
//@param m Matrix or vector to be augmented by ones along row or column.
//@param dim Ones are added to each column (dim = 0) or each row (dim = 1). By
// default, dim = 0.
//@return h_m -- Homogenized matrix or vector.
template <typename T>
arma::Mat<T> /* h_m */
homogeneous(const arma::Mat<T>& m, const int dim = 0) {
  if (m.empty()) LOG(ERROR) << "Empty input matrix / vector.";

  arma::Mat<T> h_m;
  arma::Mat<T> ones;
  if (dim == 0) {
    ones = arma::ones<arma::Mat<T>>(1, m.n_cols);
    h_m = arma::join_vert(m, ones);
  } else if (dim == 1) {
    ones = arma::ones<arma::Mat<T>>(m.n_rows, 1);
    h_m = arma::join_horiz(m, ones);
  } else
    LOG(ERROR) << "Invalid dim value.";

  return h_m;
}

}  // namespace uzh

#endif  // UZH_ARMA_TRATIS_ARMA_HOMOGENEOUS_H_