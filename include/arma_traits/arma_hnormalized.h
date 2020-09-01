#ifndef UZH_ARMA_TRAITS_ARMA_HNORMALIZED_H_
#define UZH_ARMA_TRAITS_ARMA_HNORMALIZED_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Dehomogenize matrix or vector along each row or column.
//@param m Matrix or vector to be dehomogenized.
//@param dim The trailing ones are removed from each column (dim = 0) or each
// row (dim = 1). By default, dim is 0.
//@return hn_m -- Dehomogenized matrix or vector.
//! The rows / columns but the last row / column are divided by the elements in
//! the last row / column and the resulted reduced matrix or vector is returned.
template <typename T>
arma::Mat<T> /* hn_m */
hnormalized(const arma::Mat<T>& m, const int dim = 0) {
  if (m.empty()) LOG(ERROR) << "Empty input matrix / vector.";

  arma::Mat<T> hn_m = m;
  if (dim == 0) {
    if (m.n_rows <= 1) {
      LOG(WARNING) << "Row vector shall not be hnormalized along columns.";
      return hn_m;
    }
    const arma::Mat<T>& last_row = m.tail_rows(1);
    for (int c = 0; c < m.n_cols; ++c) {
      hn_m.col(c) /= arma::as_scalar(last_row.col(c));
    }
    hn_m = hn_m.head_rows(m.n_rows - 1);

  } else if (dim == 1) {
    if (m.n_cols <= 1) {
      LOG(WARNING) << "Column vector shall not be hnormalized along rows.";
      return hn_m;
    }
    const arma::Mat<T>& last_col = m.tail_cols(1);
    for (int r = 0; r < m.n_rows; ++r) {
      hn_m.row(r) /= arma::as_scalar(last_col.row(r));
    }
    hn_m = hn_m.head_cols(m.n_cols - 1);

  } else
    LOG(FATAL) << "Invalid dim value.";

  return hn_m;
}

}  // namespace uzh

#endif  // UZH_ARMA_TRAITS_ARMA_HNORMALIZED_H_