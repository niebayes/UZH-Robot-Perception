#ifndef UZH_MATLAB_PORT_CELL2MAT_H_
#define UZH_MATLAB_PORT_CELL2MAT_H_

#include "armadillo"

namespace uzh {

template <typename T>
arma::Mat<T> cell2mat(const arma::field<arma::Mat<T>>& cell) {
  if (cell.empty()) LOG(FATAL) << "Empty input cell.";
  //! arma::field can contain "undefined" object, that is matrix with only rows
  //! or columns specified.
  for (arma::Mat<T> mat : cell) {
    if (mat.n_rows == 0 || mat.n_cols == 0)
      LOG(FATAL)
          << "Please check the items in the cell, some of them are undefined.";
  }

  const int kNumMats = cell.n_elem;
  if (kNumMats == 1) return cell(0);

  arma::uvec rows(kNumMats), cols(kNumMats);
  for (int i = 0; i < kNumMats; ++i) {
    rows(i) = cell(i).n_rows;
    cols(i) = cell(i).n_cols;
  }
  const arma::uvec unique_rows = arma::unique(rows);
  const arma::uvec unique_cols = arma::unique(cols);

  arma::Mat<T> mat;
  arma::uword concated_rows = 0, concated_cols = 0;
  if (unique_rows.size() == 1) {
    // Concatenate matrices horizontally.
    concated_rows = unique_rows(0);
    concated_cols = arma::sum(cols);
    arma::Mat<T> concated_mat(concated_rows, concated_cols);
    int col_idx = 0;
    for (int i = 0; i < kNumMats; ++i) {
      concated_mat.cols(col_idx, col_idx + cols(i) - 1) = cell(i);
      col_idx += cols(i);
    }
    mat = concated_mat;

  } else if (unique_cols.size() == 1) {
    // Concatenate matrices vertically.
    concated_rows = arma::sum(rows);
    concated_cols = unique_cols(0);
    arma::Mat<T> concated_mat(concated_rows, concated_cols);
    int row_idx = 0;
    for (int i = 0; i < kNumMats; ++i) {
      concated_mat.rows(row_idx, row_idx + rows(i) - 1) = cell(i);
      row_idx += rows(i);
    }
    mat = concated_mat;

  } else {
    LOG(FATAL) << "At least one dimension has to be consistent.";
  }

  return mat;
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_CELL2MAT_H_