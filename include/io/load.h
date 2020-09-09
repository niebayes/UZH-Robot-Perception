#ifndef UZH_IO_LOAD_H_
#define UZH_IO_LOAD_H_

#include "Eigen/Core"
#include "armadillo"
#include "opencv2/core/core.hpp"

namespace uzh {

//@brief Load the data file utilizing armadillo library.
//! Recommended load function.
template <typename M>
M armaLoad(const std::string& file_name) {
  arma::mat arma_mat;
  arma_mat.load(file_name, arma::file_type::auto_detect, true);
  return Eigen::Map<const M>(arma_mat.memptr(), arma_mat.n_rows,
                             arma_mat.n_cols);
}

template <typename T>
arma::Mat<T> LoadArma(const std::string& file_name) {
  arma::Mat<T> mat;
  mat.load(file_name, arma::file_type::auto_detect, true);
  return mat;
}

}  // namespace uzh

#endif  // UZH_IO_LOAD_H_