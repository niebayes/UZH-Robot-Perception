#ifndef UZH_MATLAB_PORT_FIND_H_
#define UZH_MATLAB_PORT_FIND_H_

#include <algorithm>  // std::transform, std::copy_if
#include <numeric>    // std::iota
#include <tuple>
#include <vector>

#include "armadillo"
#include "matlab_port/nnz.h"

namespace uzh {

//@brief Remove zeros in an array.
std::vector<int> remove_zeros(const std::vector<int>& A) {
  //! Alternative way.
  // std::vector<int> A_(A.cbegin(), A.cend());
  // std::vector<int>::const_iterator last_non_zero =
  //     std::remove_if(A_.begin(), A_.end(), [](int x) { return x != 0; });
  // std::vector<int> A_no_zeros(A_.cbegin(), last_non_zero);
  const auto num_non_zeros =
      std::count_if(A.cbegin(), A.cend(), [](int x) { return x != 0; });
  std::vector<int> A_no_zeros(num_non_zeros);
  std::copy_if(A.cbegin(), A.cend(), A_no_zeros.begin(),
               [](int x) { return x != 0; });
  return A_no_zeros;
}

//@brief Imitate matlab's `[row, col, v] = find(A)` function. Find non-zero
// elements in an array A.
//@param A One dimensional array.
//@return row An array containing the row subscripts of the non-zero elements in
// A.
//@return col An array containing the column subscripts of the non-zero elements
// in A.
//@return v One dimensional array containing the non-zero elements with order
// being consistent with the original order in A. I.e. this function is stable.
std::tuple<std::vector<int> /*row*/, std::vector<int> /*col*/, arma::uvec /*v*/>
find(const arma::uvec& A) {
  // Assure all elements are greater than or equal to 0.
  assert(!arma::any(A < 0));

  // Get row
  const int num_nonzeros = uzh::nnz<arma::uword>(A);
  std::vector<int> row(num_nonzeros, 1);

  // Get col
  std::vector<int> indices(A.n_elem);
  std::iota(indices.begin(), indices.end(), 0);
  std::transform(indices.begin(), indices.end(), indices.begin(),
                 [&A](int i) { return A(i) > 0 ? i : 0; });
  std::vector<int> col = uzh::remove_zeros(indices);
  if (A(0) != 0) col.insert(col.begin(), 0);

  // Get v
  arma::uvec v(num_nonzeros);
  std::copy_if(A.cbegin(), A.cend(), v.begin(), [](int x) { return x > 0; });

  return {row, col, v};
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_FIND_H_