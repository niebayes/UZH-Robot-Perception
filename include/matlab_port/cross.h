#ifndef UZH_MATLAB_PORT_CROSS_H_
#define UZH_MATLAB_PORT_CROSS_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Return the corresponding skew-symmetric matrix of the input 3D vector.
//@param u [3 x 1] column vector in which the elements are going to be the
// corresponding entris of its skew-symmetric matrix counterpart.
//@return U -- [3 x 3] skew-symmetric matrix.
template <typename T>
arma::Mat<T> /* U */
cross(const arma::Col<T>& u) {
  if (u.size() != 3)
    LOG(ERROR) << "cross only works for 3D vector at this moment.";

  const arma::Mat<T> U{{0, -u(2), u(1)}, {u(2), 0, -u(0)}, {-u(1), u(0), 0}};
  return U;
}

//@brief Imitate matlab's cross. Return the cross product of pairs of vectors
// stored in matrix A and B respectively.
//@param A [3 x n] matrix.
//@param B [3 x n] matrix.
//@return C -- [3 x n] matrix where each column contains the cross product of
// pairs of corresponding columns of A and B.
template <typename T>
arma::Mat<T> /* C */
cross(const arma::Mat<T>& A, const arma::Mat<T>& B) {
  if (A.n_rows != 3 || B.n_cols != 3)
    LOG(ERROR) << "cross only supports multiplication between two 3D vectors.";
  if (A.n_cols != B.n_cols)
    LOG(ERROR) << "Number of vectors in A and B must be consistent.";

  const int kNumPairs = A.n_cols;
  // Construct matrix C to be populated.
  arma::Mat<T> C(3, kNumPairs, arma::fill::zeros);

  // Compute pair-wise cross products.
  for (int i = 0; i < kNumPairs; ++i) {
    C.col(i) = cross<T>(A.col(i)) * B.col(i);
  }

  return C;
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_CROSS_H_