#ifndef UZH_TRANSFORM_VEE_H_
#define UZH_TRANSFORM_VEE_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Return the corresponding 3D vector of the input [3 x 3] skew-symmetric
// matrix.
//@param U [3 x 3] skew-symmetric matrix.
//@return u -- [3 x 1] column vector.
template <typename T>
inline typename arma::Col<T>::template fixed<3> /* u */
vee(const arma::Mat<T>& U) {
  if (U.empty()) LOG(ERROR) << "Empty input.";
  if (arma::size(U) != arma::size(3, 3)) {
    LOG(ERROR)
        << "cross only works for [3 x 3] skew-symmetric matrix at this moment.";
  }

  const typename arma::Col<T>::template fixed<3> u{-U(1, 2), U(0, 2), -U(0, 1)};
  return u;
}

}  // namespace uzh

#endif  // UZH_TRANSFORM_VEE_H_