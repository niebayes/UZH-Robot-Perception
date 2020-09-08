#ifndef UZH_TRANSFORM_HAT_H_
#define UZH_TRANSFORM_HAT_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Return the corresponding skew-symmetric matrix of the input 3D vector.
//@param u [3 x 1] column vector in which the elements are going to be the
// corresponding entris of its skew-symmetric matrix counterpart.
//@return U -- [3 x 3] skew-symmetric matrix.
template <typename T>
inline typename arma::Mat<T>::template fixed<3, 3> /* U */
hat(const arma::Col<T>& u) {
  if (u.size() != 3)
    LOG(ERROR) << "cross only works for 3D vector at this moment.";

  const typename arma::Mat<T>::template fixed<3, 3> U{
      {0, -u(2), u(1)}, {u(2), 0, -u(0)}, {-u(1), u(0), 0}};

  return U;
}

}  // namespace uzh

#endif  // UZH_TRANSFORM_HAT_H_