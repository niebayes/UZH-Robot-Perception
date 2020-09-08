#ifndef UZH_TRANSFORM_TWIST_H_
#define UZH_TRANSFORM_TWIST_H_

#include "armadillo"
#include "glog/logging.h"
#include "transform/hat.h"
#include "transform/vee.h"

namespace uzh {

//! Preliminary version.

//@brief Return the corresponding 6D twist vector of the input SE(3) matrix.
//@param SE3 [4 x 4] homogeneous matrix representing the rigid body
// transformation T such that T = [R, t; 0, 1] where 0 is a [1 x 3] 0 row
// vector.
//@return twist -- [6 x 1] column vector where the tail 3 elements represents
// the angular part w such that exp(w_x) is the rotation matrix R where the w_x
// is nothing but the corresponding [3 x 3] skew-symmetric matrix of w, and the
// head 3 elements represents the linear part u such that t = Vu with V computed
// from matrix logrithm of SE3.
template <typename T>
inline typename arma::Col<T>::template fixed<6> /* twist */
SE3_to_twist(const arma::Mat<T>& SE3) {
  if (SE3.empty()) LOG(ERROR) << "Empty input.";
  if (arma::size(SE3) != arma::size(4, 4))
    LOG(ERROR) << "Invalid SE(3) matrix.";

  const typename arma::Mat<T>::template fixed<4, 4> se3_mat =
      arma::real(arma::logmat(SE3));
  const typename arma::Col<T>::template fixed<3> u =
      se3_mat(0, 3, arma::size(3, 1));
  const typename arma::Mat<T>::template fixed<3, 3> w_x =
      se3_mat(0, 0, arma::size(3, 3));
  const typename arma::Col<T>::template fixed<3> w = uzh::vee<T>(w_x);

  return arma::join_vert(u, w);
}

//@brief Return the corresponding SE(3) matrix of the input 6D twist vector.
template <typename T>
inline typename arma::Mat<T>::template fixed<4, 4> /* SE3 */
twist_to_SE3(const arma::Col<T>& xi) {
  if (xi.empty()) LOG(ERROR) << "Empty input.";
  if (xi.size() != 6) LOG(ERROR) << "Invalid twist vector.";

  const typename arma::Col<T>::template fixed<3> u = xi.head(3),
                                                 w = xi.tail(3);
  typename arma::Mat<T>::template fixed<4, 4> se3_mat;
  se3_mat(0, 0, arma::size(3, 3)) = uzh::hat<T>(w);
  se3_mat(0, 3, arma::size(3, 1)) = u;
  se3_mat.tail_rows(1).fill(T(0));

  return arma::expmat(se3_mat);
}

}  // namespace uzh

#endif  // UZH_TRANSFORM_TWIST_H_