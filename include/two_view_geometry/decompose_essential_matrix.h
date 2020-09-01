#ifndef UZH_TWO_VIEW_GEOMETRY_DECOMPOSE_ESSENTIAL_MATRIX_H_
#define UZH_TWO_VIEW_GEOMETRY_DECOMPOSE_ESSENTIAL_MATRIX_H_

#include <tuple>

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Decompose the essential matrix into 2 possible rotations and two
// possible translations.
//@param E [3 x 3] essential matrix computed from EstimateEssentialMatrix
// function.
//@return Rs -- The two possible rotations; u -- The 3D vector encoding the two
// possible translations where one of them is +u and another is -u.
std::tuple<arma::field<arma::mat> /* Rs */, arma::vec /* u */>
DecomposeEssentialMatrix(const arma::mat& E) {
  if (E.n_rows != 3 || E.n_cols != 3 || arma::det(E) != 0)
    LOG(ERROR) << "Invalid essential matrix.";

  // The four possible decompositions are encoded in the SVD of E.
  arma::vec s;
  arma::mat U, V;
  arma::svd(U, s, V, E);

  // Rotation is encoded in U, V and W.
  arma::field<arma::mat> Rs(2);
  const arma::mat W{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
  Rs(0) = U * W * V.t();
  Rs(1) = U * W.t() * V.t();

  // Check if the decomposed R is a valid rotation matrix, i.e. det(R)
  // If not, simply invert the sign of R.
  Rs(0) = arma::det(Rs(0)) == +1 ? Rs(0) : -Rs(0);
  Rs(1) = arma::det(Rs(1)) == +1 ? Rs(1) : -Rs(1);

  // Translation is encoded in the last column of U.
  // The two possible translations are +u and -u.
  const arma::vec u = U.tail_cols(1);

  return {Rs, u};
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_DECOMPOSE_ESSENTIAL_MATRIX_H_