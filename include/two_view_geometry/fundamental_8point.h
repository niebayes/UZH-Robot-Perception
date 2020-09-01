#ifndef UZH_TWO_VIEW_GEOMETRY_FUNDAMENTAL_8POINT_H_
#define UZH_TWO_VIEW_GEOMETRY_FUNDAMENTAL_8POINT_H_

#include "arma_traits.h"
#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Find fundamental matrix F from 2D point correspondences using 8 point
// algorithm.
//@param p1 [3 x n] matrix where each column contains the homogeneous
// coordinates of image points p1 on the left camera.
//@param p2 [3 x n] matrix where each column contains the homogeneous
// coordinates of image points p2 on the right camera.
//@return F -- [3 x 3] fundamental matrix encapsulating the two view geometry.
//! F is found by accumulating 8 point correspondences each contributing an
//! independent linear equation involving p1, p2 and F.
arma::mat /* F */
Fundamental8Point(const arma::mat& p1, const arma::mat& p2) {
  if (p1.n_cols != p2.n_rows)
    LOG(FATAL) << "Number of points of p1 and p2 must be consistent.";
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_FUNDAMENTAL_8POINT_H_