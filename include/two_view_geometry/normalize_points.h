#ifndef UZH_TWO_VIEW_GEOMETRY_NORMALIZE_POINTS_H_
#define UZH_TWO_VIEW_GEOMETRY_NORMALIZE_POINTS_H_

#include <cmath>
#include <tuple>

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Normalize the set of points by scaling and shifting such that the
// centroid of the normalized set of points is the origin (0, 0) for 2D or (0,
// 0, 0) for 3D, and the root mean squared distance is sqrt(2) for 2D or sqrt(3)
// for 3D.
//@param p [3 x n] or [4 x n] matrix where each column contains the homogeneous
// coordinates of the points.
//@return normalized_p -- [3 x n] or [4 x n] matrix where each column contains
// the normalized homogeneous coordinates; and T -- [3 x 3] or [4 x 4] matrix
// which is the transformation applied to achieve the normalization.
std::tuple<arma::mat /* normalized_p */, arma::mat /* T */> NormalizePoints(
    const arma::mat& p) {
  if (p.empty()) LOG(ERROR) << "Empty input points.";
  if (p.n_rows != 3 || p.n_rows != 4)
    LOG(ERROR) << "The points must be homogeneous 2D or 3D points.";

  const int kDim = p.n_rows - 1;

  arma::mat normalized_p, T;

  if (kDim == 2) {  // if 2D points.
    // Compute mean along each dimension and then compute standard deviation.
    const arma::vec2 mu = arma::mean(p.head_rows(2), 1);
    const double sigma =
        arma::norm(p.head_rows(2) - arma::repmat(mu, 1, p.n_cols), "fro") /
        (p.n_cols * p.n_cols);
    // Build normalization transformation matrix T.
    const double s = std::sqrt(2) / sigma;
    const arma::mat33 T_tmp{{s, 0, -s * mu(0)}, {0, s, -s * mu(1)}, {0, 0, 1}};
    T = T_tmp;
    normalized_p = T * p;

  } else {  // if 3D points
    // Compute mean along each dimension and then compute standard deviation.
    const arma::vec3 mu = arma::mean(p.head_rows(3), 1);
    const double sigma =
        arma::norm(p.head_rows(3) - arma::repmat(mu, 1, p.n_cols), "fro") /
        (p.n_cols * p.n_cols);
    // Build normalization transformation matrix T.
    const double s = std::sqrt(3) / sigma;
    const arma::mat44 T_tmp{{s, 0, 0, -s * mu(0)},
                            {0, s, 0, -s * mu(1)},
                            {0, 0, s, -s * mu(2)},
                            {0, 0, 0, 1}};
    T = T_tmp;
    normalized_p = T * p;
  }

  return {normalized_p, T};
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_NORMALIZE_POINTS_H_