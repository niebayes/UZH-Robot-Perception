#ifndef UZH_TWO_VIEW_GEOMETRY_POINTS_TO_EPIPOLAR_LINE_DISTANCE_H_
#define UZH_TWO_VIEW_GEOMETRY_POINTS_TO_EPIPOLAR_LINE_DISTANCE_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Compute the RMS (Root-Mean-Squared) distance between the image
//correspondences and the corresponding epipolar line.
//@param p1
//@param p2
//@param F
//@return rms_distance
double /* rms_distance */
PointsToEpipolarLineDistance(const arma::mat& p1, const arma::mat& p2,
                             const arma::mat& F) {
  // assert size.
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_POINTS_TO_EPIPOLAR_LINE_DISTANCE_H_