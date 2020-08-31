#ifndef UZH_TWO_VIEW_GEOMETRY_POINTS_TO_EPIPOLAR_LINE_DISTANCE_H_
#define UZH_TWO_VIEW_GEOMETRY_POINTS_TO_EPIPOLAR_LINE_DISTANCE_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Compute the RMS (Root-Mean-Squared) distance between the image
// points and the epipolar line implicitly revealed by fundamental matrix
//@param p1 [3 x n] matrix where each column contains the homogeneous
// coordinates of the image points on the left camera.
//@param p2 [3 x n] matrix where each column contains the homogeneous
// coordinates of the image points on the right camera.
//@param F [3 x 3] fundamental matrix encapsulating the two-view geometry in
// uncalibrated space.
//@return rms_distance RMS distance between the image points and the epipolar
// line normalized by the number of point correspondences.
double /* rms_distance */
PointsToEpipolarLineDistance(const arma::mat& p1, const arma::mat& p2,
                             const arma::mat& F) {
  // assert size.
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_POINTS_TO_EPIPOLAR_LINE_DISTANCE_H_