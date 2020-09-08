#ifndef UZH_BA_PROJECTION_POINT_H_
#define UZH_BA_PROJECTION_POINT_H_

#include "arma_traits/arma_hnormalized.h"
#include "arma_traits/arma_homogeneous.h"
#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Project 3D scene points given calibration matrix K.
//@param object_points [3 x n] column vector containing the (X, Y, Z)
// coordinates of the 3D scene points. n is the number of points.
//@param K [3 x 3] calibration matrix.
//@return image_point -- [2 x n] column vector containing the (x, y) coordinates
// of the projection 2D image point expressed in pixels. n is the number of
// points.
//! This 3D scene points need to be first transformed into certain frame if it's
//! desired to project points into that frame.
//! This function currently assumes no distortion.
arma::mat /* image_points */
ProjectPoint(const arma::mat& object_point, const arma::mat33& K) {
  if (object_point.empty() || K.empty()) LOG(ERROR) << "Empty input.";
  if (object_point.n_rows != 3) LOG(ERROR) << "Invalid input.";

  // Get normalized image points.
  const arma::mat normalized_image_points =
      uzh::hnormalized<double>(object_point);

  // Apply K to get pixel coordinates.
  const arma::mat image_points =
      K * uzh::homogeneous<double>(normalized_image_points);

  return image_points.head_rows(2);
}

}  // namespace uzh

#endif  // UZH_BA_PROJECTION_POINT_H_