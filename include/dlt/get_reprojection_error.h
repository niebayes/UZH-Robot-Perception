#ifndef UZH_DLT_GET_REPROJECTION_ERROR_H_
#define UZH_DLT_GET_REPROJECTION_ERROR_H_

#include <cmath>

#include "Eigen/Core"

namespace uzh {

//@brief Get root-mean-squared reprojection error computed from between
// observations and
// reprojected image points.
double GetReprojectionError(
    const Eigen::Ref<const Eigen::Matrix2Xd>& observations,
    const Eigen::Ref<const Eigen::Matrix2Xd>& reprojected_points) {
  eigen_assert(observations.cols() == reprojected_points.cols());
  const int kNumPoints = static_cast<int>(observations.cols());
  return std::sqrt((observations - reprojected_points).squaredNorm() /
                   kNumPoints);
}

}  // namespace uzh

#endif  // UZH_DLT_GET_REPROJECTION_ERROR_H_