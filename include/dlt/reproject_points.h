#ifndef UZH_IMAGE_FORMATION_REPROJECT_POINTS_H_
#define UZH_IMAGE_FORMATION_REPROJECT_POINTS_H_

#include <optional>

#include "Eigen/Core"
#include "image_formation/distort_points.h"
#include "image_formation/project_points.h"

namespace uzh {

//@brief Reproject 3D scene points according to calibration matrix K, camera
// pose M = [R|t] and optional user provided distortion coefficients D
// This function differ with the ProjectPoints in that it additionally takes as
// input camera pose, hence the name "reproject" -- project the 3D scene points
// visible at one camera pose to another.
//@warning This function assumes that all 3D scene points are visible for every
// camera pose, i.e. not condisering visibility.
// TODO Implement a version considering visibility.
static void ReprojectPoints(
    const Eigen::Ref<const Eigen::Matrix3Xd>& object_points,
    Eigen::Matrix2Xd* image_points, const Eigen::Ref<const Eigen::Matrix3d>& K,
    const Eigen::Ref<const Eigen::Matrix<double, 3, 4>>& M,
    const int project_mode,
    const std::optional<Eigen::Vector2d>& D_opt = std::nullopt) {
  Eigen::Vector2d D = Eigen::Vector2d::Zero();
  if (project_mode == uzh::PROJECT_WITH_DISTORTION && D_opt) {
    D = D_opt.value();
  }
  const Eigen::Matrix2Xd normalized_image_points =
      object_points.colwise().hnormalized();
  DistortPoints(normalized_image_points, image_points, D);
  *image_points = (K * M * object_points.colwise().homogeneous())
                      .colwise()
                      .hnormalized()
                      .topRows(2);
}

}  // namespace uzh

#endif  // UZH_IMAGE_FORMATION_REPROJECT_POINTS_H_