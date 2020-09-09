#ifndef UZH_IMAGE_FORMATION_PROJECT_POINTS_H_
#define UZH_IMAGE_FORMATION_PROJECT_POINTS_H_

#include "Eigen/Core"
#include "image_formation/distort_points.h"

namespace uzh {

//@brief Project modes used in ProjectPoints.
// If PROJECT_WITH_DISTORTION, the function protects 3D scene points considering
// distortion and otherwise not.
enum ProjectModes : int { PROJECT_WITH_DISTORTION, PROJECT_WITHOUT_DISTORTION };

//@brief Project 3D scene points according to calibration matrix K and optional
// user provided distortion coefficients D
//! The std::optional is a feature of C++17. Alternatively you can avoid this by
//! overloading ProjectPoints function.
void ProjectPoints(const Eigen::Ref<const Eigen::Matrix3Xd>& object_points,
                   Eigen::Matrix2Xd* image_points,
                   const Eigen::Ref<const Eigen::Matrix3d>& K,
                   const int project_mode = uzh::PROJECT_WITHOUT_DISTORTION,
                   const std::optional<Eigen::Vector2d>& D_opt = std::nullopt) {
  Eigen::Vector2d D = Eigen::Vector2d::Zero();
  if (project_mode == PROJECT_WITH_DISTORTION && D_opt) {
    D = D_opt.value();
  }
  const Eigen::Matrix2Xd normalized_image_points =
      object_points.colwise().hnormalized();
  uzh::DistortPoints(normalized_image_points, image_points, D);
  *image_points = (K * image_points->colwise().homogeneous()).topRows(2);
}

}  // namespace uzh

#endif  // UZH_IMAGE_FORMATION_PROJECT_POINTS_H_