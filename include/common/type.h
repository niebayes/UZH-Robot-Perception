#ifndef UZH_COMMON_TYPE_H
#define UZH_COMMON_TYPE_H

#include "Eigen/Core"

//@brief Rigid body transformation represented with non-full rank 3 x 4 matrix
using RigidTransformation = Eigen::Matrix<double, 3, 4>;

//@brief Project modes used in ProjectPoints.
// If PROJECT_WITH_DISTORTION, the function protects 3D scene points considering
// distortion and otherwise not.
namespace {
enum ProjectModes : int { PROJECT_WITH_DISTORTION, PROJECT_WITHOUT_DISTORTION };
}

//@brief Distortion coefficients' indices used to vividly extract values.
// The advantage also includes the good flexibility when transfering to another
// distortion model which may have another set of distortion coefficients.
namespace {
enum DistorionCoefficientIndices : int { INDEX_RADIAL_K1, INDEX_RADIAL_K2 };
}

//@brief Interpolation methods used in UndistortImage.
namespace {
enum InterpolationMethods : int { NEAREST_NEIGHBOR, BILINEAR };
}

#endif  // UZH_COMMON_TYPE_H