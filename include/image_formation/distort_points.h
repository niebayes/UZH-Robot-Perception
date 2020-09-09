#ifndef UZH_IMAGE_FORMATION_DISTORT_POINTS_H_
#define UZH_IMAGE_FORMATION_DISTORT_POINTS_H_

#include "Eigen/Core"

namespace uzh {

//@brief Distortion coefficients' indices used to vividly extract values.
// The advantage also includes the good flexibility when transfering to another
// distortion model which may have another set of distortion coefficients.
enum DistorionCoefficientIndices : int { INDEX_RADIAL_K1, INDEX_RADIAL_K2 };

//@brief Distort normalized image points to get the image points expressed in
// pixel coords.
//! Distortion model: x_d = x_u * (1 + k1 * r2 + k2 * r2 * r2)
template <typename Derived>
void DistortPoints(const Eigen::MatrixBase<Derived>& normalized_image_points,
                   Eigen::MatrixBase<Derived>* distorted_image_points,
                   const Eigen::Ref<const Eigen::Vector2d>& D) {
  // Unpack D to get the distortion coefficients
  const double k1 = D(INDEX_RADIAL_K1);
  const double k2 = D(INDEX_RADIAL_K2);

  const Eigen::VectorXd r2 = normalized_image_points.colwise().squaredNorm();
  const Eigen::VectorXd distortion_factor =
      (k1 * r2 + k2 * r2.cwiseProduct(r2)).unaryExpr([](double x) {
        return ++x;
      });
  *distorted_image_points =
      normalized_image_points * distortion_factor.asDiagonal();
}

}  // namespace uzh

#endif  // UZH_IMAGE_FORMATION_DISTORT_POINTS_H_