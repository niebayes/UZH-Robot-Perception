#ifndef UZH_COMMON_TRANSFORM_H_
#define UZH_COMMON_TRANSFORM_H_

#include <cmath>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/type.h"

//@brief Toy function for Rodrigues formula transforming rotation vector to
// corresponding rotation matrix
//! When using const Eigen::Ref binds const object, qualify the object with a
//! const. Otherwise, errors induced.
static void Rodrigues(const Eigen::Ref<const Eigen::Vector3d>& rotation_vector,
                      Eigen::Matrix3d* rotation_matrix) {
  //* Simpler way using eigen's API.
  // Eigen::MatrixXd rotation_matrix_tmp =
  //     (Eigen::AngleAxisd(rotation_vector.norm(),
  //     rotation_vector.normalized()))
  //         .matrix();
  // std::cout << rotation_matrix_tmp << '\n';

  const double theta = rotation_vector.norm();
  const Eigen::Vector3d omega = rotation_vector.normalized();
  const Eigen::Matrix3d omega_hat =
      (Eigen::Matrix3d() << 0.0, -omega.z(), omega.y(), omega.z(), 0.0,
       -omega.x(), -omega.y(), omega.x(), 0.0)
          .finished();
  *rotation_matrix = Eigen::Matrix3d::Identity() + std::sin(theta) * omega_hat +
                     (1 - std::cos(theta)) * omega_hat * omega_hat;
}

//@brief Construct a rigid transformation matrix from the pose vector
static void PoseVectorToTransformationMatrix(const std::vector<double>& pose,
                                             RigidTransformation* T) {
  const Eigen::Vector3d rotation_vector{pose[0], pose[1], pose[2]},
      translation{pose[3], pose[4], pose[5]};
  Eigen::Matrix3d rotation_matrix;
  Rodrigues(rotation_vector, &rotation_matrix);
  //! When templating the T, the methods below are ill-defined and induce
  //! errors. There's no need to templating here.
  T->leftCols(3) = rotation_matrix;
  T->rightCols(1) = translation;
}

//@brief Check if the given matrix is a valid rotation matrix or not.
template <typename Derived, typename T>
static bool IsValidRotationMatrix(const Eigen::MatrixBase<Derived>& R,
                                  const T precision) {
  //! For the more obscure usage of template and typename keywords in C++,
  //@ref http://eigen.tuxfamily.org/dox/TopicTemplateKeyword.html
  eigen_assert(R.rows() == R.cols());
  // Restrict to SO(2) and SO(3)
  eigen_assert(R.size() == 9 || R.size() == 4);
  return (R * R.transpose() - Eigen::MatrixBase<Derived>::Identity()).norm() <
         precision;
}

#endif  // UZH_COMMON_TRANSFORM_H_