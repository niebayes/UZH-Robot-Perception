#ifndef UZH_TRANSFORM_RODRIGUES_H_
#define UZH_TRANSFORM_RODRIGUES_H_

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "armadillo"
#include "glog/logging.h"

namespace uzh {

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

}  // namespace uzh

#endif  // UZH_TRANSFORM_RODRIGUES_H_