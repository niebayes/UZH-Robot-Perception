#ifndef UZH_COMMON_CERES_FUNCTOR_H_
#define UZH_COMMON_CERES_FUNCTOR_H_

#include "Eigen/Core"
#include "ceres/ceres.h"
#include "common/type.h"
#include "common/vision.h"
#include "opencv2/core/core.hpp"

struct ReprojectionError {
  ReprojectionError(const cv::Point2d& observed, const cv::Point3d& object)
      : observed_(observed), object_(object) {}

  template <typename T>
  bool operator()(const T* const camera_matrix, T* residual) {
    // Reassemble camera matrix
    const Eigen::Matrix<T, 3, 4> camera_matrix_(camera_matrix);

    // Compute reprojection error
    const Eigen::Matrix<T, 2, 1> image_point(T(observed_.x), T(observed_.y));
    const Eigen::Matrix<T, 3, 1> object_point(T(object_.x), T(object_.y),
                                              T(object_.z));

    //@warning When interfacing ceres autodifferentiation, care has to be taken
    // on the external functions that are not templated. Because ceres will use
    // Jet type in addition to the other general type, e.g. double.
    //
    //@ref http://ceres-solver.org/interfacing_with_autodiff.html
    residual[0] =
        ComputeReprojectionError(image_point, object_point, camera_matrix_);
    return true;
  }

  static ceres::CostFunction* Create(const cv::Point2d& observed,
                                     const cv::Point3d& object) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 1, 12>(
        new ReprojectionError(observed, object)));
  }

 private:
  const cv::Point2d observed_;
  const cv::Point3d object_;
};

struct ReprojectionErrorRadial {
  ReprojectionErrorRadial(const cv::Point2d& observed,
                          const cv::Point3d& object)
      : observed_(observed), object_(object) {}

  //@brief Apply simple radial distortion to normalized image point in camera
  // coordinate system to get distorted image point in pixel coordinate system
  // under the distortion factor L = 1 + k_1 * r^2 + k_2 * r^4.
  template <typename T>
  static void ApplySimpleRadialDistortion(const T& k1, const T& k2,
                                          const T& normalized_x,
                                          const T& normalized_y, T* distorted_x,
                                          T* distorted_y) {
    const T r2 = normalized_x * normalized_x + normalized_y * normalized_y;
    const T r4 = r2 * r2;
    const T distort_factor = 1.0 + k1 * r2 + k2 * r4;

    *distorted_x = distort_factor * normalized_x;
    *distorted_y = distort_factor * normalized_y;
  }

  template <typename T>
  bool operator()(const T* const camera_matrix,
                  const T* const distortion_coefficients, T* residual) {
    const Eigen::Matrix<T, 3, 4> camera_matrix_(camera_matrix);

    // Unpack the distortion coefficients according to the indices
    T k1 = distortion_coefficients[INDEX_RADIAL_K1];
    T k2 = distortion_coefficients[INDEX_RADIAL_K2];

    // Apply radial distortion
    T normalized_x = observed_.x, normalized_y = observed_.y;
    T distorted_x, distorted_y;
    ApplySimpleRadialDistortion(k1, k2, normalized_x, normalized_y,
                                &distorted_x, &distorted_y);

    // Compute reprojection error
    const Eigen::Matrix<T, 2, 1> image_point(distorted_x, distorted_y);
    const Eigen::Matrix<T, 3, 1> object_point(object_.x, object_.y, object_.z);
    residual[0] =
        ComputeReprojectionError(image_point, object_point, camera_matrix_);
    return true;
  }

  template <typename T>
  static ceres::CostFunction* Create(const cv::Point2d& observed,
                                     const cv::Point3d& object) {
    return (new ceres::AutoDiffCostFunction<ReprojectionErrorRadial, 1, 12, 2>(
        new ReprojectionErrorRadial(observed, object)));
  }

 private:
  const cv::Point2d observed_;
  const cv::Point3d object_;
};

#endif  // UZH_COMMON_CERES_FUNCTOR_H_