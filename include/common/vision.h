#ifndef UZH_COMMON_VISION_H_
#define UZH_COMMON_VISION_H_

#include <cmath>
#include <optional>  // C++17: std::optional

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/type.h"
#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"

//@brief Distort normalized image points to get the image points expressed in
// pixel coords.
// Distortion model: x_d = x_u * (1 + k1 * r2 + k2 * r2 * r2)
// TODO Change normalized_image_points from Matrix2Xd to Matrix3Xd to accomodate
// the convection that normalized image points are expressed at Z = 1 where the
// coordinates are 3D vectors.
template <typename Derived>
static void DistortPoints(
    const Eigen::MatrixBase<Derived>& normalized_image_points,
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

//@brief Project 3D scene points according to calibration matrix K and optional
// user provided distortion coefficients D
//! The std::optional is a feature of C++17. Alternatively you can avoid this by
//! overloading ProjectPoints function.
//? This paradigm is not working.
// template <typename Derived>
// const Eigen::Ref<const Eigen::MatrixBase<Derived>>& object_points
static void ProjectPoints(
    const Eigen::Ref<const Eigen::Matrix3Xd>& object_points,
    Eigen::Matrix2Xd* image_points, const Eigen::Ref<const Eigen::Matrix3d>& K,
    const int project_mode,
    const std::optional<Eigen::Vector2d>& D_opt = std::nullopt) {
  Eigen::Vector2d D = Eigen::Vector2d::Zero();
  if (project_mode == PROJECT_WITH_DISTORTION && D_opt) {
    D = D_opt.value();
  }
  const Eigen::Matrix2Xd normalized_image_points =
      object_points.colwise().hnormalized();
  DistortPoints(normalized_image_points, image_points, D);
  *image_points = (K * image_points->colwise().homogeneous()).topRows(2);
}

//@brief Undistort image according to distortion function \Tau specified with
// distortion coefficients D
cv::Mat UndistortImage(const cv::Mat& distorted_image,
                       const Eigen::Ref<const Eigen::Matrix3d>& K,
                       const Eigen::Ref<const Eigen::Vector2d>& D,
                       const int interpolation_method) {
  if (distorted_image.channels() > 1) {
    LOG(ERROR) << "Only support grayscale image at this moment";
  }

  const cv::Size& image_size = distorted_image.size();
  cv::Mat undistorted_image = cv::Mat::zeros(image_size, CV_8UC1);
  Eigen::MatrixXd distorted_image_eigen(image_size.height, image_size.width);
  cv::cv2eigen(distorted_image, distorted_image_eigen);

  // Use backward warping
  // TODO Optimize the interpolation processing; use vectorized techniques. For
  // bilinear interpolation, consider using "shift" to wrap single interpolating
  // process into a matrix-like one.
  for (int u = 0; u < image_size.width; ++u) {
    for (int v = 0; v < image_size.height; ++v) {
      // x = (u, v) is the pixel in undistorted image

      // Find the corresponding distorted image if applied the disortion
      // coefficients K
      // First, find the normalized image coordinates
      Eigen::Vector2d normalized_image_point =
          (K.inverse() * Eigen::Vector2d{u, v}.homogeneous()).hnormalized();

      // Apply the distortion
      Eigen::Vector2d distorted_image_point;
      DistortPoints(normalized_image_point, &distorted_image_point, D);

      // Convert back to pixel coordinates
      distorted_image_point.noalias() =
          (K * distorted_image_point.homogeneous()).hnormalized();

      // Apply interpolation
      // up: distorted x coordinate; vp: distorted y coordinate.
      const double up = distorted_image_point.x(),
                   vp = distorted_image_point.y();
      // up_0: squeeze up to the closest pixel nearest to up along the upper
      // left direction; vp_0, in the same principle.
      const double up_0 = std::floor(up), vp_0 = std::floor(vp);
      uchar intensity = 0;
      switch (interpolation_method) {
        case NEAREST_NEIGHBOR:
          // Nearest-neighbor interpolation
          //@ref https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
          //! The correct way do this may be using std::round. However, we use
          //! std::floor here for the sake of simplicity and consistency.
          if (up_0 >= 0 && up_0 < image_size.width && vp_0 >= 0 &&
              vp_0 < image_size.height) {
            // TODO Elegantly resolve narrowing issue here.
            intensity = distorted_image.at<uchar>({up_0, vp_0});
          }
          break;

        case BILINEAR:
          // Bilinear interpolation
          // Use bilinear interpolation to counter against edge artifacts.
          //! We apply the unit square paradigm considering the simplicity.
          //@ref
          // https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
          if (up_0 + 1 >= 0 && up_0 + 1 < image_size.width && vp_0 + 1 >= 0 &&
              vp_0 + 1 < image_size.height) {
            const double x = up - up_0, y = vp - vp_0;
            // TODO Elegantly resolve narrowing issue here.
            const Eigen::Matrix2d four_corners =
                (Eigen::Matrix<uchar, 2, 2>()
                     << distorted_image.at<uchar>({up_0, vp_0}),
                 distorted_image.at<uchar>({up_0, vp_0 + 1}),
                 distorted_image.at<uchar>({up_0 + 1, vp_0}),
                 distorted_image.at<uchar>({up_0 + 1, vp_0 + 1}))
                    .finished()
                    .cast<double>();
            intensity = cv::saturate_cast<uchar>(
                Eigen::Vector2d{1 - x, x}.transpose() * four_corners *
                Eigen::Vector2d{1 - y, y});
          }
          break;

        default:
          LOG(ERROR) << "Invalid interpolation method";
          break;
      }
      undistorted_image.at<uchar>({u, v}) = intensity;
    }
  }
  const std::string log_info{std::string{"Undistorted an image with "}.append(
      interpolation_method ? "bilinear interpolation"
                           : "nearest-neighbor interpolation")};
  LOG(INFO) << log_info;
  return undistorted_image;
}

#endif  // UZH_COMMON_VISION_H_