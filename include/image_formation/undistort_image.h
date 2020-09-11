#ifndef UZH_IMAGE_FORMATION_UNDISTORT_IMAGE_H_
#define UZH_IMAGE_FORMATION_UNDISTORT_IMAGE_H_

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "glog/logging.h"
#include "image_formation/distort_points.h"
#include "opencv2/core.hpp"
#include "opencv2/core/eigen.hpp"

namespace uzh {

//@brief Interpolation methods used in UndistortImage.
enum InterpolationMethod : int { NEAREST_NEIGHBOR, BILINEAR };

//@brief Undistort image according to distortion function \Tau specified with
// distortion coefficients D
//@note A good tip for using Eigen::Ref with derived types:
//@ref https://stackoverflow.com/a/58463638/14007680
cv::Mat UndistortImage(const cv::Mat &distorted_image,
                       const Eigen::Ref<const Eigen::Matrix3d> &K,
                       const Eigen::Ref<const Eigen::Vector2d> &D,
                       const int interpolation_method) {
  if (distorted_image.channels() > 1) {
    LOG(ERROR) << "Only support grayscale image at this moment";
  }

  const cv::Size &image_size = distorted_image.size();
  cv::Mat undistorted_image = cv::Mat::zeros(image_size, CV_8UC1);
  Eigen::MatrixXd distorted_image_(image_size.height, image_size.width);
  cv::cv2eigen(distorted_image, distorted_image_);

  // Use backward warping
  for (int u = 0; u < image_size.width; ++u) {
    for (int v = 0; v < image_size.height; ++v) {
      // x = (u, v) is the pixel in undistorted image

      // Find the corresponding distorted image if applied the disortion
      // coefficients K
      // First, find the normalized image coordinates
      Eigen::Vector2d normalized_image_point =
          (K.inverse() * Eigen::Vector2d{u, v}.homogeneous()).hnormalized();

      // Apply distortion
      Eigen::Vector2d distorted_image_point;
      uzh::DistortPoints(normalized_image_point, &distorted_image_point, D);

      // Convert back to pixel coordinates
      distorted_image_point =
          ((K * distorted_image_point.homogeneous()).hnormalized()).eval();

      // Interpolate.
      // up: distorted x coordinate; vp: distorted y coordinate.
      const double up = distorted_image_point.x(),
                   vp = distorted_image_point.y();

      // up_0: squeeze up to the closest pixel nearest to up along the upper
      // left direction; vp_0, in the same principle.
      const double up_0 = std::floor(up), vp_0 = std::floor(vp);
      uchar intensity = 0;
      switch (interpolation_method) {
        case NEAREST_NEIGHBOR:
          //! The correct way do this may be using std::round. However, we use
          //! std::floor here for the sake of simplicity and consistency.
          if (up_0 >= 0 && up_0 < image_size.width && vp_0 >= 0 &&
              vp_0 < image_size.height) {
            //! Follow the convection that x -> col, y -> row.
            intensity = distorted_image_(vp_0, up_0);
          }
          break;

        case BILINEAR:
          // Bilinear interpolation
          // Use bilinear interpolation to counter against edge artifacts.
          if (up_0 + 1 >= 0 && up_0 + 1 < image_size.width && vp_0 + 1 >= 0 &&
              vp_0 + 1 < image_size.height) {
            const double x = up - up_0, y = vp - vp_0;
            const Eigen::Matrix2d four_corners =
                (Eigen::Matrix<double, 2, 2>() << distorted_image_(vp_0, up_0),
                 distorted_image_(vp_0 + 1, up_0),
                 distorted_image_(vp_0, up_0 + 1),
                 distorted_image_(vp_0 + 1, up_0 + 1))
                    .finished();
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

}  // namespace uzh

#endif  // UZH_IMAGE_FORMATION_UNDISTORT_IMAGE_H_