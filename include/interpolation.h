#ifndef UZH_INTERPOLATION_H_
#define UZH_INTERPOLATION_H_

#include "interpolation/bicubic.h"
#include "interpolation/bilinear.h"
#include "interpolation/nearest_neighbor.h"

// //! Temp codes
// // Use backward warping
// // TODO Optimize the interpolation processing; use vectorized techniques. For
// // bilinear interpolation, consider using "shift" to wrap single interpolating
// // process into a matrix-like one.
// for (int u = 0; u < image_size.width; ++u) {
//   for (int v = 0; v < image_size.height; ++v) {
//     // x = (u, v) is the pixel in undistorted image

//     // Find the corresponding distorted image if applied the disortion
//     // coefficients K
//     // First, find the normalized image coordinates
//     Eigen::Vector2d normalized_image_point =
//         (K.inverse() * Eigen::Vector2d{u, v}.homogeneous()).hnormalized();

//     // Apply the distortion
//     Eigen::Vector2d distorted_image_point;
//     DistortPoints(normalized_image_point, &distorted_image_point, D);

//     // Convert back to pixel coordinates
//     distorted_image_point.noalias() =
//         (K * distorted_image_point.homogeneous()).hnormalized();

//     // Apply interpolation
//     // up: distorted x coordinate; vp: distorted y coordinate.
//     const double up = distorted_image_point.x(), vp = distorted_image_point.y();
//     // up_0: squeeze up to the closest pixel nearest to up along the upper
//     // left direction; vp_0, in the same principle.
//     const double up_0 = std::floor(up), vp_0 = std::floor(vp);
//     uchar intensity = 0;
//     // TODO(bayes) Modularize the interpolation methods below.
//     switch (interpolation_method) {
//       case NEAREST_NEIGHBOR:
//         // Nearest-neighbor interpolation
//         //@ref https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
//         //! The correct way do this may be using std::round. However, we use
//         //! std::floor here for the sake of simplicity and consistency.
//         if (up_0 >= 0 && up_0 < image_size.width && vp_0 >= 0 &&
//             vp_0 < image_size.height) {
//           // TODO Elegantly resolve narrowing issue here.
//           intensity = distorted_image.at<uchar>({(uchar)up_0, (uchar)vp_0});
//         }
//         break;

//       case BILINEAR:
//         // Bilinear interpolation
//         // Use bilinear interpolation to counter against edge artifacts.
//         //! We apply the unit square paradigm considering the simplicity.
//         //@ref
//         // https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
//         if (up_0 + 1 >= 0 && up_0 + 1 < image_size.width && vp_0 + 1 >= 0 &&
//             vp_0 + 1 < image_size.height) {
//           const double x = up - up_0, y = vp - vp_0;
//           // TODO Elegantly resolve narrowing issue here.
//           const Eigen::Matrix2d four_corners =
//               (Eigen::Matrix<uchar, 2, 2>()
//                    << distorted_image.at<uchar>({(uchar)up_0, (uchar)vp_0}),
//                distorted_image.at<uchar>({(uchar)up_0, (uchar)(vp_0 + 1)}),
//                distorted_image.at<uchar>({(uchar)(up_0 + 1), (uchar)vp_0}),
//                distorted_image.at<uchar>(
//                    {(uchar)(up_0 + 1), (uchar)(vp_0 + 1)}))
//                   .finished()
//                   .cast<double>();
//           intensity = cv::saturate_cast<uchar>(
//               Eigen::Vector2d{1 - x, x}.transpose() * four_corners *
//               Eigen::Vector2d{1 - y, y});
//         }
//         break;

//       default:
//         LOG(ERROR) << "Invalid interpolation method";
//         break;
//     }
//     undistorted_image.at<uchar>({u, v}) = intensity;
//   }
// }
#endif  // UZH_INTERPOLATION_H_