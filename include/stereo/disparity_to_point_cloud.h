#ifndef UZH_STEREO_DISPARITY_TO_POINT_CLOUD_H
#define UZH_STEREO_DISPARITY_TO_POINT_CLOUD_H

#include <tuple>

#include "armadillo"
#include "glog/logging.h"
#include "matlab_port/meshgrid.h"

namespace stereo {

//@brief Triangulate 3D scene point cloud from disparity map and the left image
// which produces the disparity in conjunction with the corresponding right
// image.
//@param disparity_map Disparity map where each pixel corresponds to the
// disparity between the left_img and its counterpart the right image. The
// disparities are measured as the shift to left viewed from the right camera.
//@param left_img Left image. The size of the left_img is consistent with that
// of the disparity_map.
//@param K [3 x 3] calibration matrix used to back project the pixels to the ray
// through the camera center and one 3D scene point.
//@param baseline Horizontal distance between the two camera centers.
//@return A point cloud represented as a [3 x n] matrix where n is the number of
// points with valid disparity assigned to its projections. The intensities of
// these points are also returned as a [1 x n] row vector.
std::tuple<arma::mat /*point_cloud*/, arma::umat /*intensities*/>
DisparityToPointCloud(const arma::mat& disparity_map,
                      const arma::umat& left_img, const arma::mat& K,
                      const double baseline) {
  // Assure valid input.
  if (arma::size(disparity_map) != arma::size(left_img) || K.empty() ||
      baseline <= 0) {
    LOG(FATAL) << "Invalid inputs";
  }

  // Quickly get pixel coordinates. The coordinates are in homogeneous
  // representation for use of transformation.
  arma::mat X, Y;
  std::tie(X, Y) = uzh::meshgrid<double>(left_img.n_cols, left_img.n_rows);
  arma::mat left_pixels =
      arma::join_horiz(arma::vectorise(Y), arma::vectorise(X),
                       arma::ones(left_img.n_elem))
          .t();

  // Obtain pixels in right image from pixels in left image and disparity map
  // y coordinates are same.
  arma::mat right_pixels = left_pixels;
  // x coordinates are computed as u_r = u_l - disparity.
  right_pixels.row(1) -= arma::vectorise(disparity_map).t();

  // Filter out pixels without disparity assigned.
  const arma::uvec is_valid = arma::find(disparity_map > 0);
  left_pixels = left_pixels.cols(is_valid);
  right_pixels = right_pixels.cols(is_valid);

  // Transform (row, col, 1) coordinates to (u, v, 1) pixel coordinates.
  left_pixels.head_rows(2) = arma::flipud(left_pixels.head_rows(2));
  right_pixels.head_rows(2) = arma::flipud(right_pixels.head_rows(2));

  // Apply calibration matrix K to back project all pixels to corresponding rays
  // through the left camera center and corresponding 3D scene points.
  const arma::mat left_rays = K.i() * left_pixels;
  const arma::mat right_rays = K.i() * right_pixels;

  // Construct point cloud and intentensity matrices to be populated.
  arma::mat point_cloud(3, is_valid.n_elem, arma::fill::zeros);
  arma::umat intensities(1, is_valid.n_elem, arma::fill::zeros);

  // Triangulate 3D scene points by finding the points closest to the left and
  // right rays in least square sense.
  const arma::vec b{baseline, 0.0, 0.0};
  for (int i = 0; i < is_valid.n_elem; ++i) {
    // Join the pair of rays to form A.
    const arma::mat A = arma::join_horiz(left_rays.col(i), -right_rays.col(i));
    // The scales lambda_l and lambda_r are computed as the solution vector of
    // Ax = b where b is the vector of baseline. Since A is a [3 x 2] matrix, x
    // is solved by the normal equation: A'Ax = A'b, where ' denotes the
    // transpose.
    const arma::vec x = arma::solve(A, b);
    point_cloud.col(i) = x(0) * left_rays.col(i);
  }

  // Fetch the intensities for all valid points.
  intensities = left_img(is_valid).as_row();

  return {point_cloud, intensities};
}

}  // namespace stereo

#endif  // UZH_STEREO_DISPARITY_TO_POINT_CLOUD_H