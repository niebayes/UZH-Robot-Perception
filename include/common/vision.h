#ifndef UZH_COMMON_VISION_H_
#define UZH_COMMON_VISION_H_

#include <cmath>
#include <optional>  // C++17: std::optional

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/QR"
#include "Eigen/SVD"
#include "common/normalize.h"
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
            intensity = distorted_image.at<uchar>({(uchar)up_0, (uchar)vp_0});
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
                     << distorted_image.at<uchar>({(uchar)up_0, (uchar)vp_0}),
                 distorted_image.at<uchar>({(uchar)up_0, (uchar)(vp_0 + 1)}),
                 distorted_image.at<uchar>({(uchar)(up_0 + 1), (uchar)vp_0}),
                 distorted_image.at<uchar>(
                     {(uchar)(up_0 + 1), (uchar)(vp_0 + 1)}))
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

//@brief Compute the reprojection error between the measured image points and
// the projections of the measured object points with the given camera matrix.
//
// The reprojection error is given by: E(x, X, P) =  sum_i  |x_i - P X_i|^2.
//
//@warning The reprojection error should always be "averaged", i.e. the RMS of
// the reprojection error is returned.
//
// Templated for use in both multiple points case and single point case.
// const Eigen::Ref<const Eigen::Matrix2Xd>& image_points,
// const Eigen::Ref<const Eigen::Matrix3Xd>& object_points,
template <typename DerivedA, typename DerivedB>
static double ComputeReprojectionError(
    const Eigen::MatrixBase<DerivedA>& image_points,
    const Eigen::MatrixBase<DerivedB>& object_points,
    const Eigen::Ref<const Eigen::Matrix<double, 3, 4>>& camera_matrix) {
  const Eigen::Matrix2Xd image_points_hat =
      (camera_matrix * object_points.colwise().homogeneous())
          .colwise()
          .hnormalized();
  //? The error computed this way is the rms reprojection error?
  double reprojection_error = std::sqrt(
      (image_points - image_points_hat).squaredNorm() / image_points.cols());

  return reprojection_error;
}

//@brief Estimate the camera matrix using DLT algorithm from the given point
// correspondences.
void DLT(const Eigen::Ref<const Eigen::Matrix2Xd>& image_points,
         const Eigen::Ref<const Eigen::Matrix3Xd>& object_points,
         Eigen::Matrix<double, 3, 4>* camera_matrix) {
  // Construct the coefficient matrix A containing the constraints introduced by
  // the 2D-3D points correspondences
  const int kNumCorrespondences = static_cast<int>(image_points.cols());
  Eigen::MatrixXd A(2 * kNumCorrespondences, 12);
  for (int i = 0; i < kNumCorrespondences; ++i) {
    double x, y, w;
    x = image_points.col(i).x();
    y = image_points.col(i).y();
    w = 1;

    double X, Y, Z, W;
    X = object_points.col(i).x();
    Y = object_points.col(i).y();
    Z = object_points.col(i).z();
    W = 1;

    Eigen::RowVectorXd constraint_xw(12), constraint_yw(12);
    constraint_xw << w * X, w * Y, w * Z, w * W, 0, 0, 0, 0, -x * X, -x * Y,
        -x * Z, -x * W;
    constraint_yw << 0, 0, 0, 0, -w * X, -w * Y, -w * Z, -w * W, y * X, y * Y,
        y * Z, y * W;

    A.row(2 * i) = constraint_xw;
    A.row(2 * i + 1) = constraint_yw;
  }

  // Solve the linear system equations A*P = 0 using SVD A = USV* to get the
  // approximate camera matrix \hat{P} with which the algebraic error is
  // minimized, where \hat{P} is the right null vector of A corresponding to the
  // smallest singular value.
  //
  //@note If the being decomposed matrix has dynamic-size columns, you can
  // compute the thin U and V if you'd like. Eigen does the checking based on
  // the _MatrixType template parameter passed to it. In our case, full or thin,
  // both are okay and we choose full here in spirit of the fixed-size columns
  // here.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(2 * kNumCorrespondences, 12);
  //? the computed V is transposed or not? So the smallest singular vector is
  // the last row or last col?
  // Eigen::VectorXd smallest_singular_vector = svd.matrixV().rightCols(1);
  svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Index smallest_singular_value_index;
  svd.singularValues().minCoeff(&smallest_singular_value_index);
  const Eigen::VectorXd smallest_singular_vector =
      svd.matrixV().col(smallest_singular_value_index);
  // TODO Make the code neater (possibly) using Eigen::Map or (conservative)
  // resizing technique.
  camera_matrix->row(0) =
      Eigen::RowVector4d(smallest_singular_vector.head<4>());
  camera_matrix->row(1) =
      Eigen::RowVector4d(smallest_singular_vector.segment<4>(4));
  camera_matrix->row(2) =
      Eigen::RowVector4d(smallest_singular_vector.tail<4>());

  // Normalize the camera matrix P
  camera_matrix->array() /= camera_matrix->coeff(2, 3);
}

//@brief Decompose the camera matrix into the calibration matrix K, and the
// camera pose composed of the rotation matrix R and the translation vector t
// which is also the camera center C.
//
// By construction, P = [M|-MC] = K[R|-RC], where M = KR can be decomposed with
// an RQ-decomposition, and C = t satisfies PC = 0 which means C is the
// (smallest) right null vector of P and can be found with the SVD.
void Decompose(
    const Eigen::Ref<const Eigen::Matrix<double, 3, 4>>& camera_matrix,
    Eigen::Matrix3d* calibration_matrix, Eigen::Matrix3d* rotation,
    Eigen::Vector3d* translation) {
  // Find K and R from M using QR decomposition, where M is the left three cols
  // of P.
  // For more about QR:
  //@ref https://en.wikipedia.org/wiki/QR_decomposition
  //@ref https://en.wikipedia.org/wiki/Householder_transformation
  //@ref https://en.wikipedia.org/wiki/Pivot_element
  Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 3, 3>> qr(3, 3);
  qr.compute(camera_matrix.leftCols(3));
  *calibration_matrix = qr.matrixQ();
  *rotation = qr.matrixR();

  // Find t (C) from PC = 0 using SVD on P.
  //
  //@note C is 4d vector in homogeneous representation and t is assumed a 3d
  // vector in non-homogeneous representation for this function. So the
  // hnormalized() function is applied.
  Eigen::JacobiSVD<Eigen::Matrix<double, 3, 4>> svd(3, 4);
  svd.compute(camera_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  //@warning Eigen is heavily dependent on the underlying type. hnormalized() is
  // only callable for Vector type, so you have to in advance use the colwise on
  // the expression resulted from rightCols(1) which is Matrix type.
  // *translation = svd.matrixV().rightCols(1).colwise().hnormalized();
  //
  //@note To avoid the ambiguous introduced by using Eigen::ComputeThinV or
  // Eigen::ComputeFullV, the most unambiguous method is used here: use the
  // index to fetch the singular vector corresponding to the smallest singular
  // value. Hence, we'll always use Eigen::ComputeFullV from now on.
  Eigen::Index smallest_singular_value_index;
  svd.singularValues().minCoeff(&smallest_singular_value_index);
  *translation = svd.matrixV().col(smallest_singular_value_index).hnormalized();
}

struct ResultDLT {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix<double, 3, 4> camera_matrix;

  Eigen::Matrix3d calibration_matrix;
  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;

  double reprojection_error;
};

//@brief Wrap the principal procedures of DLT algorithms.
//
// TODO Don't pass matrices by copying
ResultDLT EstimateCameraPoseDLT(Eigen::Matrix2Xd& image_points,
                                Eigen::Matrix3Xd& object_points,
                                const bool do_normalization = true) {
  Eigen::Matrix<double, 3, 4> camera_matrix;

  if (do_normalization) {
    Eigen::Affine2d T;
    Eigen::Affine3d U;
    Normalize(&image_points, &object_points, &T, &U);
    DLT(image_points, object_points, &camera_matrix);
    Denormalize(&camera_matrix, T, U);
  } else {
    DLT(image_points, object_points, &camera_matrix);
  }

  Eigen::Matrix3d calibration_matrix, rotation;
  Eigen::Vector3d translation;
  Decompose(camera_matrix, &calibration_matrix, &rotation, &translation);

  const double reprojection_error =
      ComputeReprojectionError(image_points, object_points, camera_matrix);

  // Store the estimation result
  ResultDLT res;
  res.camera_matrix = camera_matrix;
  res.calibration_matrix = calibration_matrix;
  res.rotation = rotation;
  res.translation = translation;
  res.reprojection_error = reprojection_error;
  return res;
}
#endif  // UZH_COMMON_VISION_H_