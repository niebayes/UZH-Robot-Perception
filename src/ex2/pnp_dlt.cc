#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>  // std::optional, feature of C++17
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "common_utils.h"
#include "google_suite.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"
// #include "unsupported/Eigen/KroneckerProduct"

//@brief Estimate camera pose from constructing a DLT (direct linear transform)
// solver.
//@param image_points [2xN] 2D observations, expressed in pixel coordinates.
//@param object_points [3xN] 3D reference points, expressed in inhomogeneous
// representation.
//@param calibration_matrix calibration matrix K. If given, pre-multiply the
// observations to get the calibrated / normalized coordinates expressed in
// camera coordinate system, where Z = 1. Note the unit is adimensional.
//@return Camera [3x4] matrix M = K[R|t]. If calibration matrix K is given in
// advance, the returned M is reduced to [R|t], compressing rigid transformation
// composed of rotation R and translation t wrt. world coordinate. I.e. this is
// T_C_W, mapping from world coordinate to camera coordinate.
CameraMatrixDLT EstimatePoseDLT(
    const Eigen::Ref<const Eigen::Matrix2Xd>& image_points,
    const Eigen::Ref<const Eigen::Matrix3Xd>& object_points,
    const std::optional<const Eigen::Ref<const Eigen::Matrix3d>>&
        calibration_matrix = std::nullopt) {
  // Construct the coefficient matrix Q containing the constraints introduced by
  // the 2D-3D points correspondences
  const int kNumCorrespondences = static_cast<int>(image_points.cols());

  //! If the calibration matrix K is given, then it's more convenient to
  //! pre-multiply the image points with the inverse of K.
  Eigen::Matrix3d K;
  if (calibration_matrix) {
    K = calibration_matrix.value();
  } else {
    // If not given, simply set K to identity, i.e. no effects at all.
    K.setIdentity();
  }

  // Incrementally populate Q.
  Eigen::MatrixXd Q(2 * kNumCorrespondences, 12);  // [2n x 12] matrix.
  for (int i = 0; i < kNumCorrespondences; ++i) {
    double x, y, w;
    // Get the calibrated coordinates, aka. normalized coornidates by
    // multiplying the inverse of the calibration matrix K.
    Eigen::Vector2d normalized =
        (K.inverse() * image_points.col(i).homogeneous()).hnormalized();
    x = normalized.x();
    y = normalized.y();
    w = 1;

    double X, Y, Z, W;
    X = object_points.col(i).x();
    Y = object_points.col(i).y();
    Z = object_points.col(i).z();
    W = 1;

    // Each correspondence contributes 2 constraints.
    Eigen::RowVectorXd constraint_xw(12), constraint_yw(12);
    constraint_xw << w * X, w * Y, w * Z, w * W, 0, 0, 0, 0, -x * X, -x * Y,
        -x * Z, -x * W;
    constraint_yw << 0, 0, 0, 0, -w * X, -w * Y, -w * Z, -w * W, y * X, y * Y,
        y * Z, y * W;

    Q.row(2 * i) = constraint_xw;
    Q.row(2 * i + 1) = constraint_yw;
  }

  // Solve the linear system equations QM = 0 using SVD Q = USV* to get the
  // approximate camera matrix \hat{M} with which the algebraic error is
  // minimized, where \hat{M} is the right null vector of A corresponding to the
  // smallest singular value.
  //
  //@note If the being decomposed matrix has dynamic-size columns, you can
  // compute the thin U and V if you'd like to. Eigen does the dynamic-ness
  // checking based on the _MatrixType template parameter passed to the
  // function.
  //
  // FIXME In our case, full or thin, both are okay and we choose full here
  // in spirit of the fixed-size columns here.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(2 * kNumCorrespondences, 12);
  //! The computed matrix V is not transposed, and Q = U \Sigma V* is the
  //! desired SVD of Q.
  svd.compute(Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::VectorXd smallest_singular_vector = svd.matrixV().rightCols(1);
  // FIXME Differ with the codes below?
  // Eigen::Index smallest_singular_value_index;
  // svd.singularValues().minCoeff(&smallest_singular_value_index);
  // Eigen::VectorXd smallest_singular_vector =
  //     svd.matrixV().col(smallest_singular_value_index);
  // TODO Make the code neater (possibly) using Eigen::Map or (conservative)
  // resizing technique.
  CameraMatrixDLT M_dlt;
  M_dlt.setM(Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(
      smallest_singular_vector.data()));
  M_dlt.setK(K);

  // Normalize the camera matrix P
  // FIXME Shall we do this?
  // M.array() /= M.coeff(2, 3);
  return M_dlt;
}

//

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  // Load data files
  const std::string kFilePath{"data/ex2/"};

  const Eigen::Matrix3d K = armaLoad<Eigen::Matrix3d>(kFilePath + "K.txt");
  const Eigen::MatrixXd observations =
      armaLoad<Eigen::MatrixXd>(kFilePath + "detected_corners.txt");
  // The p_W_corners.txt contains coordinates of 3D reference points expressed
  // in centimeters which is better to be transformed to canonical unit meter.
  const Eigen::Matrix3Xd p_W_corners =
      0.01 *
      armaLoad<Eigen::MatrixX3d>(kFilePath + "p_W_corners.txt").transpose();

  // Run DLT
  Eigen::RowVectorXd row_vec = observations.row(0);
  const Eigen::Matrix2Xd image_points =
      Eigen::Map<Eigen::Matrix<double, 2, 12>>(row_vec.data());
  CameraMatrixDLT M_dlt = EstimatePoseDLT(image_points, p_W_corners, K);
  M_dlt.DecomposeDLT();

  // Compare reprojected points and the obvervations.
  const int KImageIndex = 1;
  cv::Mat image = cv::imread(
      cv::format((kFilePath + "images_undistorted/img_%04d.jpg").c_str(),
                 KImageIndex),
      cv::IMREAD_COLOR);
  const Matrix34d M = M_dlt.getM();
  Eigen::Matrix2Xd reprojected_points;
  ReprojectPoints(p_W_corners, &reprojected_points, K, M,
                  PROJECT_WITHOUT_DISTORTION);
  const Eigen::VectorXi& reproj_x = reprojected_points.row(0).cast<int>();
  const Eigen::VectorXi& reproj_y = reprojected_points.row(1).cast<int>();
  Scatter(image, reproj_x, reproj_y, 4, {0, 0, 255});

  const Eigen::VectorXi& observed_x = image_points.row(0).cast<int>();
  const Eigen::VectorXi& observed_y = image_points.row(1).cast<int>();
  Scatter(image, observed_x, observed_y, 4, {0, 255, 0});

  cv::imshow("", image);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
