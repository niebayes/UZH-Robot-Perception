#ifndef UZH_DLT_ESTIMATE_POSE_DLT_H_
#define UZH_DLT_ESTIMATE_POSE_DLT_H_

#include <optional>

#include "Eigen/Core"
#include "glog/logging.h"

namespace uzh {

//@brief Camera matrix wrapper, containing the data and some useful methods.
class CameraMatrixDLT {
 public:
  // Eigen now uses c++11 alignas keyword for static alignment. Users
  // targeting c++17 only and recent compilers (e.g., GCC>=7, clang>=5,
  // MSVC>=19.12) will thus be able to completely forget about all issues
  // related to static alignment, including EIGEN_MAKE_ALIGNED_OPERATOR_NEW.
  //@ref https://en.cppreference.com/w/cpp/language/alignas
  // TODO(bayes) Use alignas in place of eigen alignment macros.
  // alignas(CameraMatrixDLT);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CameraMatrixDLT() {
    camera_matrix_.setZero();
    calibration_matrix_.setIdentity();
    rotation_matrix_.setIdentity();
    translation_.setZero();
  }

  void setM(const Eigen::Ref<const Eigen::Matrix<double, 3, 4>>& M) {
    camera_matrix_ = M;
  }
  void setK(const Eigen::Ref<const Eigen::Matrix3d>& K) {
    calibration_matrix_ = K;
  }
  void setR(const Eigen::Ref<const Eigen::Matrix3d>& R) {
    rotation_matrix_ = R;
  }
  void sett(const Eigen::Ref<const Eigen::Vector3d>& t) { translation_ = t; }

  Eigen::Matrix<double, 3, 4> getM() const { return camera_matrix_; }
  Eigen::Matrix3d getK() const { return calibration_matrix_; }
  Eigen::Matrix3d getR() const { return rotation_matrix_; }
  Eigen::Vector3d gett() const { return translation_; }

  //@brief Decompose M = \alpha * K[R|t] to get the component K, R and t, where
  // \alpha is the unknown scale thus far.
  // The M is computed using unconstrained DLT method. Hence we will do some
  // post-processing to enforce the R and t to be decomposed is compatible with
  // the corresponding constraints.
  void DecomposeDLT() {
    if (getM().isZero()) {
      LOG(ERROR) << "The camera matrix M is not set yet";
      return;
    }
    if (!getK().isIdentity()) {  // Indicates the M is composed simply with
                                 // [R|t] and can be easily decomposed
      //! Thus far, we did not impose any constraints on solving for M.
      //! This checking ensures the R to be decomposed from Mis a valid rotation
      //! matrix belongs to SO(3) in which det(R) = +1;
      //
      //! Alternatively, by checking the sign of the t_z, aka. M(2, 3) the last
      //! entry of M, if sign(t_z) is negative, flip the sign of all the entries
      //! of M.
      if (!(getM().leftCols(3).determinant() < 0)) {
        setM(-getM());
      }
      // Extract the initial R
      const Eigen::Matrix3d R = getM().leftCols(3);

      // Enforce the det(R) = +1 constraint by projecting R to SO(3).
      // This can be done by factorizing R with SVD and set all singular values
      // to 1, whereby the corrected R_tilde is a valid rotation matrix.
      Eigen::JacobiSVD<Eigen::Matrix3d> R_svd;
      R_svd.compute(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
      const Eigen::Matrix3d R_tilde =
          R_svd.matrixU() * R_svd.matrixV().transpose();

      // Recover the previously unknown scale \alpha by computing the quotient
      // of the norm(R_tilde) wrt. norm(R);
      //
      // This is by observing the fact that when R was not corrected
      // M = \alpha * K[R|t] and after R was corrected
      // M = K[R_tilde| \alpha * t];
      const double alpha = R_tilde.norm() / R.norm();

      // Reconstruct M with the corrected R_tilde and computed scale \alpha.
      setR(R_tilde);
      sett(alpha * getM().rightCols(1));
      Eigen::Matrix<double, 3, 4> M_tilde;
      M_tilde.leftCols(3) = getR();
      M_tilde.rightCols(1) = gett();
      setM(M_tilde);
    } else {
      // TODO Implement non-indentity version CameraMatrix::DecomposeDLT
      LOG(ERROR) << "To be implemented soon!";
      return;
    }
  }

 private:
  Eigen::Matrix<double, 3, 4> camera_matrix_;
  Eigen::Matrix3d calibration_matrix_;
  Eigen::Matrix3d rotation_matrix_;
  Eigen::Vector3d translation_;
};

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
  CameraMatrixDLT M_dlt;
  M_dlt.setM(Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(
      smallest_singular_vector.data()));
  M_dlt.setK(K);

  // Normalize the camera matrix P
  // FIXME Shall we do this?
  // M.array() /= M.coeff(2, 3);
  return M_dlt;
}

}  // namespace uzh

#endif  // UZH_DLT_ESTIMATE_POSE_DLT_H_