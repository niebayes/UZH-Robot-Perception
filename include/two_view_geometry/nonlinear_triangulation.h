#ifndef UZH_TWO_VIEW_GEOMETRY_NONLINEAR_TRIANGULATION_H_
#define UZH_TWO_VIEW_GEOMETRY_NONLINEAR_TRIANGULATION_H_

#include <vector>

#include "arma_traits/arma_hnormalized.h"
#include "arma_traits/arma_homogeneous.h"
#include "armadillo"
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "two_view_geometry/triangulation.h"

namespace uzh {

//@brief Reprojection error functor.
struct ReprojectionError {
  // Observed image point and camera matrix are given.
  ReprojectionError(const arma::vec2& observed, const arma::mat& M)
      : observed_(observed), M_(M) {}

  // 3D scene point is the parameter to be refined.
  // The Euclidean distance between the reprojected image point and the observed
  // image point is the residual to be minimized.
  template <typename T>
  bool operator()(const T* const point, T* residual) const {
    // Extract 3D point coordinates.
    T X = T(point[0]), Y = T(point[1]), Z = T(point[2]);
    // Do camera projection in terms of element-wise operation.
    T z = T(M_(2, 0)) * X + T(M_(2, 1)) * Y + T(M_(2, 2)) * Z + T(M_(2, 3));
    T reprojected_x =
        (T(M_(0, 0)) * X + T(M_(0, 1)) * Y + T(M_(0, 2)) * Z + T(M_(0, 3))) / z;
    T reprojected_y =
        (T(M_(1, 0)) * X + T(M_(1, 1)) * Y + T(M_(1, 2)) * Z + T(M_(1, 3))) / z;

    // Compute the residual.
    residual[0] = T(observed_(0)) - reprojected_x;
    residual[1] = T(observed_(1)) - reprojected_y;

    return true;
  }

  // Factory function to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const arma::vec2& observed,
                                     const arma::mat& M) {
    // 2 = dim of residuals, 3 = dim of parameters, i.e. 3D point.
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3>(
        new ReprojectionError(observed, M)));
  }

 private:
  arma::vec2 observed_;
  arma::mat M_;
};

//@brief Nonlinear triangulation where the 3D scene points P's are computed from
// non-linear least squares by minimizing the geometric error, i.e. the SSRE
// (Sum of Squared Reprojection Error).
//@param p1 [3 x n] matrix where each column contains the homogeneous
// coordinates for an image point p_i on the left camera.
//@param p2 [3 x n] matrix where each column contains the homogeneous
// coordinates for an image point p_j on the right camera. The image points p_i
// and p_j are the correspondent projections of a same 3D scene point P.
//@param M1 [3 x 4] matrix representing the projection matrix K[I|0] for the
// left camera, where the I is a [3 x 3] identity matrix and 0 is a [3 x 1]
// vector.
//@param M2 [3 x 4] matrix representing the projection matrix K[R|t] where it is
// assumed the calibration matrices K1 = K2 = K for the left and right camera
// respectively.
//@return P -- [4 x n] matrix where each column contains the homogeneous
// coordinates
// for a 3D scene point P.
//! In practice, this function takes as initial estimate the result of
//! LinearTriangulation.
arma::mat /* P */
NonlinearTriangulation(const arma::mat& p1, const arma::mat& p2,
                       const arma::mat& M1, const arma::mat& M2) {
  if (p1.n_cols != p2.n_cols)
    LOG(ERROR) << "Number of points of p1 and p2 must be consistent.";
  if (p1.n_rows != 3 || p2.n_rows != 3)
    LOG(ERROR) << "Points must be represented as homogeneous coordinates.";
  if (M1.n_rows != 3 || M1.n_cols != 4 || M2.n_rows != 3 || M2.n_cols != 4)
    LOG(ERROR) << "Invalid projection matrix.";

  // Obtain the initial estimate of P from LinearTriangulation function.
  const arma::mat& P_init = LinearTriangulation(p1, p2, M1, M2);

  // Build a non-linear least square problem.
  arma::mat P = uzh::hnormalized<double>(P_init);
  ceres::Problem problem;
  const int kNumPoints = p1.n_cols;
  for (int i = 0; i < kNumPoints; ++i) {
    ceres::CostFunction* residual_1 =
        ReprojectionError::Create(uzh::hnormalized<double>(p1.col(i)), M1);
    ceres::CostFunction* residual_2 =
        ReprojectionError::Create(uzh::hnormalized<double>(p2.col(i)), M2);
    double* point = P.begin_col(i);
    problem.AddResidualBlock(residual_1, nullptr, point);
    problem.AddResidualBlock(residual_2, nullptr, point);
  }

  // Perform optimization.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << '\n';

  return uzh::homogeneous<double>(P);
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_NONLINEAR_TRIANGULATION_H_