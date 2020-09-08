#ifndef UZH_BA_ALIGN_VO_TO_GROUND_TRUTH_H_
#define UZH_BA_ALIGN_VO_TO_GROUND_TRUTH_H_

#include <iostream>
#include <vector>

#include "armadillo"
#include "ba/compute_align_error.h"
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "transform/twist.h"

namespace uzh {

//@brief Align the VO estimated trajectory to the ground truth trajectory.
//@param p_V_C [3 x n] matrix where each column contains the VO estimated
// (x, y, z) coordinates of the camera in the VO estimated world frame, and n is
// the number of frames.
//@param pp_G_C [3 x n] matrix where each column contains the ground truth
// (x, y, z) coordinates of the camera in the ground truth world frame, and n is
// the number of frames.
//@return p_G_C -- [3 x n] matrix containing the aligned VO estimated trajectory
// transformed into the ground truth world frame G.
//! This function finds a [4 x 4] similarity transformation matrix transforming
//! p_V_C to p_G_C such that the trajectory error between p_G_C and pp_G_C is
//! minimized.
arma::mat /* p_G_C */
AlignVOToGroundTruth(const arma::mat& p_V_C, const arma::mat& pp_G_C) {
  if (p_V_C.empty() || pp_G_C.empty()) LOG(ERROR) << "Empty input.";
  if (p_V_C.n_rows != 3 || pp_G_C.n_rows != 3 || p_V_C.n_cols != pp_G_C.n_cols)
    LOG(ERROR) << "Invalid input.";

  // Initial guess.
  const arma::vec6 twist_init =
      uzh::SE3_to_twist<double>(arma::eye<arma::mat>(4, 4));
  const double scale_init = 1.0;

  // Parameters to be optimized.
  double sim3[7] = {twist_init[0], twist_init[1], twist_init[2], twist_init[3],
                    twist_init[4], twist_init[5], scale_init};
  for (int i = 0; i < 7; ++i) {
    std::cout << sim3[i] << '\n';
  }

  // Add residuals.
  ceres::Problem problem;
  const int num_frames = pp_G_C.n_cols;
  for (int i = 0; i < num_frames; ++i) {
    ceres::CostFunction* cost_function =
        uzh::AlignError::Create(pp_G_C.col(i), p_V_C.col(i));
    problem.AddResidualBlock(cost_function, nullptr, sim3);
  }

  // Perform optimization.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << '\n';

  // Obtain the optimized sim3.
  const arma::vec7 optimized_sim3(sim3);
  const arma::mat44 twist_G_V =
      uzh::twist_to_SE3<double>(optimized_sim3.head(6));
  const double scale_G_V = arma::as_scalar(optimized_sim3.tail(1));

  // Align the VO estimate to the ground truth using the optimized similarity
  // transformation.
  const arma::mat p_G_C =
      scale_G_V * twist_G_V(0, 0, arma::size(3, 3)) * p_V_C +
      arma::repmat(twist_G_V(0, 3, arma::size(3, 1)), 1, num_frames);

  return p_G_C;
}

}  // namespace uzh

#endif  // UZH_BA_ALIGN_VO_TO_GROUND_TRUTH_H_