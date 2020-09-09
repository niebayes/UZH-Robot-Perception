#ifndef UZH_BA_RUN_BA_H_
#define UZH_BA_RUN_BA_H_

#include "armadillo"
#include "ba/compute_reprojection_error.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

namespace uzh {

//@brief Run bundle adjustment to optimize the hidden states by minimizing the
// reprojection errors.
//@param hidden_state [6*n+3*m x 1] column vector that stores the stacked states
// involved in the BA problem, where n is the number of 6D twist vectors
// representing poses of camera and m is the number of 3D landmarks.
//@param observations Column vector that stores the information of obvervations
// of all frames. The first two elements n and m as stated above followed by
// sets of observations with each set of observations representing the
// observations of the i-th frame stored as a column vector which can be
// separated into 3 segments. The first segment is a single number k_i which
// states the number of landmarks observed in the i-th frame. The second segment
// is sets of 2D coordinates of the projections of the landmarks, specified as
// (row, col). The final segment is the indices of the landmarks corresponding
// to the observed projections.
//@param K [3 x 3] calibration matrix.
//@return optimized_hidden_state -- optimized hidden state such that the sum of
// reprojection errors between estimated projections and observations are
// minimized.
arma::vec /* optimized_hidden_state */
RunBA(const arma::vec& hidden_state, const arma::vec& observations,
      const arma::mat33& K) {
  if (hidden_state.empty() || observations.empty() || K.empty())
    LOG(ERROR) << "Empty input.";

  const int num_frames = observations(0), num_landmarks = observations(1);
  const arma::mat twist_W_C =
      arma::reshape(hidden_state.head(6 * num_frames), 6, num_frames);
  const arma::mat p_W_landmarks = arma::reshape(
      hidden_state.subvec(6 * num_frames, hidden_state.n_elem - 1), 3,
      num_landmarks);

  // Parameters to be optimized.
  arma::mat optimized_twist_W_C = twist_W_C;
  arma::mat optimized_p_W_landmarks = p_W_landmarks;

  // Build the optimization problem.
  ceres::Problem problem;
  // Get the index of the end of the cropped observations.
  // The frist two elements are n and m as stated.
  int end = 2;
  for (int i = 0; i < num_frames; ++i) {
    const int num_observations_frame_i = observations(end);
    const arma::vec observations_frame_i(2 * num_observations_frame_i,
                                         arma::fill::none);
    const arma::mat observed_kpts = arma::flipud(arma::reshape(
        observations.subvec(end + 1, arma::size(observations_frame_i)), 2,
        num_observations_frame_i));
    const arma::vec landmarks_frame_i(num_observations_frame_i,
                                      arma::fill::none);
    const arma::uvec landmark_indices = arma::conv_to<arma::uvec>::from(
        observations.subvec(end + 1 + 2 * num_observations_frame_i,
                            arma::size(landmarks_frame_i)));
    const int num_landmarks_frame_i = landmark_indices.n_elem;
    assert(num_observations_frame_i == num_landmarks_frame_i);

    // Add residual blocks introduced from corresponding pair of
    // (frame, landmark).
    for (int j = 0; j < num_landmarks_frame_i; ++j) {
      double* twist = optimized_twist_W_C.begin_col(i);
      // Minus 1 making indices start from 0.
      double* landmark =
          optimized_p_W_landmarks.begin_col(landmark_indices(j) - 1);
      ceres::CostFunction* cost_function =
          uzh::ReprojectionError::Create(observed_kpts.col(j), K);
      problem.AddResidualBlock(cost_function, nullptr, twist, landmark);
    }

    end += 3 * num_observations_frame_i + 1;
  }

  // Perform optimization.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 200;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << '\n';

  // Reconstruct hidden_state.
  const arma::vec optimized_hidden_state =
      arma::join_vert(arma::vectorise(optimized_twist_W_C),
                      arma::vectorise(optimized_p_W_landmarks));

  assert(optimized_hidden_state.n_elem == hidden_state.n_elem);
  return optimized_hidden_state;
}

}  // namespace uzh

#endif  // UZH_BA_RUN_BA_H_