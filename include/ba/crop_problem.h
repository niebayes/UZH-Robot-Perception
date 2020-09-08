#ifndef UZH_BA_CROP_PROBLEM_H_
#define UZH_BA_CROP_PROBLEM_H_

#include <tuple>

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Crop the original bundle adjustment (BA) problem.
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
//@param ground_truth [3 x n] matrix representing the ground-truth trajectory of
// camera.
//@param cropped_num_frames The first num_frames frames will be keeped after
// cropping.
//@returns
// cropped_hidden_state -- Column vector that stores the cropped states.
// cropped_observations -- Column vector that stores the cropped observations.
// cropped_ground_truth -- [3 x num_frames] matrix representing the cropped
// ground-truth trajectory of camera.
std::tuple<arma::vec /* cropped_hidden_state */,
           arma::vec /* cropped_observations */,
           arma::mat /* cropped_ground_truth */>
CropProblem(const arma::vec& hidden_state, const arma::vec& observations,
            const arma::mat& ground_truth, const int cropped_num_frames) {
  if (hidden_state.empty() || observations.empty() || ground_truth.empty())
    LOG(ERROR) << "Empty input.";
  if (ground_truth.n_rows != 3 || cropped_num_frames <= 0)
    LOG(ERROR) << "Invalid input.";

  const int num_frames = observations(0);
  if (cropped_num_frames > num_frames) LOG(ERROR) << "Invalid input.";

  // Get the index of the end of the cropped observations.
  // The frist two elements are n and m as stated.
  int end = 2;
  // Also get the cropped number of landmarks. Assume the indices of the
  // landmarks increase with frame indices.
  int cropped_num_landmarks = 0;
  for (int i = 0; i < cropped_num_frames; ++i) {
    const int num_observations_frame_i = observations(end);
    if (i == cropped_num_frames - 1) {
      const arma::vec landmark_indices(num_observations_frame_i,
                                       arma::fill::none);
      cropped_num_landmarks =
          arma::max(observations.subvec(end + 1 + 2 * num_observations_frame_i,
                                        arma::size(landmark_indices)));
    }
    end += 3 * num_observations_frame_i + 1;
  }

  const arma::vec cropped_twists = hidden_state.head(6 * cropped_num_frames);
  const arma::vec cropped_landmarks_3d(3 * cropped_num_landmarks,
                                       arma::fill::none);
  const arma::vec cropped_landmarks =
      hidden_state.subvec(6 * num_frames, arma::size(cropped_landmarks_3d));
  const arma::vec cropped_hidden_state =
      arma::join_vert(cropped_twists, cropped_landmarks);

  const int num_cropped_observations = observations.subvec(2, end - 1).n_elem;
  arma::vec cropped_observations(2 + num_cropped_observations,
                                 arma::fill::zeros);
  cropped_observations(0) = cropped_num_frames;
  cropped_observations(1) = cropped_num_landmarks;
  cropped_observations.tail(num_cropped_observations) =
      observations.subvec(2, end - 1);

  const arma::mat cropped_ground_truth =
      ground_truth.head_cols(cropped_num_frames);

  return {cropped_hidden_state, cropped_observations, cropped_ground_truth};
}

}  // namespace uzh

#endif  // UZH_BA_CROP_PROBLEM_H_