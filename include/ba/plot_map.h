#ifndef UZH_BA_PLOT_MAP_H_
#define UZH_BA_PLOT_MAP_H_

#include <optional>
#include <vector>

#include "armadillo"
#include "glog/logging.h"
#include "pcl/visualization/pcl_plotter.h"
#include "transform/twist.h"

namespace uzh {

//@brief Plot positions of landmarks the cameras.
//@param plotter Plot where the positions of landmarks and cameras will be
// rendered on.
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
//@param range [4 x 1] column vector where the first two elements are the lower
// and upper bound of the x range respectively and the last two elements are the
// lower and upper bound of the y range respectively.
void PlotMap(pcl::visualization::PCLPlotter* plotter,
             const arma::vec& hidden_state, const arma::vec& observations,
             const arma::vec4& range = arma::vec4{0.0, 0.0, 0.0, 0.0}) {
  if (hidden_state.empty() || observations.empty()) LOG(ERROR) << "Empty input";

  const int num_frames = observations(0), num_landmarks = observations(1);
  const arma::mat twist_W_frames =
      arma::reshape(hidden_state.head(6 * num_frames), 6, num_frames);
  const arma::mat p_W_landmarks = arma::reshape(
      hidden_state.subvec(6 * num_frames, hidden_state.n_elem - 1), 3,
      num_landmarks);

  // Extract position from twist.
  arma::mat p_W_frames(3, num_frames, arma::fill::zeros);
  for (int i = 0; i < num_frames; ++i) {
    p_W_frames.col(i) = uzh::twist_to_SE3<double>(twist_W_frames.col(i))(
        0, 3, arma::size(3, 1));
  }

  // Add data to the plot.
  if (arma::any(range)) {
    plotter->setXRange(range(0), range(1));
    plotter->setYRange(range(2), range(3));
  }
  plotter->addPlotData(
      arma::conv_to<std::vector<double>>::from(p_W_landmarks.row(2)),
      arma::conv_to<std::vector<double>>::from(-p_W_landmarks.row(0)),
      "landmarks", vtkChart::POINTS);
  plotter->addPlotData(
      arma::conv_to<std::vector<double>>::from(p_W_frames.row(2)),
      arma::conv_to<std::vector<double>>::from(-p_W_frames.row(0)), "cameras",
      vtkChart::POINTS);
}

}  // namespace uzh

#endif  // UZH_BA_PLOT_MAP_H_