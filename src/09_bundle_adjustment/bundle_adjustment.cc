#include <string>
#include <tuple>

#include "armadillo"
#include "ba.h"
#include "ceres/ceres.h"
#include "google_suite.h"
#include "io.h"
#include "matlab_port.h"
#include "opencv2/opencv.hpp"
#include "pcl/visualization/pcl_plotter.h"
#include "transform.h"

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string file_path{"data/09_bundle_adjustment/"};

  // Load data.
  // Loaded hidden state and observations are obtained from a visual odometry
  // system, while loaded poses are the ground truth.
  const arma::vec hidden_state_all =
      uzh::LoadArma<double>(file_path + "hidden_state.txt");
  const arma::vec observations_all =
      uzh::LoadArma<double>(file_path + "observations.txt");
  // FIXME What is the data format of the poses.txt?
  const arma::mat poses = uzh::LoadArma<double>(file_path + "poses.txt");
  const arma::mat K = uzh::LoadArma<double>(file_path + "K.txt");

  // Get the ground truth trajectory.
  // The first p states that the poses (the second p) are the ground truth. The
  // poses in consider contain only translation part. G is the world frame of
  // the ground truth trajectory and C is the camera frame.
  const arma::mat pp_G_C_all = poses.cols(arma::uvec{3, 7, 11}).t();

  // Prepare data.
  // Take the first 150 frames as the whole problem.
  const int kNumFrames = 150;
  arma::vec hidden_state, observations;
  arma::mat pp_G_C;
  std::tie(hidden_state, observations, pp_G_C) = uzh::CropProblem(
      hidden_state_all, observations_all, pp_G_C_all, kNumFrames);
  // Take the frist 4 frames as the cropped problem.
  const int kNumFramesForSmallBA = 4;
  arma::vec cropped_hidden_state, cropped_observations;
  std::tie(cropped_hidden_state, cropped_observations, std::ignore) =
      uzh::CropProblem(hidden_state_all, observations_all, pp_G_C_all,
                       kNumFramesForSmallBA);

  // Part I: trajectory alignment.
  // Compare estimates from VO to ground truth.
  // V stands for VO.
  const arma::mat twists_V_C =
      arma::reshape(hidden_state.head(6 * kNumFrames), 6, kNumFrames);
  arma::mat p_V_C(3, kNumFrames, arma::fill::zeros);
  for (int i = 0; i < kNumFrames; ++i) {
    p_V_C.col(i) =
        uzh::twist_to_SE3<double>(twists_V_C.col(i))(0, 3, arma::size(3, 1));
  }

  // Plot the VO trajectory and ground truth trajectory.
  pcl::visualization::PCLPlotter* plotter(
      new pcl::visualization::PCLPlotter);
  plotter->setTitle("VO trajectory and ground truth trajectory");
  plotter->setShowLegend(true);
  plotter->setXRange(-5, 95);
  plotter->setYRange(-30, 10);
  //! The figure axes are not the same with the camera axes.
  //! For camera, positive x axis points to right of the camere, positive y
  //! axis points to the down of the camera and positive z axis points to the
  //! front of the camera.
  //! Therefore, x and z are the two axes needed for plotting in 2D figure,  and
  //! x needs to be negated to be consistent with the x axis of the figure.
  plotter->addPlotData(arma::conv_to<std::vector<double>>::from(pp_G_C.row(2)),
                       arma::conv_to<std::vector<double>>::from(-pp_G_C.row(0)),
                       "Ground truth", vtkChart::LINE);
  plotter->addPlotData(arma::conv_to<std::vector<double>>::from(p_V_C.row(2)),
                       arma::conv_to<std::vector<double>>::from(-p_V_C.row(0)),
                       "Original estimate", vtkChart::LINE);
  plotter->plot();

  // Apply non-linear least square (NLLS) method to align VO estimate to the
  // ground truth.
  const arma::mat p_G_C = uzh::AlignVOToGroundTruth(p_V_C, pp_G_C);
//   Show the optimized trajectory.
  plotter->addPlotData(
      arma::conv_to<std::vector<double>>::from(p_G_C.row(2)),
      arma::conv_to<std::vector<double>>::from(-p_G_C.row(0)),
      "Aligned estimate", vtkChart::LINE);
  plotter->plot();

  // Part II: small bundle adjustment.
  // Part III: determine jacob pattern.
  // FIXME Seems no need to perform this using ceres?
  // Plot map before BA.
  plotter->clearPlots();
  plotter->setTitle("Cropped problem before bundle adjustment");
  uzh::PlotMap(plotter, cropped_hidden_state, cropped_observations,
               {0, 20, -5, 5});
  plotter->plot();
  // Run BA and plot.
  const arma::vec optimized_cropped_hidden_state =
      uzh::RunBA(cropped_hidden_state, cropped_observations, K);
  plotter->clearPlots();
  plotter->setTitle("Cropped problem after bundle adjustment");
  uzh::PlotMap(plotter, optimized_cropped_hidden_state, cropped_observations,
               {0, 20, -5, 5});
  plotter->plot();

  // Part IV: larger bundle adjustment and evaluation.
  // Plot full map before BA.
  plotter->clearPlots();
  plotter->setTitle("Full problem before bundle adjustment");
  uzh::PlotMap(plotter, hidden_state, observations, {0, 40, -10, 10});
  plotter->plot();
  // Run BA and plot.
  const arma::vec optimized_full_hidden_state =
      uzh::RunBA(hidden_state, observations, K);
  plotter->clearPlots();
  plotter->setTitle("Full problem after bundle adjustment");
  uzh::PlotMap(plotter, optimized_full_hidden_state, observations,
               {0, 40, -10, 10});
  plotter->plot();

  // Evaluate the BA performance by plotting.
  const arma::mat optimized_twists_V_C = arma::reshape(
      optimized_full_hidden_state.head(6 * kNumFrames), 6, kNumFrames);
  arma::mat optimized_p_V_C(3, kNumFrames, arma::fill::zeros);
  for (int i = 0; i < kNumFrames; ++i) {
    optimized_p_V_C.col(i) = uzh::twist_to_SE3<double>(
        optimized_twists_V_C.col(i))(0, 3, arma::size(3, 1));
  }
  const arma::mat optimized_p_G_C =
      uzh::AlignVOToGroundTruth(optimized_p_V_C, pp_G_C);
  // Plot for comparision.
  plotter->clearPlots();
  plotter->setTitle("Optimized VO trajectory and ground truth trajectory");
  plotter->setShowLegend(true);
  plotter->setXRange(-5, 95);
  plotter->setYRange(-30, 10);
  plotter->addPlotData(arma::conv_to<std::vector<double>>::from(pp_G_C.row(2)),
                       arma::conv_to<std::vector<double>>::from(-pp_G_C.row(0)),
                       "Ground truth", vtkChart::LINE);
  plotter->addPlotData(arma::conv_to<std::vector<double>>::from(p_G_C.row(2)),
                       arma::conv_to<std::vector<double>>::from(-p_G_C.row(0)),
                       "Aligned original estimate", vtkChart::LINE);
  plotter->addPlotData(
      arma::conv_to<std::vector<double>>::from(optimized_p_G_C.row(2)),
      arma::conv_to<std::vector<double>>::from(-optimized_p_G_C.row(0)),
      "Aligned optimized estimate", vtkChart::LINE);
  plotter->plot();

  return EXIT_SUCCESS;
}