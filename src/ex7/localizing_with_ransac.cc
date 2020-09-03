#include <functional>
#include <random>
#include <string>
#include <tuple>

#include "Eigen/Dense"
#include "armadillo"
#include "ceres/ceres.h"
#include "feature.h"
#include "google_suite.h"
#include "io.h"
#include "matlab_port.h"
#include "opencv2/opencv.hpp"
#include "pcl/visualization/pcl_plotter.h"
#include "ransac.h"
#include "transfer.h"

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath{"data/ex7/"};

  // Random seed.
  // arma::arma_rng::set_seed(1);
  arma::arma_rng::set_seed_random();

  // Part I: fit a parabola with RANSAC.
  // Create data for ParabolaRANSAC.
  const int kNumInliers = 20, kNumOutliers = 10;
  const double kNoiseRatio = 0.1;
  const arma::vec3 poly_coeffs = arma::randu<arma::vec>(3);
  // The extremum at the quadratic curve is at x = -b / 2a.
  const double x_extremum = -poly_coeffs(1) / (2 * poly_coeffs(0));
  const double y_lowest =
      arma::as_scalar(arma::polyval(poly_coeffs, arma::vec{x_extremum}));
  // Bound x in the range of length 1.0
  const double x_span = 1.0;
  const double x_start = x_extremum - 0.5;  // x_end = x_extremeum + 0.5
  const double y_highest =
      arma::as_scalar(arma::polyval(poly_coeffs, arma::vec{x_start}));
  const double y_span = y_highest - y_lowest;
  // Random sample x coordinates.
  const arma::rowvec x = arma::randu<arma::rowvec>(kNumInliers) + x_start;
  // Compute corresponding y coordinates.
  arma::rowvec y = arma::polyval(poly_coeffs, x);
  // The data is perturbed by a factor of at most 0.1 along y direction.
  const double max_noise = kNoiseRatio * y_span;
  y += (arma::randu<arma::rowvec>(y.size()) - 0.5) * 2.0 * max_noise;
  arma::mat data(2, kNumInliers + kNumOutliers);
  data.row(0) =
      arma::join_horiz(x, arma::randu<arma::rowvec>(kNumOutliers) + x_start);
  data.row(1) = arma::join_horiz(
      y, arma::randu<arma::rowvec>(kNumOutliers) * y_span + y_lowest);

  // Fit a model with RANSAC.
  arma::mat best_poly_coeffs;
  arma::urowvec max_inlier_counts;
  std::tie(best_poly_coeffs, max_inlier_counts) =
      uzh::ParabolaRANSAC(data, max_noise);

  //   // Compare various fits with RANSAC.
  //   pcl::visualization::PCLPlotter::Ptr plotter(
  //       new pcl::visualization::PCLPlotter);
  //   plotter->setTitle("Fitted parabolas");
  //   plotter->setShowLegend(true);
  //   plotter->setXRange(x_start, x_start + x_span);
  //   plotter->setYRange(y_lowest - max_noise, y_highest + max_noise);

  //   // Plot scattered data points.
  //   plotter->addPlotData(arma::conv_to<std::vector<double>>::from(data.row(0)),
  //                        arma::conv_to<std::vector<double>>::from(data.row(1)),
  //                        "data points", vtkChart::POINTS);
  //   // Plot ground truth.
  //   const arma::vec x_plot =
  //       arma::linspace<arma::vec>(x_start, x_start + x_span, 100);
  //   plotter->addPlotData(arma::conv_to<std::vector<double>>::from(x_plot),
  //                        arma::conv_to<std::vector<double>>::from(
  //                            arma::polyval(poly_coeffs, x_plot)),
  //                        "ground truth");
  //   // Plot full fit with all the data.
  //   const arma::vec full_fit = arma::polyfit(data.row(0), data.row(1), 2);
  //   plotter->addPlotData(
  //       arma::conv_to<std::vector<double>>::from(x_plot),
  //       arma::conv_to<std::vector<double>>::from(arma::polyval(full_fit,
  //       x_plot)), "full data fit");
  //   // Plot RANSAC fitted at each iteration except the last one.
  //   for (int i = 0; i < best_poly_coeffs.n_cols - 1; ++i) {
  //     const arma::vec coeffs = best_poly_coeffs.col(i);
  //     plotter->addPlotData(
  //         arma::conv_to<std::vector<double>>::from(x_plot),
  //         arma::conv_to<std::vector<double>>::from(arma::polyval(coeffs,
  //         x_plot)),
  //         "");
  //   }
  //   // Plot RANSAC fitted result.
  //   const arma::vec best_coeffs = best_poly_coeffs.tail_cols(1);
  //   plotter->addPlotData(arma::conv_to<std::vector<double>>::from(x_plot),
  //                        arma::conv_to<std::vector<double>>::from(
  //                            arma::polyval(best_coeffs, x_plot)),
  //                        "RANSAC result");
  //   plotter->plot();

  //   // Plot the maximum inlier counts at eack iteration.
  //   pcl::visualization::PCLPlotter::Ptr inlier_cnt_plotter(
  //       new pcl::visualization::PCLPlotter);
  //   inlier_cnt_plotter->setTitle("Maximum inliers count");
  //   inlier_cnt_plotter->setXTitle("iteration");
  //   inlier_cnt_plotter->setYTitle("Number of inliers");
  //   const arma::vec iterations = arma::linspace<arma::vec>(
  //       1, max_inlier_counts.size(), max_inlier_counts.size());
  //   inlier_cnt_plotter->addPlotData(
  //       arma::conv_to<std::vector<double>>::from(iterations),
  //       arma::conv_to<std::vector<double>>::from(max_inlier_counts),
  //       "max inliers");
  //   inlier_cnt_plotter->plot();

  //   // Compare the RMS error of the full fit and that of RANSAC fit.
  //   const arma::vec x_rms = arma::linspace<arma::vec>(0.0, 1.0, 100);
  //   const double full_fit_rms =
  //       std::sqrt(uzh::arma2eigen(arma::polyval(poly_coeffs, x_rms) -
  //                                 arma::polyval(full_fit, x_rms))
  //                     .squaredNorm() /
  //                 best_poly_coeffs.n_cols);
  //   const double ransac_fit_rms =
  //       std::sqrt(uzh::arma2eigen(arma::polyval(poly_coeffs, x_rms) -
  //                                 arma::polyval(best_coeffs, x_rms))
  //                     .squaredNorm() /
  //                 best_poly_coeffs.n_cols);
  //   std::cout << "Full fit RMS: " << full_fit_rms << '\n';
  //   std::cout << "RANSAC fit RMS: " << ransac_fit_rms << '\n';

  // Part II: localizing with RANSAC.
  // Load data.
  const arma::Mat<int> database_keypoints_arma =
      uzh::LoadArma<int>(kFilePath + "keypoints.txt").t();
  //! Note the keypoints are stored in (row, col) layout in the file. To pass
  //! them to the DescribeKeypoints function, the (row, col) should be filped to
  //! (x, y).
  // TODO(bayes) Always use (row, col) layout except drawing.
  const cv::Mat database_keypoints =
      uzh::arma2cv<int>(arma::flipud(database_keypoints_arma));
  const arma::mat p_W_landmarks =
      uzh::LoadArma<double>(kFilePath + "p_W_landmarks.txt");
  const arma::mat K = uzh::LoadArma<double>(kFilePath + "K.txt");
  cv::Mat img_0 = cv::imread(kFilePath + "000000.png", cv::IMREAD_COLOR);
  cv::Mat img_1 = cv::imread(kFilePath + "000001.png", cv::IMREAD_COLOR);
  cv::Mat database_image, query_image;
  cv::cvtColor(img_0, database_image, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img_1, query_image, cv::COLOR_BGR2GRAY);

  // Port parameters from ex3.
  const int kPatchSize = 9;
  const double kHarrisKappa = 0.08;
  const int kNonMaximumRadius = 8;
  const int kDescriptorPatchRadius = 9;
  const double kDistanceRatio = 5;

  const int kNumKeypoints = 1000;

  // Detect Harris keypoints in the query image.
  cv::Mat harris_response;
  uzh::HarrisResponse(query_image, harris_response, kPatchSize, kHarrisKappa);
  cv::Mat query_keypoints;
  uzh::SelectKeypoints(harris_response, query_keypoints, kNumKeypoints,
                       kNonMaximumRadius);
  // Describe keypoints.
  cv::Mat database_descriptors, query_descriptors;
  uzh::DescribeKeypoints(database_image, database_keypoints,
                         database_descriptors, kDescriptorPatchRadius);
  uzh::DescribeKeypoints(query_image, query_keypoints, query_descriptors,
                         kDescriptorPatchRadius);
  // Match descriptors.
  cv::Mat matches_tmp;
  uzh::MatchDescriptors(query_descriptors, database_descriptors, matches_tmp,
                        kDistanceRatio);
  // Obtain matched query keypoints and corresponding landmarks.
  // Convert from cv::Mat to arma::Mat
  const arma::umat query_keypoints_tmp =
      arma::conv_to<arma::umat>::from(uzh::cv2arma<int>(query_keypoints).t());
  const arma::urowvec all_matches =
      arma::conv_to<arma::urowvec>::from(uzh::cv2arma<int>(matches_tmp).t());
  const arma::umat matched_query_keypoints =
      query_keypoints_tmp.cols(arma::find(all_matches > 0));
  //! The result of linear indexing is always a column vector in Armadillo.
  const arma::urowvec corresponding_matches =
      all_matches(arma::find(all_matches > 0)).t();
  const arma::mat corresponding_landmarks =
      p_W_landmarks.rows(corresponding_matches).t();

  // Use these matched 3D-2D correspondences to find best Pose and inliers using
  // RANSAC.
  arma::mat33 R_C_W;
  arma::vec3 t_C_W;
  arma::urowvec inlier_mask, max_num_inliers_history;
  arma::rowvec num_iterations_history;
  std::tie(R_C_W, t_C_W, inlier_mask, max_num_inliers_history,
           num_iterations_history) =
      uzh::RANSACLocalization(matched_query_keypoints, corresponding_landmarks,
                              K);
  // Show the result.
  arma::mat44 T_C_W(arma::fill::eye);
  T_C_W(0, 0, arma::size(3, 3)) = R_C_W;
  T_C_W(0, 3, arma::size(3, 1)) = t_C_W;
  T_C_W.print("Found T_C_W:");
  std::cout << "Estimated inlier ratio is: "
            << arma::size(arma::nonzeros(inlier_mask)).n_rows /
                   double(inlier_mask.size())
            << '\n';

  // Show all keypoints and all matches.
  cv::Mat query_image_show_1 = img_1.clone();
  uzh::scatter(query_image_show_1, query_keypoints_tmp.row(0).as_col(),
               query_keypoints_tmp.row(1).as_col(), 3, {0, 0, 255}, cv::FILLED);
  uzh::PlotMatches(
      uzh::arma2cv<int>(arma::conv_to<arma::Mat<int>>::from(all_matches)).t(),
      query_keypoints, database_keypoints, query_image_show_1);
  cv::imshow("All keypoints and matches", query_image_show_1);
  cv::waitKey(0);

  // Show inlier and outlier matches along with the corresponding keypoints.
  cv::Mat query_image_show_2 = img_1.clone();
  // Outliers
  uzh::scatter(
      query_image_show_2,
      matched_query_keypoints(arma::uvec{0}, arma::find(1 - inlier_mask > 0))
          .as_col(),
      matched_query_keypoints(arma::uvec{1}, arma::find(1 - inlier_mask > 0))
          .as_col(),
      3, {0, 0, 255}, cv::FILLED);
  // Inliers
  uzh::scatter(
      query_image_show_2,
      matched_query_keypoints(arma::uvec{0}, arma::find(inlier_mask > 0))
          .as_col(),
      matched_query_keypoints(arma::uvec{1}, arma::find(inlier_mask > 0))
          .as_col(),
      3, {255, 0, 0}, cv::FILLED);
  uzh::PlotMatches(
      uzh::arma2cv<int>(arma::conv_to<arma::Mat<int>>::from(
          corresponding_matches(arma::find(inlier_mask > 0)))),
      uzh::arma2cv<int>(arma::conv_to<arma::Mat<int>>::from(
          matched_query_keypoints.cols(arma::find(inlier_mask > 0)))),
      database_keypoints, query_image_show_2);
  cv::imshow("Inlier and outlier matches along with the keypoints",
             query_image_show_2);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}