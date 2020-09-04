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
// #include "pcl/visualization/pcl_plotter.h"
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
  //! Note the keypoints are stored in (row, col) layout in the file, and the
  //! indices start from 1 rather than 0.
  const cv::Mat database_keypoints_cv =
      uzh::arma2cv<int>(uzh::LoadArma<int>(kFilePath + "keypoints.txt") - 1)
          .t();
  const arma::mat p_W_landmarks =
      uzh::LoadArma<double>(kFilePath + "p_W_landmarks.txt").t();
  const arma::mat K = uzh::LoadArma<double>(kFilePath + "K.txt");

  // Load images.
  cv::Mat database_image =
      cv::imread(kFilePath + "000000.png", cv::IMREAD_GRAYSCALE);
  cv::Mat query_image =
      cv::imread(kFilePath + "000001.png", cv::IMREAD_GRAYSCALE);

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
  cv::Mat query_keypoints_cv;
  uzh::SelectKeypoints(harris_response, query_keypoints_cv, kNumKeypoints,
                       kNonMaximumRadius);
  // Describe keypoints.
  cv::Mat database_descriptors, query_descriptors;
  uzh::DescribeKeypoints(database_image, database_keypoints_cv,
                         database_descriptors, kDescriptorPatchRadius);
  uzh::DescribeKeypoints(query_image, query_keypoints_cv, query_descriptors,
                         kDescriptorPatchRadius);
  // Match descriptors.
  cv::Mat matches_cv;
  uzh::MatchDescriptors(query_descriptors, database_descriptors, matches_cv,
                        kDistanceRatio);
  // Obtain matched query keypoints and corresponding landmarks.
  // Convert from cv::Mat to arma::Mat
  const arma::umat query_keypoints_arma = arma::conv_to<arma::umat>::from(
      uzh::cv2arma<int>(query_keypoints_cv).t());
  const arma::urowvec all_matches =
      arma::conv_to<arma::urowvec>::from(uzh::cv2arma<int>(matches_cv).t());
  const arma::umat matched_query_keypoints =
      query_keypoints_arma.cols(arma::find(all_matches > 0));
  //! The result of linear indexing is always a column vector in Armadillo.
  const arma::urowvec corresponding_matches =
      all_matches(arma::find(all_matches > 0)).as_row();
  const arma::mat corresponding_landmarks =
      p_W_landmarks.cols(corresponding_matches);

  // Use these matched 3D-2D correspondences to find pose and best inlier
  // matches using RANSAC.
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
  const double inlier_ratio =
      100 * uzh::nnz<arma::uword>(inlier_mask) / double(inlier_mask.size());

  // Show the history of maximum number of inliers and number of iterations at
  // each iteration.
  // const int num_iterations = num_iterations_history.n_cols;
  // pcl::visualization::PCLPlotter::Ptr max_num_inliers_plot(
  //     new pcl::visualization::PCLPlotter);
  // max_num_inliers_plot->setTitle("Maximum number of inliers");
  // max_num_inliers_plot->setXTitle("iterations");
  // max_num_inliers_plot->setShowLegend(true);
  // max_num_inliers_plot->addPlotData(
  //     arma::conv_to<std::vector<double>>::from(
  //         arma::linspace<arma::urowvec>(1, num_iterations, num_iterations)),
  //     arma::conv_to<std::vector<double>>::from(max_num_inliers_history),
  //     "inliers", vtkChart::BAR);
  // max_num_inliers_plot->spinOnce(2000);
  // max_num_inliers_plot->clearPlots();

  // pcl::visualization::PCLPlotter::Ptr num_iterations_plot(
  //     new pcl::visualization::PCLPlotter);
  // num_iterations_plot->setTitle("Number of iterations");
  // max_num_inliers_plot->setXTitle("iterations");
  // max_num_inliers_plot->setShowLegend(true);
  // num_iterations_plot->addPlotData(
  //     arma::conv_to<std::vector<double>>::from(
  //         arma::linspace<arma::urowvec>(1, num_iterations, num_iterations)),
  //     arma::conv_to<std::vector<double>>::from(num_iterations_history),
  //     "iterations", vtkChart::BAR);
  // num_iterations_plot->spinOnce(2000);
  // num_iterations_plot->clearPlots();

  // Show all keypoints and all matches.
  std::vector<cv::Mat> imgs_show;
  cv::Mat query_image_show_all;
  cv::cvtColor(query_image, query_image_show_all, cv::COLOR_GRAY2BGR, 3);
  uzh::scatter(query_image_show_all, query_keypoints_arma.row(1).as_col(),
               query_keypoints_arma.row(0).as_col(), 4, {0, 0, 255},
               cv::FILLED);
  uzh::PlotMatches(
      uzh::arma2cv<int>(arma::conv_to<arma::Mat<int>>::from(all_matches)).t(),
      query_keypoints_cv, database_keypoints_cv, query_image_show_all);
  imgs_show.push_back(query_image_show_all);

  // Show inlier and outlier matches along with the corresponding keypoints.
  cv::Mat query_image_show_inlier_outlier;
  cv::cvtColor(query_image, query_image_show_inlier_outlier, cv::COLOR_GRAY2BGR,
               3);
  // Outliers as red circles.
  uzh::scatter(
      query_image_show_inlier_outlier,
      matched_query_keypoints(arma::uvec{1}, arma::find(1 - inlier_mask > 0))
          .as_col(),
      matched_query_keypoints(arma::uvec{0}, arma::find(1 - inlier_mask > 0))
          .as_col(),
      4, {0, 0, 255}, cv::FILLED);
  // Inliers as blue circles.
  uzh::scatter(
      query_image_show_inlier_outlier,
      matched_query_keypoints(arma::uvec{1}, arma::find(inlier_mask > 0))
          .as_col(),
      matched_query_keypoints(arma::uvec{0}, arma::find(inlier_mask > 0))
          .as_col(),
      4, {255, 0, 0}, cv::FILLED);
  uzh::PlotMatches(
      uzh::arma2cv<int>(arma::conv_to<arma::Mat<int>>::from(
          corresponding_matches(arma::find(inlier_mask > 0)))),
      uzh::arma2cv<int>(arma::conv_to<arma::Mat<int>>::from(
          matched_query_keypoints.cols(arma::find(inlier_mask > 0)))),
      database_keypoints_cv, query_image_show_inlier_outlier);

  // Superimpose the number of percentage of inliers.
  cv::putText(query_image_show_inlier_outlier,
              cv::format("Ratio of inliers: %.2f", inlier_ratio), {30, 50},
              cv::FONT_HERSHEY_PLAIN, 3, {0, 255, 0}, 5);
  imgs_show.push_back(query_image_show_inlier_outlier);
  cv::imshow(
      "All matches vs. inlier matches found with RANSAC (first two frame)",
      uzh::MakeCanvas(imgs_show, 3 * query_image.rows, 2));
  cv::waitKey(0);

  // Apply RANSAC to all frames
  // Every subsequent frame is matched against the first frame.
  const int kNumFrames = 8;
  const std::string winname{"All matches vs. inlier matches found with RANSAC"};
  for (int i = 1; i < kNumFrames + 2; ++i) {
    cv::Mat query_img = cv::imread(
        cv::format((kFilePath + "%06d.png").c_str(), i), cv::IMREAD_GRAYSCALE);

    // Detect Harris keypoints in the query image.
    cv::Mat harris_res;
    uzh::HarrisResponse(query_img, harris_res, kPatchSize, kHarrisKappa);
    cv::Mat query_kpts_cv;
    uzh::SelectKeypoints(harris_res, query_kpts_cv, kNumKeypoints,
                         kNonMaximumRadius);
    // Describe keypoints.
    cv::Mat query_descs;
    uzh::DescribeKeypoints(query_img, query_kpts_cv, query_descs,
                           kDescriptorPatchRadius);
    // Match descriptors.
    cv::Mat matches_cv_frame_i;
    uzh::MatchDescriptors(query_descs, database_descriptors, matches_cv_frame_i,
                          kDistanceRatio);
    // Obtain matched query keypoints and corresponding landmarks.
    // Convert from cv::Mat to arma::Mat
    const arma::umat query_keypoints_arma_frame_i =
        arma::conv_to<arma::umat>::from(uzh::cv2arma<int>(query_kpts_cv).t());
    const arma::urowvec all_matches_frame_i =
        arma::conv_to<arma::urowvec>::from(
            uzh::cv2arma<int>(matches_cv_frame_i).t());
    const arma::umat matched_query_kpts =
        query_keypoints_arma_frame_i.cols(arma::find(all_matches_frame_i > 0));
    //! The result of linear indexing is always a column vector in Armadillo.
    const arma::urowvec corresponding_matches_frame_i =
        all_matches_frame_i(arma::find(all_matches_frame_i > 0)).as_row();
    const arma::mat corresponding_landmarks_frame_i =
        p_W_landmarks.cols(corresponding_matches_frame_i);

    // Use these matched 3D-2D correspondences to find pose and best inlier
    // matches using RANSAC.
    arma::urowvec inlier_mask_frame_i;
    std::tie(std::ignore, std::ignore, inlier_mask_frame_i, std::ignore,
             std::ignore) =
        uzh::RANSACLocalization(matched_query_kpts,
                                corresponding_landmarks_frame_i, K);

    // Show the result only if RANSAC succeeds.
    if (inlier_mask_frame_i.empty()) {
      std::cout << "RANSAC failed at frame " << i << '\n';
      continue;
    }

    // Obtain the percentage of inliers.
    const double inlier_ratio_frame_i =
        100 * uzh::nnz<arma::uword>(inlier_mask_frame_i) /
        double(inlier_mask_frame_i.size());

    // Show all keypoints and all matches.
    std::vector<cv::Mat> imgs_show_frame_i;
    cv::Mat query_image_show_all_frame_i;
    cv::cvtColor(query_img, query_image_show_all_frame_i, cv::COLOR_GRAY2BGR,
                 3);
    uzh::scatter(query_image_show_all_frame_i,
                 query_keypoints_arma_frame_i.row(1).as_col(),
                 query_keypoints_arma_frame_i.row(0).as_col(), 4, {0, 0, 255},
                 cv::FILLED);
    uzh::PlotMatches(
        uzh::arma2cv<int>(
            arma::conv_to<arma::Mat<int>>::from(all_matches_frame_i))
            .t(),
        query_kpts_cv, database_keypoints_cv, query_image_show_all_frame_i);
    imgs_show_frame_i.push_back(query_image_show_all_frame_i);

    // Show inlier and outlier matches along with the corresponding keypoints.
    cv::Mat query_image_show_inlier_outlier_frame_i;
    cv::cvtColor(query_img, query_image_show_inlier_outlier_frame_i,
                 cv::COLOR_GRAY2BGR, 3);
    // Outliers as red circles.
    uzh::scatter(query_image_show_inlier_outlier_frame_i,
                 matched_query_kpts(arma::uvec{1},
                                    arma::find(1 - inlier_mask_frame_i > 0))
                     .as_col(),
                 matched_query_kpts(arma::uvec{0},
                                    arma::find(1 - inlier_mask_frame_i > 0))
                     .as_col(),
                 4, {0, 0, 255}, cv::FILLED);
    // Inliers as blue circles.
    uzh::scatter(
        query_image_show_inlier_outlier_frame_i,
        matched_query_kpts(arma::uvec{1}, arma::find(inlier_mask_frame_i > 0))
            .as_col(),
        matched_query_kpts(arma::uvec{0}, arma::find(inlier_mask_frame_i > 0))
            .as_col(),
        4, {255, 0, 0}, cv::FILLED);

    uzh::PlotMatches(
        uzh::arma2cv<int>(
            arma::conv_to<arma::Mat<int>>::from(corresponding_matches_frame_i(
                arma::find(inlier_mask_frame_i > 0)))),
        uzh::arma2cv<int>(arma::conv_to<arma::Mat<int>>::from(
            matched_query_kpts.cols(arma::find(inlier_mask_frame_i > 0)))),
        database_keypoints_cv, query_image_show_inlier_outlier_frame_i);

    // Superimpose the number of percentage of inliers.
    cv::putText(query_image_show_inlier_outlier_frame_i,
                cv::format("Ratio of inliers: %.2f", inlier_ratio_frame_i),
                {30, 50}, cv::FONT_HERSHEY_PLAIN, 3, {0, 255, 0}, 5);
    imgs_show_frame_i.push_back(query_image_show_inlier_outlier_frame_i);
    cv::imshow(winname,
               uzh::MakeCanvas(imgs_show_frame_i, 3 * query_img.rows, 2));
    cv::waitKey(2000);
  }

  return EXIT_SUCCESS;
}