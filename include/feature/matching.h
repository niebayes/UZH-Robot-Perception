#ifndef UZH_FEATURE_MATCHING_H_
#define UZH_FEATURE_MATCHING_H_

#include <algorithm>   // std::stable_sort
#include <functional>  // std::less, std::greater
#include <numeric>     // std::iota
#include <optional>    // std::optional

#include "Eigen/Core"
#include "armadillo"
#include "matlab_port/find.h"
#include "matlab_port/pdist2.h"
#include "matlab_port/scatter.h"
#include "matlab_port/unique.h"
#include "opencv2/core/eigen.hpp"
#include "transfer.h"

namespace uzh {

//@brief Match descriptors based on the Sum of Squared Distance (SSD) measure.
//@param query_descriptors [m x q] matrix where each column corresponds to a
// m-dimensional descriptor vector formed by stacking the intensities inside a
// patch and q is the number of descriptors.
//@param database_descriptors [m x p] matrix where each column corresponds to
// a m-dimensional descriptor vector and p is the number of descriptors to be
// matched against the query descriptors.
//@param matches [1 x q] row vector where the i-th column contains the column
// index of the keypoint in the database_keypoints which matches the i-th
// keypoint in the query_keypoints.
//@param distance_ratio A parameter controls the range of the acceptable
// SSD distance within which two descriptors will be viewed as matched.
void MatchDescriptors(const cv::Mat& query_descriptors,
                      const cv::Mat& database_descriptors, cv::Mat& matches_,
                      const double distance_ratio) {
  // Convert to Eigen::Matrix
  Eigen::MatrixXd query, database;
  cv::cv2eigen(query_descriptors, query);
  cv::cv2eigen(database_descriptors, database);

  // For each query descriptor, find the nearest descriptor in database
  // descriptors whose index is stored in the matches matrix and the
  // corresponding distance is stored in the distances matrix.
  arma::mat distances_arma;
  arma::umat matches_arma;
  std::tie(distances_arma, matches_arma) =
      uzh::pdist2(uzh::eigen2arma(database), uzh::eigen2arma(query),
                  uzh::EUCLIDEAN, uzh::SMALLEST_FIRST, 1);
  Eigen::MatrixXd distances = uzh::arma2eigen(distances_arma);
  Eigen::MatrixXi matches =
      (uzh::arma2eigen(arma::conv_to<arma::mat>::from(matches_arma)))
          .cast<int>();

  // Find the overall minimal non-zero distance.
  eigen_assert(distances.rows() == 1);
  const arma::rowvec dist_arma = uzh::eigen2arma(distances.row(0));
  const double min_non_zero_distance =
      dist_arma(arma::find(dist_arma > 0)).min();

  // Discard -- set to 0 -- all matches that out of the
  // distance_ratio * min_non_zero_distance range.
  matches = (distances.array() > distance_ratio * min_non_zero_distance)
                .select(0, matches);

  // Remove duplicate matches.
  std::vector<int> unique_match_indices;
  std::tie(std::ignore, unique_match_indices, std::ignore) =
      uzh::unique<double>(
          arma::vectorise(uzh::eigen2arma(matches.cast<double>())));
  Eigen::MatrixXi unique_matches(1, matches.size());
  unique_matches.setZero();
  std::vector<int> unique_mask(matches.size(), 0);
  for (int i = 0; i < unique_match_indices.size(); ++i) {
    unique_mask[unique_match_indices[i]] = 1;
  }
  for (int i = 0; i < matches.size(); ++i) {
    if (unique_mask[i]) {
      unique_matches.col(i) = matches.col(i);
    }
  }

  // Convert back to cv::Mat
  cv::eigen2cv(unique_matches, matches_);
}

//@brief Draw a line between each matched pair of keypoints.
//@param matches [1 x q] row vector where the i-th column contains the column
// index of the keypoint in the database_keypoints which matches the keypoint
// in the query_keypoints stored in the i-th column.
//@param query_keypoints [2 x q] matrix where each column contains the x and y
// coordinates of the detected keypoints in the query frame.
//@param database_keypoints [2 x n] matrix where each column contains the x
// and y coordinates of the detected keypoints in the database frames.
//@param image The image on which the plot is rendered.
//@param plot_all_keypoints If true, not only lines linking matched points
// are drawned but also all the keypoints. By default, it's set to false.
void PlotMatches(const cv::Mat& matches, const cv::Mat& query_keypoints,
                 const cv::Mat& database_keypoints, cv::Mat& image,
                 const bool plot_all_keypoints = false) {
  // Convert to arma::Mat
  const arma::urowvec matches_ =
      arma::conv_to<arma::urowvec>::from(uzh::cv2arma<int>(matches).t());
  const arma::umat query_kps =
      arma::conv_to<arma::umat>::from(uzh::cv2arma<int>(query_keypoints).t());
  const arma::umat database_kps = arma::conv_to<arma::umat>::from(
      uzh::cv2arma<int>(database_keypoints).t());

  // Plot all keypoints, query as red whilst database as blue.
  if (plot_all_keypoints) {
    //! Follow the convention, row -> y and col -> x.
    const arma::urowvec query_x = query_kps.row(1);
    const arma::urowvec query_y = query_kps.row(0);
    const arma::urowvec database_x = database_kps.row(1);
    const arma::urowvec database_y = database_kps.row(0);
    uzh::scatter(image, query_x.t(), query_y.t(), 3, {0, 0, 255}, cv::FILLED);
    uzh::scatter(image, database_x.t(), database_y.t(), 3, {255, 0, 0},
                 cv::FILLED);
  }

  // Isolate query and match indices.
  //! These indices are used to access corresponding keypoints later on.
  std::vector<int> query_indices;
  arma::uvec match_indices;
  std::tie(std::ignore, query_indices, match_indices) =
      uzh::find(arma::vectorise(matches_));
  // Extract coordinates of keypoints.
  arma::urowvec from_kp_x, from_kp_y, to_kp_x, to_kp_y;
  from_kp_x =
      query_kps(arma::uvec{1}, arma::conv_to<arma::uvec>::from(query_indices));
  from_kp_y =
      query_kps(arma::uvec{0}, arma::conv_to<arma::uvec>::from(query_indices));
  to_kp_x = database_kps(arma::uvec{1}, match_indices);
  to_kp_y = database_kps(arma::uvec{0}, match_indices);

  // Link each set of matches.
  const int num_matches = match_indices.n_elem;
  for (int i = 0; i < num_matches; ++i) {
    int from_x, from_y, to_x, to_y;
    from_x = from_kp_x(i);
    from_y = from_kp_y(i);
    to_x = to_kp_x(i);
    to_y = to_kp_y(i);
    cv::line(image, {from_x, from_y}, {to_x, to_y}, {0, 255, 0}, 2);
  }
}

}  // namespace uzh

#endif  // UZH_FEATURE_MATCHING_H_