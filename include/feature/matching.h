#ifndef UZH_FEATURE_MATCHING_H_
#define UZH_FEATURE_MATCHING_H_

#include <algorithm>   // std::stable_sort
#include <functional>  // std::less, std::greater
#include <numeric>     // std::iota
#include <optional>    // std::optional

#include "Eigen/Core"
#include "common/tools.h"
#include "common/type.h"
#include "matlab_port/find.h"
#include "matlab_port/pdist2.h"
#include "matlab_port/scatter.h"
#include "matlab_port/unique.h"
#include "opencv2/core/eigen.hpp"

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
  Eigen::MatrixXd distances;
  Eigen::MatrixXi matches;
  uzh::pdist2(database, query, &distances, uzh::EUCLIDEAN, &matches,
              uzh::SMALLEST_FIRST, 1);

  // Find the overall minimal non-zero distance.
  //@note This could also be accomplished with std::sort / std::statble in
  // juction with std::find_if
  eigen_assert(distances.rows() == 1);
  Eigen::RowVectorXd dist = distances.row(0);
  const double kMinNonZeroDistance =
      find_min_if_not<double>(dist, [](double x) { return x <= 0; });

  // Discard -- set to 0 -- all matches that out of the
  // distance_ratio * kMinNonZeroDistance range.
  matches = (distances.array() > distance_ratio * kMinNonZeroDistance)
                .select(0, matches);

  // Remove duplicate matches.
  std::vector<int> unique_match_indices;
  std::tie(std::ignore, unique_match_indices, std::ignore) =
      uzh::unique(matches.cast<double>().reshaped());
  Eigen::MatrixXi unique_matches(1, matches.size());
  unique_matches.setZero();
  unique_matches.reshaped()(unique_match_indices) =
      matches.reshaped()(unique_match_indices);

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
void PlotMatches(const cv::Mat& matches, const cv::Mat& query_keypoints,
                 const cv::Mat& database_keypoints, cv::Mat& image) {
  // Convert to Eigen::Matrix
  Eigen::MatrixXi matches_;
  Eigen::MatrixXi query_kps, database_kps;
  cv::cv2eigen(matches, matches_);
  cv::cv2eigen(query_keypoints, query_kps);
  cv::cv2eigen(database_keypoints, database_kps);

  // Plot the keypoints, query as red whilst database as blue.
  const Eigen::VectorXi query_x = query_kps.row(0);
  const Eigen::VectorXi query_y = query_kps.row(1);
  const Eigen::VectorXi database_x = database_kps.row(0);
  const Eigen::VectorXi database_y = database_kps.row(1);
  uzh::scatter(image, query_x, query_y, 3, {0, 0, 255}, cv::FILLED);
  uzh::scatter(image, database_x, database_y, 3, {255, 0, 0}, cv::FILLED);

  // Isolate query and match indices.
  //! These indices are used to access corresponding keypoints later on.
  std::vector<int> query_indices;
  Eigen::ArrayXi match_indices;
  std::tie(std::ignore, query_indices, match_indices) =
      uzh::find(matches_.reshaped());

  // Extract coordinates of keypoints.
  Eigen::RowVectorXi from_kp_x, from_kp_y, to_kp_x, to_kp_y;
  from_kp_x = query_kps(0, query_indices);
  from_kp_y = query_kps(1, query_indices);
  to_kp_x = database_kps(0, match_indices);
  to_kp_y = database_kps(1, match_indices);

  // Link each set of matches.
  const int kNumMatches = match_indices.size();
  for (int i = 0; i < kNumMatches; ++i) {
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