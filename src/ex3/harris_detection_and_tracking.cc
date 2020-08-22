#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "common/plot.h"
#include "common/type.h"
#include "feature.h"
#include "glog/logging.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

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
  cv::cv2eigen(query_descriptors, database);

  std::cout << query.rows() << " " << query.cols() << '\n';
  std::cout << database.rows() << " " << database.cols() << '\n';

  // For each query descriptor, find the nearest descriptor in database
  // descriptors whose index is stored in the matches matrix and the
  // corresponding distance is stored in the distances matrix.
  Eigen::MatrixXd distances;
  Eigen::MatrixXi matches;
  PDist2(database, query, &distances, EUCLIDEAN, &matches, SMALLEST_FIRST, 1);
  std::cout << distances << '\n';
  std::cout << matches << '\n';

  // Find the overall minimal non-zero distance.
  eigen_assert(distances.rows() == 1);
  Eigen::RowVectorXd dist = distances.row(0);
  const double kMinNonZeroDistance = *std::min_element(
      dist.begin(), std::remove_if(dist.begin(), dist.end(),
                                   [](double x) { return x <= 0; }));

  // Discard -- set to 0 -- all matches that out of the
  // distance_ratio * kMinNonZeroDistance range.
  matches = (distances.array() > distance_ratio * kMinNonZeroDistance)
                .select(0, matches);

  //
}

// function plotMatches(matches, query_keypoints, database_keypoints)

// [~, query_indices, match_indices] = find(matches);

// x_from = query_keypoints(1, query_indices);
// x_to = database_keypoints(1, match_indices);
// y_from = query_keypoints(2, query_indices);
// y_to = database_keypoints(2, match_indices);
// plot([y_from; y_to], [x_from; x_to], 'g-', 'Linewidth', 3);

// end

//@brief Draw a line between each matched pair of keypoints.
//@param matches [1 x q] row vector where the i-th column contains the column
// index of the keypoint in the database_keypoints which matches the keypoint
// in the query_keypoints stored in the i-th column.
//@param query_keypoints [2 x q] matrix where each column contains the x and y
// coordinates of the detected keypoints in the query frame.
//@param database_keypoints [2 x n] matrix where each column contains the x
// and
// y coordinates of the detected keypoints in the database frames.
void PlotMatches(const cv::Mat& matches, const cv::Mat& query_keypoints,
                 const cv::Mat& database_keypoints) {
  //
}

int main(int /*argv*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath = "data/ex3/";
  cv::Mat image = cv::imread(kFilePath + "KITTI/000000.png",
                             cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);

  // Part I: compute response.
  // The shi_tomasi_response is computed as comparison whilst the
  // harris_response is used through out the remainder of this program.
  const int kPatchSize = 9;
  const double kHarrisKappa = 0.06;
  cv::Mat harris_response, shi_tomasi_response;
  HarrisResponse(image, harris_response, kPatchSize, kHarrisKappa);
  ShiTomasiResponse(image, shi_tomasi_response, 9);
  // ImageSC(harris_response);
  // ImageSC(shi_tomasi_response);

  // Part II: select keypoints
  const int kNumKeypoints = 200;
  const int kNonMaximumRadius = 8;
  cv::Mat keypoints;
  SelectKeypoints(harris_response, keypoints, kNumKeypoints, kNonMaximumRadius);
  // Superimpose the selected keypoins to the original image.
  Eigen::MatrixXd k;
  cv::cv2eigen(keypoints, k);
  const Eigen::VectorXi x = k.row(0).cast<int>(), y = k.row(1).cast<int>();
  Scatter(image, x, y, 4, {0, 0, 255});
  // cv::imshow("", image);
  // cv::waitKey(0);

  // Part III: describe keypoints
  const int kPatchRadius = 9;
  cv::Mat descriptors;
  DescribeKeypoints(image, keypoints, descriptors, kPatchRadius);

  // Show the top 16 descritors ranked by strengh of response.
  bool show_descriptors = false;
  for (int i = 0; i < 16; ++i) {
    if (show_descriptors) {
      cv::Mat descriptor = descriptors.col(i);
      // Eigen::MatrixXi d;
      // cv::cv2eigen(descriptor, d);
      // d.resize(19, 19);
      // cv::eigen2cv(d, descriptor);

      // reshaped introduced since eigen 3.4
      //@note assigning a reshaped matrix to itself is currently not supported
      // and will result to undefined-behavior because of aliasing

      // ImageSC();
    }
  }

  // Part IV: match descriptors
  const double kDistanceRatio = 4;
  cv::Mat query_image =
      cv::imread(kFilePath + "KITTI/000001.png", cv::IMREAD_GRAYSCALE);
  cv::Mat query_harris_response;
  HarrisResponse(query_image, query_harris_response, kPatchSize, kHarrisKappa);
  cv::Mat query_keypoints;
  SelectKeypoints(query_harris_response, query_keypoints, kNumKeypoints,
                  kNonMaximumRadius);
  cv::Mat query_descriptors;
  DescribeKeypoints(query_image, query_keypoints, query_descriptors,
                    kPatchRadius);
  cv::Mat matches;
  MatchDescriptors(query_descriptors, descriptors, matches, kDistanceRatio);
  PlotMatches(matches, query_keypoints, keypoints);

  // Part V: match descriptors for all 200 images in the reduced KITTI
  // dataset.
  const int kNumImages = 200;
  for (int i = 0; i < kNumImages; ++i) {
    // Wrap Part IV
  }

  // Optional: profile the program

  Eigen::VectorXd a(9);
  a << 1, 2, 3, -1, -2, 0, 0, 0, 4;

  // Imitate matlab's unique.
  std::vector<int> v = {9, 2, 9, 5};
  std::vector<int> c = {2, 5, 9};

  std::vector<int> u;
  u.reserve(v.size());

  std::transform(v.begin(), v.end(), std::back_inserter(u), [&](int x) {
    return (std::distance(c.begin(), std::lower_bound(c.begin(), c.end(), x)));
  });

  for (int x : u) std::cout << x << ' ';
  std::cout << std::endl;

  // -------------------------------------------------------------------
  std::vector<int> v = {9, 2, 9, 5};
  std::set<int> c(v.begin(), v.end());

  std::vector<int> u;
  u.reserve(v.size());

  std::transform(v.begin(), v.end(), std::back_inserter(u),
                 [&](int x) { return (std::distance(c.begin(), c.find(x))); });

  for (int x : u) std::cout << x << ' ';
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
