#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "Eigen/Dense"
#include "armadillo"
#include "common/plot.h"
#include "common/type.h"
#include "feature.h"
#include "glog/logging.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

//@brief Imitate matlab's unique. Unique values in an array and store them in a
// sorted order (by default descending).
//@param A One dimensional array containing the original values.
//@return C One dimensional array containing the unique values of A.
//@return ia One dimensional array containing indices such that C = A(ia).
//@return ic One dimensional array containing indices such that A = C(ic).
//
//! Eigen provides surprising flexibility and genericity, so you can pass as
//! arguments ArrayXd, VectorXd, RowVectorXd as well as MatrixXd to the
//! parameter A and the returned object C. They all work properly.
//
//@note How to implement matlab's unique using C++?
//@ref https://stackoverflow.com/q/63537619/14007680
//@note Return unmovable and uncopyable values with C++17.
//@ref https://stackoverflow.com/a/38531743/14007680
// TODO(bayes) Templatize this function to make the parameter parsing more
// flexible.
std::tuple<Eigen::ArrayXd /*C*/, std::vector<int> /*ia*/,
           std::vector<int> /*ic*/>
Unique(const Eigen::ArrayXd& A) {
  //! Alternatively, explicitly passing into pointers.
  // Eigen::ArrayXd* C,
  // std::optional<Eigen::ArrayXi*> ia = std::nullopt,
  // std::optional<Eigen::ArrayXi*> ic = std::nullopt) {

  // Copy the values of A.
  const std::vector<double> original(A.begin(), A.end());

  // Get uniqued values.
  const std::set<double> uniqued(original.begin(), original.end());
  // Eigen::ArrayXd uniqued;

  // Get ia.
  std::vector<int> indices_ori;
  indices_ori.reserve(uniqued.size());
  std::transform(uniqued.cbegin(), uniqued.cend(),
                 std::back_inserter(indices_ori), [&original](double x) {
                   return std::distance(
                       original.cbegin(),
                       std::find(original.cbegin(), original.cend(), x));
                 });

  // Get ic.
  std::vector<int> indices_uni;
  indices_uni.reserve(original.size());
  std::transform(original.cbegin(), original.cend(),
                 std::back_inserter(indices_uni), [&uniqued](double x) {
                   return std::distance(
                       uniqued.cbegin(),
                       std::find(uniqued.cbegin(), uniqued.cend(), x));
                 });

  // Output C.
  Eigen::ArrayXd C_out(uniqued.size());
  std::copy(uniqued.cbegin(), uniqued.cend(), C_out.begin());

  return {C_out, indices_ori, indices_uni};
}

//@brief Wrap STL's std::min_element and std::remove_if. Find the minimum
// not satisfying the given rule.
template <typename T, typename Derived>
T find_min_if_not(const Eigen::DenseBase<Derived>& X,
                  std::function<bool(typename Derived::Scalar)> pred) {
  // Derived X_(X.size());
  // std::copy(X.cbegin(), X.cend(), X_.template begin());
  // return (*std::min_element(
  //     X_.template begin(),
  //     std::remove_if(X_.template begin(), X_.template end(), pred)));
  Derived X_(X.size());
  std::copy(X.cbegin(), X.cend(), X_.begin());
  return (*std::min_element(X_.begin(),
                            std::remove_if(X_.begin(), X_.end(), pred)));
}

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

  // For each query descriptor, find the nearest descriptor in database
  // descriptors whose index is stored in the matches matrix and the
  // corresponding distance is stored in the distances matrix.
  Eigen::MatrixXd distances;
  Eigen::MatrixXi matches;
  PDist2(database, query, &distances, EUCLIDEAN, &matches, SMALLEST_FIRST, 1);

  // Find the overall minimal non-zero distance.
  //@note This could also be accomplished with std::sort / std::statble in
  // juction with std::find_if
  eigen_assert(distances.rows() == 1);
  Eigen::RowVectorXd dist = distances.row(0);
  const double kMinNonZeroDistance = *std::min_element(
      dist.begin(), std::remove_if(dist.begin(), dist.end(),
                                   [](double x) { return x <= 0; }));

  const double min =
      find_min_if_not<double>(dist, [](double x) { return x <= 0; });

  // Discard -- set to 0 -- all matches that out of the
  // distance_ratio * kMinNonZeroDistance range.
  matches = (distances.array() > distance_ratio * kMinNonZeroDistance)
                .select(0, matches);

  // Remove duplicate matches.
  std::vector<int> unique_match_indices;
  std::tie(std::ignore, unique_match_indices, std::ignore) =
      Unique(matches.cast<double>().reshaped());
  Eigen::MatrixXi unique_matches(1, matches.size());
  unique_matches.setZero();
  unique_matches.reshaped()(unique_match_indices) =
      matches.reshaped()(unique_match_indices);

  // Convert back to cv::Mat
  cv::eigen2cv(unique_matches, matches_);
}

//@brief Remove zeros in an array.
template <typename T>
T RemoveZeros(const T& A) {
  // std::vector<int> A_(A.cbegin(), A.cend());
  // std::vector<int>::const_iterator last_non_zero =
  //     std::remove_if(A_.begin(), A_.end(), [](int x) { return x != 0; });
  // std::vector<int> A_no_zeros(A_.cbegin(), last_non_zero);
  const auto num_non_zeros =
      std::count_if(A.cbegin(), A.cend(), [](int x) { return x != 0; });
  T A_no_zeros(num_non_zeros);
  std::copy_if(A.cbegin(), A.cend(), A_no_zeros.begin(),
               [](int x) { return x != 0; });
  return A_no_zeros;
}

//@brief Imitate matlab's `[row, col, v] = find(A)` function. Find non-zero
// elements in an array A.
//@param A One dimensional array.
//@return row An array containing the row subscripts of the non-zero elements in
// A.
//@return col An array containing the column subscripts of the non-zero elements
// in A.
//@return v One dimensional array containing the non-zero elements with order
// being consistent with the original order in A. I.e. this function is stable.
// TODO(bayes) Generalize this function to multi-dimensional array and make the
// parameter parsing more flexible by using template.
std::tuple<std::vector<int> /*row*/, std::vector<int> /*col*/,
           Eigen::ArrayXi /*v*/>
Find(const Eigen::ArrayXi& A) {
  // Assure all elements are greater than or equal to 0.
  //! This constraint can be simply removed later on. For now, it is set for
  //! safety.
  eigen_assert(!(A.unaryExpr([](int x) { return x < 0; }).any()));

  // Get row
  std::vector<int> row(A.count(), 1);

  // Get col
  std::vector<int> indices(A.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::transform(indices.begin(), indices.end(), indices.begin(),
                 [&A](int i) { return A(i) > 0 ? i : 0; });
  std::vector<int> col = RemoveZeros<std::vector<int>>(indices);
  if (A(0) != 0) col.insert(col.begin(), 0);

  // Get v
  Eigen::ArrayXi v(A.count());
  std::copy_if(A.cbegin(), A.cend(), v.begin(), [](int x) { return x > 0; });

  return {row, col, v};
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

  // Isolate query and match indices.
  //! These indices are used to access corresponding keypoints later on.
  std::vector<int> query_indices;
  Eigen::ArrayXi match_indices;
  std::tie(std::ignore, query_indices, match_indices) =
      Find(matches_.reshaped());

  // Extract coordinates of keypoints.
  Eigen::RowVectorXi from_kp_x, from_kp_y, to_kp_x, to_kp_y;
  from_kp_x = query_kps(0, query_indices);
  from_kp_y = query_kps(1, query_indices);
  to_kp_x = database_kps(0, match_indices);
  to_kp_y = database_kps(1, match_indices);

  // Link the each set of matches.
  const int kNumMatches = match_indices.size();
  for (int i = 0; i < kNumMatches; ++i) {
    int from_x, from_y, to_x, to_y;
    from_x = from_kp_x(i);
    from_y = from_kp_y(i);
    to_x = to_kp_x(i);
    to_y = to_kp_y(i);
    cv::line(image, {from_x, from_y}, {to_x, to_y}, {0, 255, 0}, 1);
  }
}

int main(int /*argv*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath = "data/ex3/";
  //! For debugging ...
  // Image to show the rendered objects.
  // cv::Mat image_show =
  //     cv::imread(kFilePath + "KITTI/000000.png", cv::IMREAD_COLOR);
  arma::mat X;
  X.load("img.csv");
  cv::Mat image_show(X.n_rows, X.n_cols, CV_64F, X.memptr());
  cv::Mat image_show_shi = image_show.clone();
  cv::Mat image = image_show.clone();
  // cv::cvtColor(image_show, image, cv::COLOR_BGR2GRAY, 1);

  // Part I: compute response.
  // The shi_tomasi_response is computed as comparison whilst the
  // harris_response is used through out the remainder of this program.
  const int kPatchSize = 9;
  const double kHarrisKappa = 0.08;
  cv::Mat harris_response, shi_tomasi_response;
  HarrisResponse(image, harris_response, kPatchSize, kHarrisKappa);
  ShiTomasiResponse(image, shi_tomasi_response, kPatchSize);
  // Compare the colormaps to see the detail of differences.
  ImageSC(harris_response);
  ImageSC(shi_tomasi_response);

  // Part II: select keypoints
  const int kNumKeypoints = 200;
  const int kNonMaximumRadius = 8;
  cv::Mat keypoints;
  SelectKeypoints(harris_response, keypoints, kNumKeypoints, kNonMaximumRadius);
  // Superimpose the selected keypoins to the original image.
  Eigen::MatrixXd k;
  cv::cv2eigen(keypoints, k);
  const Eigen::VectorXi x = k.row(0).cast<int>(), y = k.row(1).cast<int>();
  Scatter(image_show, x, y, 4, {0, 0, 255}, cv::FILLED);
  cv::imshow("Harris keypoints", image_show);
  cv::waitKey(0);

  // Show the Shi-Tomasi keypoints for comparison.
  cv::Mat shi_keypoints;
  SelectKeypoints(shi_tomasi_response, shi_keypoints, kNumKeypoints,
                  kNonMaximumRadius);
  Eigen::MatrixXd shi_k;
  cv::cv2eigen(shi_keypoints, shi_k);
  const Eigen::VectorXi shi_x = shi_k.row(0).cast<int>(),
                        shi_y = shi_k.row(1).cast<int>();
  Scatter(image_show_shi, shi_x, shi_y, 4, {0, 255, 0}, cv::FILLED);
  cv::imshow("Shi-Tomasi keypoints", image_show_shi);
  cv::waitKey(0);

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
  cv::Mat match_show =
      cv::imread(kFilePath + "KITTI/000001.png", cv::IMREAD_COLOR);
  cv::Mat query_image;
  cv::cvtColor(match_show, query_image, cv::COLOR_BGR2GRAY, 1);

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
  PlotMatches(matches, query_keypoints, keypoints, match_show);
  cv::imshow("", match_show);
  cv::waitKey(0);

  // Part V: match descriptors for all 200 images in the reduced KITTI
  // dataset.
  // Prepare database containers.
  cv::Mat database_kps, database_descs;
  const int kNumImages = 200;
  bool plot_matches = true;
  if (plot_matches) {
    for (int i = 0; i < kNumImages; ++i) {
      cv::Mat img_show =
          cv::imread(cv::format((kFilePath + "KITTI/%06d.png").c_str(), i),
                     cv::IMREAD_COLOR);
      cv::Mat query_img;
      cv::cvtColor(img_show, query_img, cv::COLOR_BGR2GRAY, 1);

      // Prepare query containers.
      cv::Mat query_harris, query_kps, query_descs;
      cv::Mat matches_qd;

      HarrisResponse(query_img, query_harris, kPatchSize, kHarrisKappa);
      SelectKeypoints(query_harris, query_kps, kNumKeypoints,
                      kNonMaximumRadius);
      DescribeKeypoints(query_img, query_kps, query_descs, kPatchRadius);

      // Match query and database after the first iteration.
      if (i >= 1) {
        MatchDescriptors(query_descs, database_descs, matches_qd,
                         kDistanceRatio);
        PlotMatches(matches_qd, query_kps, database_kps, img_show);
        cv::imshow("Matches", img_show);
        char key = cv::waitKey(10);  // Pause 10 ms.
        if (key == 27) break;        // 'ESC' key -> exit.
      }

      database_kps = query_kps;
      database_descs = query_descs;
    }
  }

  // // cv::Mat_<double> im{1, 3, 2, 3, 1, 5, 1, 3, 3, 4, 4, 2, 1,
  // //                     2, 2, 2, 1, 5, 5, 2, 3, 1, 5, 2, 3};
  // // im = im.reshape(1, 5);
  // arma::mat X;
  // X.load(kFilePath + "test_mat.txt", arma::file_type::auto_detect, true);
  // // std::cout << X << '\n';
  // // im.convertTo(im, CV_32F);
  // cv::Mat im(X.n_rows, X.n_cols, CV_64F, X.memptr());
  // std::cout << im << '\n';

  // cv::Mat Ix, Iy;
  // const cv::Mat sobel_hor = (cv::Mat_<double>(3, 1) << -1, 0, 1);
  // const cv::Mat sobel_ver = (cv::Mat_<double>(3, 1) << 1, 2, 1);
  // //@note OpenCV's cv::filter2D, cv::sepFilter2D and other filter functions
  // // actually do correlation rather than convolution. To do convolution, use
  // // cv::flip to flip the kernels along the anchor point (default the kernel
  // // center) in advance. N.B. For symmetric kernels, this step could be
  // skipped.
  // // The new anchor point can be computed as (kernel.cols - anchor.x - 1,
  // // kernel.rows - anchor.y - 1). For separable filters as well as the Sobel
  // // operators, the flipping operation can be accomplished with alternating
  // the
  // // sign of the sobel_hor or sobel_ver.
  // //@note cv::BORDER_ISOLATED
  // // When the source image is a part (ROI) of a bigger image, the function
  // will
  // // try to use the pixels outside of the ROI to form a border. To disable
  // this
  // // feature and always do extrapolation, as if src was not a ROI, use
  // // borderType | BORDER_ISOLATED
  // // TODO(bayes) Replace the cv's filter functions with self-implemented
  // Conv2D. cv::sepFilter2D(im, Ix, im.depth(), -sobel_hor.t(), sobel_ver, {-1,
  // -1}, 0.0,
  //                 cv::BORDER_ISOLATED);
  // cv::sepFilter2D(im, Iy, im.depth(), -sobel_ver.t(), sobel_hor, {-1, -1},
  // 0.0,
  //                 cv::BORDER_ISOLATED);

  // std::cout << "Ix , Iy\n\n";
  // std::cout << Ix << '\n';
  // std::cout << Iy << '\n';

  // // cv::Mat Ixp, Iyp;
  // // Gx = [[-1 0 +1],  Gy = [[-1 -2 -1],
  // //       [-2 0 +2],        [ 0  0  0],
  // //       [-1 0 +1]]        [+1 +2 +1]]
  // // cv::Mat_<int> sobel_x{-1, 0, 1, -2, 0, 2, -1, 0, 1},
  // //     sobel_y{-1, -2, -1, 0, 0, 0, 1, 2, 1};
  // // sobel_x = sobel_x.reshape(1, 3);
  // // sobel_y = sobel_y.reshape(1, 3);
  // // std::cout << sobel_x << '\n';
  // // std::cout << sobel_y << '\n';
  // cv::Mat Ixx, Iyy, Ixy;
  // Ixx = Ix.mul(Ix);
  // Iyy = Iy.mul(Iy);
  // Ixy = Ix.mul(Iy);

  // std::cout << "Ixx, Iyy, Ixy" << '\n' << '\n';
  // std::cout << Ixx << '\n';
  // std::cout << Iyy << '\n';
  // std::cout << Ixy << '\n';

  // int patch_size = 9;
  // int patch_radius = std::floor(patch_size / 2);
  // const cv::Mat patch = cv::Mat::ones(patch_size, patch_size, image.depth());
  // cv::Mat ssd_Ixx, ssd_Iyy, ssd_Ixy;
  // cv::filter2D(Ixx, ssd_Ixx, Ixx.depth(), patch, {-1, -1}, 0.0,
  //              cv::BORDER_ISOLATED);
  // cv::filter2D(Iyy, ssd_Iyy, Iyy.depth(), patch, {-1, -1}, 0.0,
  //              cv::BORDER_ISOLATED);
  // cv::filter2D(Ixy, ssd_Ixy, Ixy.depth(), patch, {-1, -1}, 0.0,
  //              cv::BORDER_ISOLATED);

  // std::cout << "ssd Ixx, Iyy, Ixy"
  //           << "\n\n";
  // std::cout << ssd_Ixx << '\n';
  // std::cout << ssd_Iyy << '\n';
  // std::cout << ssd_Ixy << '\n';

  // Eigen::MatrixXd s_Ixx, s_Iyy, s_Ixy;
  // cv::cv2eigen(ssd_Ixx, s_Ixx);
  // cv::cv2eigen(ssd_Iyy, s_Iyy);
  // cv::cv2eigen(ssd_Ixy, s_Ixy);

  // std::cout << "eigen ssd Ixx, Iyy, Ixy"
  //           << "\n\n";
  // std::cout << s_Ixx << '\n' << '\n';
  // std::cout << s_Iyy << '\n' << '\n';
  // std::cout << s_Ixy << '\n' << '\n';

  // // Compute trace and determinant.
  // // The structure tensor M = [a, b; c, d] and the trace is computed as trace
  // =
  // // a + d while the determinant = a*d - b*c.
  // Eigen::MatrixXd trace, determinant;
  // trace = s_Ixx.array() + s_Iyy.array();
  // determinant = s_Ixx.cwiseProduct(s_Iyy) - s_Ixy.cwiseProduct(s_Ixy);

  // std::cout << trace << "\n\n";
  // std::cout << determinant << "\n\n";

  // double kappa = 0.08;
  // Eigen::MatrixXd response;
  // response = determinant - kappa * trace.cwiseProduct(trace);
  // // Simply set all responses smaller than 0 to 0.
  // response = response.unaryExpr([](double x) { return x < 0.0 ? 0.0 : x; });

  // std::cout << "response\n\n";
  // std::cout << response << '\n' << '\n';

  // // Keep only the parts which do not include zero-padded edges.
  // // For the "valid" optional, `C = conv2(A, B, "valid")` returns C with size
  // as
  // // max(size(A) - size(B) + 1, 0).
  // // Because we've convolved twice, so the desired size(C) = size(A) -
  // size(B1)
  // // - size(B2) + 2, where size is the length one dimension, #rows or #cols.
  // int valid_rows = im.rows - sobel_hor.rows - patch.rows + 2;
  // int valid_cols = im.cols - sobel_ver.rows - patch.cols + 2;
  // valid_rows = std::max(valid_rows, 0);
  // valid_cols = std::max(valid_cols, 0);
  // if (valid_rows == 0 || valid_cols == 0) {
  //   LOG(ERROR) << "Invalid ROI";
  // }
  // std::cout << "valid size: " << valid_rows << " " << valid_cols << '\n';

  // // Compute the starting point of the valid block.
  // // starting_point = radius(B1) + radius(B2).
  // const int sobel_radius = static_cast<int>(std::floor(sobel_hor.rows / 2));

  // // Assume the kernels are square, then starting_x = starting_y.
  // const int starting_x = sobel_radius + patch_radius, starting_y =
  // starting_x; Eigen::MatrixXd response_valid =
  //     response.block(starting_x, starting_y, valid_rows, valid_cols);

  // // Convert back to cv::Mat and store it to the output harris_response.
  // cv::Mat harriss;
  // cv::eigen2cv(response_valid, harriss);
  // // Pad the harris_response making its size consistent with the input image.
  // // And set the pixels on borders to 0. When the dst.size > src.size whereby
  // // diff < 0, the function truncates the src image, which is exactly what we
  // // need.
  // // const int diff = im.rows - harriss.rows;
  // // cv::copyMakeBorder(harriss, harriss, diff, diff, diff, diff,
  // //                    cv::BORDER_CONSTANT, {0, 0, 0, 0});

  // const int padding_size = sobel_radius + patch_radius;
  // std::cout << "padding_size: " << padding_size << '\n';
  // PadArray(harriss, {padding_size, padding_size, padding_size,
  // padding_size});

  // std::cout << "harris\n\n";
  // std::cout << harriss << '\n';
  // -------------------------------------------------------------------
  // Optional: profile the program

  return EXIT_SUCCESS;
}
