#ifndef UZH_RANSAC_RANSAC_LOCALIZATION_H_
#define UZH_RANSAC_RANSAC_LOCALIZATION_H_

#include <cmath>
#include <tuple>
#include <vector>

#include "Eigen/Core"
#include "armadillo"
#include "glog/logging.h"
#include "ransac/kneip_p3p.h"
#include "vision.h"

namespace uzh {

enum MethodToFindCameraPose : int { DLT, P3P };

//@brief Localize camera given 3D-2D point correspondences using DLT or P3P.
//@param keypoints [2 x n] matrix where each column contains a matched image
// point expressed in pixels, such that p = (row, col).
//@param landmarks [3 x n] matrix where each column contains the corresponding
// landmark wrt. the matched keypoint, such that P = (X, Y, Z).
//@param method Method to find the camera pose, DLT or P3P. By default, P3P is
// applied.
//@param adaptive_iterations If true, the number of iterations of RANSAC is not
// fixed but incrementally updated from the estimated ratio of outliers at each
// iteration. By default, it's set to true.
//@param tweak_for_more_robust If true, the number of iterations and the minimum
// number of inliers, exceed which the inlier mask at each iteration is then
// possible to be selected as the best inlier mask, are set bigger. By default,
// it's set to true.
//@returns
//- R_W_C -- [3 x 3] rotation matrix.
//- t_W_C -- [3 x 1] translation vector. The rigid transformation formed with
// R_W_C and t_W_C mapping 3D scene points from camera frame to world frame.
//- best_inlier_mask -- [1 x n] row vector where each entry that indicates the
// corresponding match is true if 1 or false if 0.
//- max_num_inliers_history -- [1 x k] row vector recording the maximum number
// of inliers passed the RANSAC test at each iteration so far. k is number of
// iterations the function actually processed.
//- num_iterations_history -- [1 x k] row vector records the adapted number of
// iterations at each iteration i = 1, 2, ..., k, where k is number of
// iterations the function actually processed.
std::tuple<arma::mat33 /* R_W_C */, arma::vec3 /* t_W_C */,
           arma::urowvec /* best_inlier_mask */,
           arma::urowvec /* max_num_inliers_history */,
           arma::rowvec /* num_iterations_history */>
RANSACLocalization(const arma::umat& keypoints, const arma::mat& landmarks,
                   const arma::mat& K, const int method = uzh::P3P,
                   const int adaptive_iterations = true,
                   const bool tweak_for_more_robust = true) {
  if (keypoints.empty() || landmarks.empty() || K.empty())
    LOG(ERROR) << "Empty input.";
  if (keypoints.n_rows != 2 || landmarks.n_rows != 3 ||
      arma::size(K) != arma::size(3, 3))
    LOG(ERROR) << "Invalid input.";
  if (keypoints.n_cols != landmarks.n_cols) {
    LOG(ERROR)
        << "Number of keypoints must be the same with that of landmarks.";
  }
  if (method != uzh::P3P || method != uzh::DLT)
    LOG(ERROR) << "Unsupported method.";

  // Construct objects to be computed.
  arma::mat33 R_W_C;
  arma::vec3 t_W_C;
  const int kNumCorrespondences = keypoints.n_cols;
  arma::urowvec best_inlier_mask(kNumCorrespondences, arma::fill::zeros);
  std::vector<int> max_num_inliers_history;
  // Double type to account for the adaptive number of iterations computed from
  // the formula involving log and division.
  std::vector<double> num_iterations_history;

  // Determine settings according to parameters.
  int num_iterations;
  int s;  // Minimum number of points with which the corresponding method is
          // able to be performed.
  if (method == uzh::P3P) {
    s = 3;
    if (tweak_for_more_robust)
      num_iterations = 1000;
    else
      num_iterations = 200;
  } else {
    s = 6;
    num_iterations = 200;
  }
  if (adaptive_iterations) {
    // Number of iterations is not fixed if using adaptive iterations.
    num_iterations = arma::datum::inf;
  }
  // Error bound within which a match is selected as inlier.
  const double kPixelTolerance = 10.0;

  int k = 0;                // Iteration counter.
  int max_num_inliers = 0;  // Maximum number of inliers found so far.
  while (k < num_iterations) {
    arma::mat landmark_samples;
    arma::uvec sample_indices;
    std::tie(landmark_samples, sample_indices) =
        uzh::datasample<double>(landmarks, s, 1, false);
    const arma::umat corresponding_kpts = keypoints.cols(sample_indices);

    // Compute camera pose using P3P or DLT.
    arma::cube R_W_C_guess(3, 3, 4, arma::fill::eye);
    arma::cube t_W_C_guess(3, 1, 4, arma::fill::zeros);
    if (method == uzh::P3P) {
      const arma::mat bearing_vectors =
          K.i() * arma::conv_to<arma::mat>::from(corresponding_kpts);
      // P3P require three unitary bearing vectors.
      const arma::mat unitary_bearing_vectors =
          arma::normalise(bearing_vectors);
      Eigen::Matrix<Eigen::Matrix<double, 3, 4>, 4, 1> poses;
      uzh::P3P::computePoses(uzh::arma2eigen(unitary_bearing_vectors),
                             uzh::arma2eigen(landmark_samples), poses);
      // Decode the result of P3P.
      for (int i = 0; i < poses.rows(); ++i) {
        const Eigen::Matrix<double, 3, 4> pose_tmp = poses(0);
        const arma::mat pose = uzh::eigen2arma(pose_tmp);
        R_W_C_guess.slice(i) = pose.head_cols(3);
        t_W_C_guess.slice(i) = pose.tail_cols(1);
      }
    } else {
      R_W_C_guess.set_size(3, 3, 1);
      t_W_C_guess.set_size(3, 1, 1);
      uzh::CameraMatrixDLT M_DLT = uzh::EstimatePoseDLT(
          uzh::arma2eigen(arma::conv_to<arma::mat>::from(corresponding_kpts)),
          uzh::arma2eigen(landmark_samples), uzh::arma2eigen(K));
      M_DLT.DecomposeDLT();
      const arma::mat M_W_C = uzh::eigen2arma(M_DLT.getM());
      R_W_C_guess.slice(0) = M_W_C.head_cols(3);
      t_W_C_guess.slice(0) = M_W_C.tail_cols(1);
    }

    // Compute reprojection error to identify inliers.
    // Transform landmarks to camera frame and then project.
    arma::mat landmarks_C =
        R_W_C_guess.slice(0) * landmarks +
        arma::repmat(t_W_C_guess.slice(0), 1, kNumCorrespondences);
    Eigen::Matrix2Xd reprojected_points_tmp;
    uzh::ProjectPoints(uzh::arma2eigen(landmarks_C), &reprojected_points_tmp,
                       uzh::arma2eigen(K));
    arma::mat reprojected_points = uzh::eigen2arma(reprojected_points_tmp);
    arma::rowvec residuals =
        arma::sum(arma::square(reprojected_points - keypoints));
    arma::urowvec inlier_mask =
        (residuals < (kPixelTolerance * kPixelTolerance));

    // Process another 3 three pose guesses if using P3P.
    if (method == uzh::P3P) {
      for (int i = 1; i < 4; ++i) {
        landmarks_C =
            R_W_C_guess.slice(i) * landmarks +
            arma::repmat(t_W_C_guess.slice(i), 1, kNumCorrespondences);
        uzh::ProjectPoints(uzh::arma2eigen(landmarks_C),
                           &reprojected_points_tmp, uzh::arma2eigen(K));
        reprojected_points = uzh::eigen2arma(reprojected_points_tmp);
        residuals = arma::sum(arma::square(reprojected_points - keypoints));
        arma::urowvec alternative_inlier_mask =
            (residuals < (kPixelTolerance * kPixelTolerance));

        // Get the best inlier mask among all possible poses obtained from P3P.
        if (arma::size(arma::nonzeros(alternative_inlier_mask)).n_rows >
            arma::size(arma::nonzeros(inlier_mask)).n_rows) {
          inlier_mask = alternative_inlier_mask;
        }
      }
    }

    // Update best inlier mask if more inliers are found.

    // If tweak_for_more_robust is true, lift the threshold of the minimum
    // number of inliers exceed which the best inlier mask is then updated.
    const int min_num_inliers_threshold = tweak_for_more_robust ? 30 : 6;

    const int num_inliers = arma::size(arma::nonzeros(inlier_mask)).n_rows;
    if (num_inliers > min_num_inliers_threshold &&
        num_inliers >= max_num_inliers) {
      max_num_inliers = num_inliers;
      best_inlier_mask = inlier_mask;
    }

    // Adaptively change number of iterations
    if (adaptive_iterations) {
      // Upper bound of outlier ratio is set to 0.90
      const double outlier_ratio = std::min(
          1 - double(max_num_inliers) / double(kNumCorrespondences), 0.90);
      // Confidence about how much correspondences are inliers.
      const double confidence = 0.95;
      num_iterations = std::log(1 - confidence) /
                       std::log(1 - std::pow((1 - outlier_ratio), s));
      // Set the upper bound of number of iterations.
      num_iterations = std::min(num_iterations, 2000);
    }

    // Record the maximum number of inliers and number of iterations.
    max_num_inliers_history.push_back(max_num_inliers);
    num_iterations_history.push_back(num_iterations);

    ++k;
  }

  // Refine the pose using DLT with more correspondences as P3P always use 3.
  arma::mat33 R_W_C_final;
  arma::vec3 t_W_C_final;
  uzh::CameraMatrixDLT M_DLT_final = uzh::EstimatePoseDLT(
      uzh::arma2eigen(arma::conv_to<arma::mat>::from(
          keypoints.cols(arma::find(best_inlier_mask)))),
      uzh::arma2eigen(landmarks.cols(arma::find(best_inlier_mask))),
      uzh::arma2eigen(K));
  M_DLT_final.DecomposeDLT();
  const arma::mat M_W_C_final = uzh::eigen2arma(M_DLT_final.getM());
  R_W_C_final = M_W_C_final.head_cols(3);
  t_W_C_final = M_W_C_final.tail_cols(1);

  if (max_num_inliers > 0) {
    return {R_W_C_final, t_W_C_final, best_inlier_mask,
            arma::conv_to<arma::urowvec>::from(max_num_inliers_history),
            arma::conv_to<arma::rowvec>::from(num_iterations_history)};
  } else {
    LOG(INFO) << "No inlier found.";
    return {R_W_C_final, t_W_C_final, best_inlier_mask,
            arma::conv_to<arma::urowvec>::from(max_num_inliers_history),
            arma::conv_to<arma::rowvec>::from(num_iterations_history)};
  }
}

}  // namespace uzh

#endif  // UZH_RANSAC_RANSAC_LOCALIZATION_H_