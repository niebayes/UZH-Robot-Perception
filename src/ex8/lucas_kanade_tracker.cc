#include <string>
#include <tuple>
#include <vector>

#include "armadillo"
#include "feature/matching.h"
#include "google_suite.h"
#include "io.h"
#include "klt.h"
#include "matlab_port.h"
#include "opencv2/opencv.hpp"
#include "transfer.h"

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath{"data/ex8/"};

  // Part I: warp an image to test the functionalities of the warping functions.
  // Reference image.
  const cv::Mat img_r =
      cv::imread(kFilePath + "000000.png", cv::IMREAD_GRAYSCALE);
  const arma::umat I_r = uzh::img2arma(img_r);

  // Various basic warpings.
  //! Note, the warpings denote inverse warping.
  const arma::mat W_t_inv = uzh::GetSimpleWarp(50.0, -30.0, 0.0, 1.0);
  const arma::mat W_R_inv = uzh::GetSimpleWarp(0.0, 0.0, 10.0, 1.0);
  const arma::mat W_s_inv = uzh::GetSimpleWarp(0.0, 0.0, 0.0, 0.5);
  // Show the warped images.
  std::vector<cv::Mat> imgs_test(4);
  imgs_test[0] = img_r;
  imgs_test[1] = uzh::arma2img(uzh::WarpImage(I_r, W_t_inv));
  imgs_test[2] = uzh::arma2img(uzh::WarpImage(I_r, W_R_inv));
  imgs_test[3] = uzh::arma2img(uzh::WarpImage(I_r, W_s_inv));
  // cv::imshow("Warped images", uzh::MakeCanvas(imgs_test, 1.5 * I_r.n_rows,
  // 2)); cv::waitKey(0);

  // Part II: track a patch warped with pure-translational warping in a
  // brute-force manner.
  // Obtain the template patch.
  const arma::mat W_0_inv = uzh::GetSimpleWarp(0.0, 0.0, 0.0, 1.0);
  const arma::vec2 x_T{899, 290};
  const int r_T = 15;
  const arma::umat template_patch = uzh::GetWarpedPatch(I_r, W_0_inv, x_T, r_T);
  // Track the template in the warped image.
  // Apply translation-only warping.
  //! W_inv denotes inverse warping wrt. reference image I_r, such that
  //! I_r = W_inv * I_w.
  const arma::mat W_inv = uzh::GetSimpleWarp(10.0, 6.0, 0.0, 1.0);
  const arma::umat I_w = uzh::WarpImage(I_r, W_inv);
  const int r_D = 20;
  arma::mat W_t;
  arma::mat ssds;
  std::tie(W_t, ssds) = uzh::TrackBruteForce(I_r, I_w, x_T, r_T, r_D);
  std::vector<cv::Mat> imgs_track(4);
  imgs_track[0] = img_r;
  imgs_track[1] = uzh::imagesc(template_patch, false);
  imgs_track[2] = uzh::arma2img(I_w);
  //* The line segment linking the upper left corner and the darkest point in
  // the SSD colormap denotes the displacement vector.
  imgs_track[3] = uzh::imagesc(arma::conv_to<arma::umat>::from(ssds), false);
  std::cout
      << "Brute-forcely recovered translation-only forward warping is: \n";
  W_t.print();
  // cv::imshow("Track template with brute-force",
  //            uzh::MakeCanvas(imgs_track, I_r.n_rows, 2));
  // cv::waitKey(0);

  // Part III: recover the warp with KLT.
  const int kNumIterations = 50;
  arma::mat W, param_history;
  std::tie(W, param_history) =
      uzh::TrackKLT(I_r, I_w, x_T, r_T, kNumIterations, true);
  std::cout << "KLT recovered translation-only forward warp is:\n";
  W.print();

  // Apply KLT on KITTI dataset.
  // Downsample by a factor of 4 to speed up tracking.
  const cv::Mat img_r_down = uzh::imresize(img_r, 0.25);
  const arma::umat I_r_down = uzh::img2arma(img_r_down);
  arma::mat keypoints = arma::conv_to<arma::mat>::from(
      uzh::LoadArma<arma::uword>(kFilePath + "keypoints.txt").t());
  // Make indices start from 0.
  keypoints -= 1.0;
  // Change (row, col) layout to (x, y) layout.
  keypoints = arma::flipud(keypoints);
  keypoints /= 4.0;  // Keypoints are scaled accordingly.
  // Only track part of keypoints.
  const int kNumKeypoints = 50;
  keypoints = keypoints.head_cols(kNumKeypoints);

  // Track the keypoints.
  arma::umat I_prev = I_r_down;
  arma::mat kpts_prev = keypoints;
  const int kNumImages = 20;
  for (int i = 1; i < kNumImages + 1; ++i) {
    const cv::Mat img_i = cv::imread(
        cv::format((kFilePath + "%06d.png").c_str(), i), cv::IMREAD_GRAYSCALE);
    const cv::Mat img_i_down = uzh::imresize(img_i, 0.25);
    const arma::umat I = uzh::img2arma(img_i_down);

    // Obtain delta translations of keypoints.
    arma::mat delta_kpts(arma::size(kpts_prev), arma::fill::zeros);
    for (int j = 0; j < kNumKeypoints; ++j) {
      arma::mat W_i;
      std::tie(W_i, std::ignore) =
          uzh::TrackKLT(I_prev, I, kpts_prev.col(j), r_T, kNumIterations);
      delta_kpts.col(j) = W_i.tail_cols(1);
    }

    // New coordinates of the tracked points.
    const arma::mat kpts = kpts_prev + delta_kpts;

    // Plot the matches.
    // Matches are one-to-one.
    const cv::Mat matches = uzh::arma2img(
        arma::linspace<arma::urowvec>(0, kNumKeypoints - 1, kNumKeypoints));
    // Display matches in original resolution.
    cv::Mat match_show;
    cv::cvtColor(img_i_down, match_show, 3);
    uzh::PlotMatches(
        matches,
        uzh::arma2img(arma::flipud(arma::conv_to<arma::umat>::from(kpts))),
        uzh::arma2img(arma::flipud(arma::conv_to<arma::umat>::from(kpts_prev))),
        match_show, true);
    cv::namedWindow("KLT tracking", cv::WINDOW_NORMAL);
    cv::imshow("KLT tracking", match_show);
    const char key = cv::waitKey(50);
    if (key == 32) cv::waitKey(0);  // 'Space' key -> pause.

    I_prev = I;
    kpts_prev = kpts;
  }

  // Part V: robustly do KLT tracking with thresholding on bidirectional error.
  I_prev = I_r_down;
  kpts_prev = keypoints;
  for (int i = 1; i < kNumImages + 1; ++i) {
    const cv::Mat img_i = cv::imread(
        cv::format((kFilePath + "%06d.png").c_str(), i), cv::IMREAD_GRAYSCALE);
    const cv::Mat img_i_down = uzh::imresize(img_i, 0.25);
    const arma::umat I = uzh::img2arma(img_i_down);

    // Obtain delta translations of keypoints.
    arma::mat delta_kpts(arma::size(kpts_prev), arma::fill::zeros);
    arma::urowvec is_kept(kpts_prev.n_cols, arma::fill::ones);
    const double kBidirectionalErrorThreshold = 0.1;
    for (int j = 0; j < kpts_prev.n_cols; ++j) {
      arma::vec2 delta_t;
      std::tie(delta_t, is_kept(j)) =
          uzh::TrackKLTRobust(I_prev, I, kpts_prev.col(j), r_T, kNumIterations,
                              kBidirectionalErrorThreshold);
      delta_kpts.col(j) = delta_t;
    }

    // New coordinates of the tracked points.
    arma::mat kpts = kpts_prev + delta_kpts;

    // Only keep points that have passed the bidirectional test.
    //! Number of tracked points is incrementally decreased.
    if (uzh::nnz<arma::uword>(is_kept) > 0) {
      kpts = kpts.cols(arma::find(is_kept));
      kpts_prev = kpts_prev.cols(arma::find(is_kept));

      // Plot the matches.
      // Matches are one-to-one.
      const int num_kept_points = kpts.n_cols;
      const cv::Mat matches = uzh::arma2img(arma::linspace<arma::urowvec>(
          0, num_kept_points - 1, num_kept_points));
      cv::Mat match_show;
      cv::cvtColor(img_i_down, match_show, 3);
      uzh::PlotMatches(
          matches,
          uzh::arma2img(arma::flipud(arma::conv_to<arma::umat>::from(kpts))),
          uzh::arma2img(
              arma::flipud(arma::conv_to<arma::umat>::from(kpts_prev))),
          match_show, true);
      cv::namedWindow("Robust KLT tracking", cv::WINDOW_NORMAL);
      cv::imshow("Robust KLT tracking", match_show);
      const char key = cv::waitKey(50);
      if (key == 32) cv::waitKey(0);  // 'Space' key -> pause.
    }

    I_prev = I;
    kpts_prev = kpts;
  }

  return EXIT_SUCCESS;
}