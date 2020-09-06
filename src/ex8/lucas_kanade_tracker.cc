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
      cv::imread(kFilePath + "000000.png", cv::IMREAD_ANYDEPTH);
  const arma::umat I_r = uzh::img2arma(img_r);

  // Various basic warpings.
  //! Note, the warpings denotes inverse warping.
  const arma::mat W_t_inv = uzh::GetSimpleWarp(50, -30, 0, 1);
  const arma::mat W_R_inv = uzh::GetSimpleWarp(0, 0, 10, 1);
  const arma::mat W_s_inv = uzh::GetSimpleWarp(0, 0, 0, 0.5);
  // Show the warped images.
  std::vector<cv::Mat> imgs_test(4);
  imgs_test[0] = img_r;
  imgs_test[1] = uzh::arma2img(uzh::WarpImage(I_r, W_t_inv));
  imgs_test[2] = uzh::arma2img(uzh::WarpImage(I_r, W_R_inv));
  imgs_test[3] = uzh::arma2img(uzh::WarpImage(I_r, W_s_inv));
  // cv::imshow("Warped images", uzh::MakeCanvas(imgs_test, 1.5 * I_r.n_rows,
  // 2)); cv::waitKey(0);

  // Part II: track an patch with pure-translational warping in a brute-force
  // manner.
  // Obtain template patch.
  const arma::mat W_0_inv = uzh::GetSimpleWarp(0, 0, 0, 1);
  const arma::uvec2 x_T{899, 290};
  const double r_T = 15;
  const arma::umat template_patch = uzh::GetWarpedPatch(I_r, W_0_inv, x_T, r_T);
  // Track the template in the warped image.
  // Apply translation-only warping.
  //! W_inv denotes inverse warping wrt. reference image I_r, such that
  //! I_r = W_inv * I_w.
  const arma::mat W_inv = uzh::GetSimpleWarp(10, 6, 0, 1);
  const arma::umat I_w = uzh::WarpImage(I_r, W_inv);
  const double r_D = 20;
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
  std::cout << "The recovered translation-only forward warping is: \n";
  W_t.print();
  // cv::imshow("Track template with brute-force",
  //            uzh::MakeCanvas(imgs_track, I_r.n_rows, 2));
  // cv::waitKey(0);

  // Part III: recover the warp with KLT.
  const int kNumIterations = 50;
  arma::mat W, param_history;
  std::tie(W, param_history) =
      uzh::TrackKLT(I_r, I_w, x_T, r_T, kNumIterations);

  return EXIT_SUCCESS;
}