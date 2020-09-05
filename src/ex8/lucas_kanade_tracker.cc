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
  const cv::Mat img_0 =
      cv::imread(kFilePath + "000000.png", cv::IMREAD_GRAYSCALE);
  std::cout << img_0.rowRange(0, 10).colRange(0, 10) << '\n';
  const arma::umat I_r = uzh::img2arma(img_0);
  I_r(0, 0, arma::size(10, 10)).print("I_r");
  cv::Mat t = uzh::arma2img(I_r);
  std::cout << img_0.rowRange(0, 10).colRange(0, 10) << '\n';
  cv::imshow("", t);
  cv::waitKey(0);

  // Various basic warpings.
  const arma::mat W_t = uzh::GetSimpleWarp(50, -30, 0, 1);
  const arma::mat W_R = uzh::GetSimpleWarp(0, 0, 10, 1);
  const arma::mat W_s = uzh::GetSimpleWarp(0, 0, 0, 0.5);
  // Show the warped images.
  std::vector<cv::Mat> imgs_test(4);
  imgs_test[0] = img_0;
  imgs_test[1] = uzh::arma2img(uzh::WarpImage(I_r, W_t));
  imgs_test[2] = uzh::arma2img(uzh::WarpImage(I_r, W_R));
  imgs_test[3] = uzh::arma2img(uzh::WarpImage(I_r, W_s));
  cv::imshow("Warped images", uzh::MakeCanvas(imgs_test, I_r.n_rows, 2));
  cv::waitKey(0);

  return EXIT_SUCCESS;
}