#ifndef UZH_MATLAB_PORT_IMGRADIENT_H_
#define UZH_MATLAB_PORT_IMGRADIENT_H_

#include "armadillo"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "transfer.h"

namespace uzh {

enum GradientMethod : int { SOBEL };

arma::field<arma::mat> imgradient(const arma::mat& image,
                                  const int method = uzh::SOBEL) {
  if (image.empty()) LOG(ERROR) << "Empty input image.";

  arma::field<arma::mat> mag_dir(2);

  std::cout << "image" << '\n';
  std::cout << image << '\n';

  // Compute the first derivatives and the magnitude.
  // Use padarray to escape from the weird behavior.
  cv::Mat img = uzh::arma2cv<double>(image);
  cv::Mat_<double> sobel_x, sobel_y, sobel_mag;
  cv::Sobel(img, sobel_x, CV_64F, 1, 0, 3, 1.0, 0.0, cv::BORDER_ISOLATED);
  cv::Sobel(img, sobel_y, CV_64F, 0, 1, 3, 1.0, 0.0, cv::BORDER_ISOLATED);
  std::cout << "sobel_x\n";
  std::cout << sobel_x << '\n';
  std::cout << "sobel_y\n";
  std::cout << sobel_y << '\n';
  cv::magnitude(sobel_x, sobel_y, sobel_mag);

  // cv test
  cv::Mat_<double> Ix, Iy;
  const cv::Mat_<double> sobel_hor = (cv::Mat_<double>(3, 1) << -1, 0, 1);
  const cv::Mat_<double> sobel_ver = (cv::Mat_<double>(3, 1) << 1, 2, 1);
  cv::sepFilter2D(img, Ix, CV_64F, -sobel_hor.t(), sobel_ver, {-1, -1}, 0.0,
                  cv::BORDER_ISOLATED);
  cv::sepFilter2D(img, Iy, CV_64F, -sobel_ver.t(), sobel_hor, {-1, -1}, 0.0,
                  cv::BORDER_ISOLATED);
  std::cout << "Ix\n";
  std::cout << Ix << '\n';
  std::cout << "Iy\n";
  std::cout << Iy << '\n';

  // arma test
  // std::cout << "arma conv2\n";
  // const arma::mat sy{{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}}, sx = sy.t();
  // const arma::mat gx = arma::conv2(image, sx, "same");
  // const arma::mat gy = arma::conv2(image, sy, "same");
  // std::cout << gx << '\n';
  // std::cout << gy << '\n';

  // Convert back to arma::mat
  arma::mat Gx, Gy, Gmag, Gdir;
  Gx = uzh::cv2arma<double>(sobel_x).t();
  Gy = uzh::cv2arma<double>(sobel_y).t();
  Gmag = uzh::cv2arma<double>(sobel_mag).t();
  Gdir = arma::atan2(Gx, Gy);

  mag_dir(0) = Gmag;
  mag_dir(1) = Gdir;
  arma::arma_assert_same_size(Gmag, Gdir, "size(Gmag) != size(Gdir)");
  return mag_dir;
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_IMGRADIENT_H_