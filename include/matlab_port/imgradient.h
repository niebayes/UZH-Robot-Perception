#ifndef UZH_MATLAB_PORT_IMGRADIENT_H_
#define UZH_MATLAB_PORT_IMGRADIENT_H_

#include "armadillo"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "transfer.h"

namespace uzh {

enum GradientMethod : int { SOBEL };

arma::cube imgradient(const arma::mat& image, const int method = uzh::SOBEL) {
  if (image.empty()) LOG(ERROR) << "Empty input image.";

  arma::cube mag_dir(image.n_rows, image.n_cols, 2);

  cv::Mat img = uzh::arma2cv<double>(image);
  cv::Mat_<double> sobel_x, sobel_y, sobel_mag;
  cv::Sobel(img, sobel_x, CV_64F, 1, 0, 3, 1.0, 0.0, cv::BORDER_ISOLATED);
  cv::Sobel(img, sobel_y, CV_64F, 0, 1, 3, 1.0, 0.0, cv::BORDER_ISOLATED);
  cv::pow(sobel_x.mul(sobel_x) + sobel_y.mul(sobel_y), 0.5, sobel_mag);
  cv::Mat_<double> mag;
  cv::magnitude(sobel_x, sobel_y, mag);
  cv::Mat count;
  cv::bitwise_and(sobel_mag, mag, count);
  std::cout << (cv::countNonZero(count) == (mag.rows * mag.cols)) << '\n';

  const arma::mat Gmag = uzh::cv2arma<double>(sobel_mag).t();
  arma::mat Gx, Gy, Gdir;
  Gx = uzh::cv2arma<double>(sobel_x).t();
  Gy = uzh::cv2arma<double>(sobel_y).t();
  Gdir = arma::atan2(Gx, Gy);

  mag_dir.slice(0) = Gmag;
  mag_dir.slice(1) = Gdir;
  return mag_dir;
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_IMGRADIENT_H_