#ifndef UZH_MATLAB_PORT_IMGRADIENT_H_
#define UZH_MATLAB_PORT_IMGRADIENT_H_

#include "armadillo"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "transfer/arma2cv.h"
#include "transfer/cv2arma.h"

namespace uzh {

enum GradientMethod : int { SOBEL };

arma::field<arma::mat> imgradient(const arma::mat& image,
                                  const int method = uzh::SOBEL) {
  if (image.empty()) LOG(ERROR) << "Empty input image.";

  arma::field<arma::mat> mag_dir(2);

  // std::cout << "image" << '\n';
  // std::cout << image << '\n';

  // Compute the first derivatives and the magnitude.
  // Use padarray to escape from the weird behavior.
  cv::Mat img = uzh::arma2cv<double>(image);
  cv::Mat_<double> sobel_x, sobel_y, sobel_mag;
  //! To compute image gradient, pad the borders with replicated values of the
  //! pixels on boundary such that the gradients are computed correctly.
  cv::Sobel(img, sobel_x, CV_64F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
  cv::Sobel(img, sobel_y, CV_64F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
  cv::magnitude(sobel_x, sobel_y, sobel_mag);

  // Convert back to arma::mat
  arma::mat Gx, Gy, Gmag, Gdir;
  Gx = uzh::cv2arma<double>(sobel_x).t();
  Gy = uzh::cv2arma<double>(sobel_y).t();
  Gmag = uzh::cv2arma<double>(sobel_mag).t();
  Gdir = arma::atan2(Gx, Gy) * 180 / arma::datum::pi;

  mag_dir(0) = Gmag;
  mag_dir(1) = Gdir;
  arma::arma_assert_same_size(Gmag, Gdir, "size(Gmag) != size(Gdir)");
  return mag_dir;
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_IMGRADIENT_H_