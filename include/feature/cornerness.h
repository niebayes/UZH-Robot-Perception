#ifndef UZH_CORNERNESS_H_
#define UZH_CORNERNESS_H_
#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "google_suite.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

//@brief Compute the Harris cornerness, i.e. Harris response for the image given
// patch_size and kappa.
//@param image Source image
//@param harris_response Output response matrix
//@param patch_size Size of the image patch, that is the size of the weight
// window used to compute the auto-correlation function, aka. SSD surface. N.B
// patch_size = patch_radius + 1, where patch_radius is the distance between the
// pixels on the border of the image patch and the center of the image, aka. the
// anchor point. Note in OpenCV, the patch_radius is treated the same as
// aperture size.
//@param kappa Parameter used in Harris response function. The default 0.06 is a
// empirically well-investigated value.
//@note InputArray in Opencv
//@ref https://stackoverflow.com/a/31820323/14007680
void HarrisResponse(const cv::Mat& image, cv::Mat& harris_response,
                    const int patch_size, const int kappa = 0.06) {
  // Compute the horizontal and vertical derivatives of the image Ix and Iy by
  // convolving the original image with derivatives of Gaussians.
  // Here, the Sobel operator is applied and is separated into two 1D filters
  // sobel_hor and sobel_ver.
  // Sobel operators:
  // Gx = [[-1 0 +1],  Gy = [[-1 -2 -1],
  //       [-2 0 +2],        [ 0  0  0],
  //       [-1 0 +1]]        [+1 +2 +1]]
  // Separated 1D components:
  // sobel_hor = [+1 0 -1], sobel_ver = [+1 +2 +1]
  // Hence, Gx = sobel_ver' * sobel_hor, Gy = sobel_hor' * sobel_ver.
  cv::Mat Ix, Iy;
  const cv::Mat sobel_hor = (cv::Mat_<double>() << 1, 0, -1);
  const cv::Mat sobel_ver = (cv::Mat_<double>() << 1, 2, 1);
  cv::sepFilter2D(image, Ix, image.depth(), sobel_hor, sobel_ver, {-1, -1}, 0.0,
                  cv::BORDER_ISOLATED);
  cv::sepFilter2D(image, Iy, image.depth(), sobel_ver, sobel_hor, {-1, -1}, 0.0,
                  cv::BORDER_ISOLATED);

  // Compute the three images corrsponding to the outer products of these
  // gradients, i.e. Ix and Iy as above.
  // Coz the structure tensor M is a 2x2 symmetric matrix, Ixy = Iyx.
  cv::Mat Ixx, Iyy, Ixy;
  Ixx = Ix.mul(Ix);
  Iyy = Iy.mul(Iy);
  Ixy = Ix.mul(Iy);

  // Convolve each of these images with a larger Gaussian to compute the
  // auto-correlation function, i.e. the SSD surface. That is the patch_size of
  // this Gaussian filter is greater than (or equal) the derivative of Gaussian
  // filter used above.
  // This is recommended, not mandatory instead.
  const int kSobelSize = 9;  // width(3) * height(3) = 9
  if (patch_size < kSobelSize) {
    LOG(INFO) << "patch_size should greater than or equal the size of the "
                 "Sobel operator";
  }
  //! We adopt the convention that the patch_radius is computed inclusively wrt.
  //! to the two end pixels. I.e. If it were treated as a circle, radius =
  //! patch_radius - 1.
  const int patch_radius = std::floor(patch_size / 2.0);
  // Check if the aperture size is odd or not, equivalently if the patch_radius
  // is even or not.
  if (patch_radius % 2 == 1) {
    LOG(ERROR) << "The aperture size should be odd to drive the "
                  "cv::getGaussianKernel to work properly";
  }
  const cv::Mat patch = cv::getGaussianKernel(patch_radius - 1, 1);
  cv::Mat ssd_Ixx, ssd_Iyy, ssd_Ixy;
  cv::filter2D(Ixx, ssd_Ixx, Ixx.depth(), patch, {-1, -1}, 0.0,
               cv::BORDER_ISOLATED);
  cv::filter2D(Iyy, ssd_Iyy, Iyy.depth(), patch, {-1, -1}, 0.0,
               cv::BORDER_ISOLATED);
  cv::filter2D(Ixy, ssd_Ixy, Ixy.depth(), patch, {-1, -1}, 0.0,
               cv::BORDER_ISOLATED);

  // Compute the Harris response R = det(M) - kappa * trace(M)^2,
  // where M is the structure tensor, aka. the second moment matrix.
  cv::Mat trace, determinant;
  trace = ssd_Ixx + ssd_Iyy;
  ssd_Ixy.forEach<double>(
      [](double x) { return x * x; });  // Element-wise square
  determinant = ssd_Ixy.mul(ssd_Iyy) - ssd_Ixy;
  // The eigen values of a matrix M=[a,b;c,d] are
  // lambda1/2 = (Tr(A)/2 +- ((Tr(A)/2)^2-det(A))^.5
  // The smaller one is the one with the negative sign
  // scores = trace/2 - ((trace/2).^2 - determinant).^0.5;
  // scores(scores<0) = 0;
  // scores = padarray(scores, [1+pr 1+pr]);
  cv::Mat response;
  cv::Mat trace2 = trace.clone();
  trace2.forEach<double>([](double x) { return x * x / 4.0; });
}

//@brief Compute the Shi-Tomasi cornerness, i.e. the Shi-Tomasi response for the
// image given patch_size.
//@param image Source image.
//@param shi_tomasi_response Output response matrix.
//@param patch_size Size of the image patch, that is the size of the weight
// window used to compute the auto-correlation function, aka. SSD surface. N.B
// patch_size = patch_radius + 1, where patch_radius is the distance between the
// pixels on the border of the image patch and the center of the image, aka. the
// anchor point. Note in OpenCV, the patch_radius is treated the same as
// aperture size.
void ShiTomasiResponse(const cv::Mat& image, cv::Mat& shi_tomasi_response,
                       const int patch_size) {
  //
}
#endif  // UZH_COMMON_CORNERNESS_H_