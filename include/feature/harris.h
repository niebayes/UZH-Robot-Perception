#ifndef UZH_FEATURE_HARRIS_H_
#define UZH_FEATURE_HARRIS_H_

#include <cmath>  // std::floor

#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/core/eigen.hpp"

//@brief Compute the Harris cornerness, i.e. Harris response for the image given
// patch_size and kappa.
//@param image Source image
//@param harris_response Output response matrix
//@param patch_size Size of the image patch, that is the size of the weight
// window used to compute the auto-correlation function, aka. SSD surface. N.B
// patch_size = patch_radius + 1, where patch_radius is the distance between the
// pixels on the border of the image patch and the center of the image, aka. the
// anchor point. Note in OpenCV, the patch_radius is treated the same as the
// aperture size.
//@param kappa Parameter used in Harris response function. The default 0.06 is
// an empirically well-investigated value.
//@note InputArray or Mat?
//@ref https://stackoverflow.com/a/31820323/14007680
void HarrisResponse(const cv::Mat& image, cv::Mat& harris_response,
                    const int patch_size, const double kappa = 0.06) {
  // Compute the horizontal and vertical derivatives of the image Ix and Iy by
  // convolving the original image with derivatives of Gaussians.
  //! The input image is compensated for noise, so there's no need to apply a
  //! Gaussian filter in advance to attenuate the effect of noise. Thus we
  //! simply apply the Sobel derivative filters.
  // Here, the Sobel operator is applied and is separated into two 1D filters
  // sobel_hor and sobel_ver.
  // Sobel operators:
  // Gx = [[-1 0 +1],  Gy = [[-1 -2 -1],
  //       [-2 0 +2],        [ 0  0  0],
  //       [-1 0 +1]]        [+1 +2 +1]]
  // Separated 1D components:
  // sobel_hor = [-1 0 +1]', sobel_ver = [+1 +2 +1]', both are column vectors.
  // Hence, Gx = sobel_ver * sobel_hor', Gy = sobel_hor * sobel_ver'.
  cv::Mat Ix, Iy;
  const cv::Mat sobel_hor = (cv::Mat_<double>(3, 1) << -1, 0, 1);
  const cv::Mat sobel_ver = (cv::Mat_<double>(3, 1) << 1, 2, 1);
  //@note OpenCV's cv::filter2D, cv::sepFilter2D and other filter functions
  // actually do correlation rather than convolution. To do convolution, use
  // cv::flip to flip the kernels along the anchor point (default the kernel
  // center) in advance. N.B. For symmetric kernels, this step could be skipped.
  // The new anchor point can be computed as (kernel.cols - anchor.x - 1,
  // kernel.rows - anchor.y - 1). For separable filters as well as the Sobel
  // operators, the flipping operation can be accomplished with alternating the
  // sign of the sobel_hor or sobel_ver.
  //@note cv::BORDER_ISOLATED
  // When the source image is a part (ROI) of a bigger image, the function will
  // try to use the pixels outside of the ROI to form a border. To disable this
  // feature and always do extrapolation, as if src was not a ROI, use
  // borderType | BORDER_ISOLATED
  // TODO(bayes) Replace the cv's filter functions with self-implemented Conv2D.
  cv::sepFilter2D(image, Ix, image.depth(), -sobel_hor.t(), sobel_ver, {-1, -1},
                  0.0, cv::BORDER_ISOLATED);
  cv::sepFilter2D(image, Iy, image.depth(), -sobel_ver.t(), sobel_hor, {-1, -1},
                  0.0, cv::BORDER_ISOLATED);

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
  const int patch_radius = static_cast<int>(std::floor(patch_size / 2.0));
  // Check if the aperture size is odd or not, equivalently if the patch_radius
  // is even or not.
  if (patch_radius % 2 == 1) {
    LOG(ERROR) << "The aperture size should be odd to drive the "
                  "cv::getGaussianKernel to work properly";
  }
  // const cv::Mat patch = cv::getGaussianKernel(patch_radius - 1, 1);
  const cv::Mat patch = cv::Mat::ones(patch_size, patch_size, image.depth());
  cv::Mat ssd_Ixx, ssd_Iyy, ssd_Ixy;
  cv::filter2D(Ixx, ssd_Ixx, Ixx.depth(), patch, {-1, -1}, 0.0,
               cv::BORDER_ISOLATED);
  cv::filter2D(Iyy, ssd_Iyy, Iyy.depth(), patch, {-1, -1}, 0.0,
               cv::BORDER_ISOLATED);
  cv::filter2D(Ixy, ssd_Ixy, Ixy.depth(), patch, {-1, -1}, 0.0,
               cv::BORDER_ISOLATED);

  // Compute the Harris response R = det(M) - kappa * trace(M)^2,
  // where M is the structure tensor, aka. the second moment matrix.
  // The vectorization trick - expressing the coefficients in M as matrices
  // representing the entire filtered image - accelates the computation and
  // simplify the codes.

  // For the sake of computation simplicity, convert cv::Mat to Eigen::Matrix.
  Eigen::MatrixXd s_Ixx, s_Iyy, s_Ixy;
  cv::cv2eigen(ssd_Ixx, s_Ixx);
  cv::cv2eigen(ssd_Iyy, s_Iyy);
  cv::cv2eigen(ssd_Ixy, s_Ixy);

  // Compute trace and determinant.
  // The structure tensor M = [a, b; c, d] and the trace is computed as trace =
  // a + d while the determinant = a*d - b*c.
  Eigen::MatrixXd trace, determinant;
  trace = s_Ixx.array() + s_Iyy.array();
  determinant = s_Ixx.cwiseProduct(s_Iyy) - s_Ixy.cwiseProduct(s_Ixy);

  Eigen::MatrixXd response;
  response = determinant - kappa * trace.cwiseProduct(trace);
  // Simply set all responses smaller than 0 to 0.
  response = response.unaryExpr([](double x) { return x < 0.0 ? 0.0 : x; });

  // Convert back to cv::Mat and store it to the output harris_response.
  cv::eigen2cv(response, harris_response);
  // Pad the harris_response making its size consistent with the input image.
  // And set the pixels on borders to 0. When the dst.size > src.size whereby
  // diff < 0, the function truncates the src image, which is exactly what we
  // need.
  const int diff = image.rows - harris_response.rows;
  cv::copyMakeBorder(harris_response, harris_response, diff, diff, diff, diff,
                     cv::BORDER_CONSTANT, {0, 0, 0, 0});
}

#endif  // UZH_FEATURE_HARRIS_H_