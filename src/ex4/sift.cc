#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "armadillo"
#include "transfer.h"
// #include "common.h"
// #include "feature.h"
#include "google_suite.h"
// #include "interpolation.h"
// #include "io.h"
#include "matlab_port.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

//@brief Return an image given file name, data depth and rescale factor.
//@param file_name String denoting the file name including the relative file
// path.
//@param rescale_factor The returned image is rescaled according to this factor.
//@return The rescaled image.
cv::Mat GetImage(const std::string& file_name,
                 const double rescale_factor = 1.0) {
  return uzh::im2double(uzh::imresize(
      cv::imread(file_name, cv::IMREAD_GRAYSCALE), rescale_factor));
}

//@brief Compute image pyramid by recursively decimating the original image.
//! This function generates original images of all octaves. That is
//! the resolutions of the original image in octave o and the original
//! image in octave o+1 differ by a factor of 2, i.e. the lower octave is
//! downsampled by a factor of 2.
//@param Image to be downsampled.
//@param num_octaves Number of octaves contained in this pyramid. An octave is
// nothing but a level of the pyramid.
//@return A image pyramid containing five images with different resolutions.
arma::field<cv::Mat> ComputeImagePyramid(const cv::Mat& image,
                                         const int num_octaves) {
  if (image.empty()) LOG(ERROR) << "Empty input image.";
  if (num_octaves <= 1) LOG(ERROR) << "Invalid num_octaves.";
  arma::field<cv::Mat> image_pyramid(num_octaves);
  image_pyramid(0) = image;
  for (int o = 1; o < num_octaves; ++o) {
    // Downsample by a factor of 2.
    image_pyramid(o) = uzh::imresize(image_pyramid(o - 1), 0.5);
  }
  return image_pyramid;
}

//@brief Compute blurred images for all images in the image pyramid.
//! Images in a certain octave are blurred with Gaussians of different sigmas.
//! Images in different octaves are blurred with the same set of Gaussians.
//@param image_pyramid Image pyramid containing set of original images to be
// blurred.
//@param num_scales Number of scales per octave from which the number of images
// per octave is computed.
//@param sigma Base sigma from which the set of sigmas used to get different
// Gaussians are generated.
arma::field<arma::cube> ComputeBlurredImages(
    const arma::field<cv::Mat>& image_pyramid, const int num_scales,
    const double base_sigma) {
  const int kNumOctaves = image_pyramid.size();
  // This formula is by observing that the each scale is formed from three DoG
  // images each of which is obtained from two Gaussian blurred images. Hence
  // the plus 3.
  const int kImagesPerOctave = num_scales + 3;
  arma::field<arma::cube> blurred_images(kNumOctaves);

  // Populate each octave with kImagesPerOctave images.
  for (int o = 0; o < kNumOctaves; ++o) {
    arma::cube octave = arma::zeros<arma::cube>(
        image_pyramid(o).rows, image_pyramid(o).cols, kImagesPerOctave);
    // Gaussian blur images in an octave with increasing sigmas.
    for (int i = 0; i < kImagesPerOctave; ++i) {
      // Such that kS = [-1, 0, ..., num_scales + 1], 6 indices in total.
      // FIXME This range could also be changed,
      // e.g. kS = [0, ..., num_scales + 2].
      cv::Mat blurred_image;
      const int kS = i - 2;
      const double kSigma = std::pow(2, kS / num_scales) * base_sigma;
      cv::GaussianBlur(image_pyramid(o), blurred_image, {}, kSigma, kSigma,
                       cv::BORDER_ISOLATED);
      octave.slice(i) = uzh::cv2arma<double>(blurred_image).t();
    }
    blurred_images(o) = octave;
  }

  return blurred_images;
}

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  // Given settings
  const int kNumOctaves = 5;  // Number of levels of the image pyramid.
  const int kNumScales = 3;   // Number of scales per octave.
  const double kBaseSigma =
      1.0;  // Sigma used to do Gaussian blurring on images. This value is
            // multiplied to generate sequences of sigmas.
  const double kKeypointsThreshold =
      0.04;  // Exceed which the keypoints is selected as a potential keypoints

  const std::string kFilePath{"data/ex4/"};
  cv::Mat img_1_show = cv::imread(kFilePath + "img_1.jpg", cv::IMREAD_COLOR);
  cv::Mat img_2_show = cv::imread(kFilePath + "img_1.jpg", cv::IMREAD_COLOR);
  // cv::Mat img_1, img_2;
  // cv::cvtColor(img_1_show, img_1, cv::COLOR_BGR2GRAY, 1);
  // cv::cvtColor(img_2_show, img_2, cv::COLOR_BGR2GRAY, 1);

  // Decimate the images for speed.
  // The original images are [3024 x 4032 x 3] color images.
  const double kRescaleFactor = 0.3;
  cv::Mat left_image = GetImage(kFilePath + "img_1.jpg", kRescaleFactor);
  cv::Mat right_image = GetImage(kFilePath + "img_2.jpg", kRescaleFactor);

  // Degrees to which the right image is rotated to test SIFT rotation
  // invariance. Positive -> counter-clockwise and negative -> clockwise.
  const double degree = 0;
  if (degree != 0) {
    right_image = uzh::imrotate(right_image, degree);
  }

  // Construct a field of images to be processed iteratively.
  arma::field<cv::Mat> images(2);
  images(0) = left_image;
  images(1) = right_image;

  // Construct fields of keypoints and descriptors to be populated.
  arma::field<cv::Mat> keypoints(2);
  arma::field<cv::Mat> descriptors(2);

  for (int i = 0; i < images.size(); ++i) {
    // Compute the image pyramid.
    // The returned image pyramid contains five images with different
    // resolutions that are later on feed into the ComputeBlurredImages function
    // to generate images of five octaves with each octave containing 6 images
    // blurred with different sigma values.
    arma::field<cv::Mat> image_pyramid =
        ComputeImagePyramid(images(i), kNumOctaves);
    for (auto& img : image_pyramid) std::cout << img.size << '\n';
    arma::field<arma::cube> blurred_images =
        ComputeBlurredImages(image_pyramid, kNumScales, kBaseSigma);
  }

  return EXIT_SUCCESS;
}
