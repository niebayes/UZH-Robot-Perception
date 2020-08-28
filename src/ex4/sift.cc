#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "Eigen/Dense"
#include "algorithm.h"
#include "armadillo"
#include "transfer.h"
// #include "common.h"
#include "google_suite.h"
// #include "interpolation.h"
// #include "io.h"
#include "feature/matching.h"
#include "matlab_port.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"
#include "sift.h"

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
  cv::Mat img_2_show = cv::imread(kFilePath + "img_2.jpg", cv::IMREAD_COLOR);
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
  arma::field<arma::umat> keypoints(2);
  arma::field<arma::mat> descriptors(2);

  for (int i = 0; i < images.size(); ++i) {
    // Compute the image pyramid.
    // The returned image pyramid contains five images with different
    // resolutions that are later on feed into the ComputeBlurredImages function
    // to generate images of five octaves with each octave containing 6 images
    // blurred with different sigma values.
    arma::field<cv::Mat> image_pyramid =
        ComputeImagePyramid(images(i), kNumOctaves);
    arma::field<arma::cube> blurred_images =
        ComputeBlurredImages(image_pyramid, kNumScales, kBaseSigma);
    arma::field<arma::cube> DoGs = ComputeDoGs(blurred_images);
    arma::field<arma::umat> keypoints_tmp =
        ExtractKeypoints(DoGs, kKeypointsThreshold);
    ComputeDescriptors(blurred_images, keypoints_tmp, descriptors(i),
                       keypoints(i), false);
    std::cout << "descriptors:\n";
    for (auto& d : descriptors) std::cout << arma::size(d) << '\n';
    std::cout << "final keypoints:\n";
    for (auto& k : keypoints) std::cout << arma::size(k) << '\n';
  }

  // for (arma::umat kpts : keypoints) {
  //   arma::urowvec kpts_x = kpts.row(0);
  //   arma::urowvec kpts_y = kpts.row(1);
  //   const int num_kpts = kpts_x.size();
  //   Eigen::VectorXi x(num_kpts), y(num_kpts);
  //   for (int i = 0; i < num_kpts; ++i) {
  //     x(i) = kpts_x(i);
  //     y(i) = kpts_y(i);
  //   }
  //   uzh::scatter(img_1_show, x, y, 10, {0, 0, 255}, cv::FILLED);
  //   cv::imshow("", img_1_show);
  //   cv::waitKey(0);

  //   uzh::scatter(img_2_show, x, y, 10, {0, 255, 0}, cv::FILLED);
  //   cv::imshow("", img_2_show);
  //   cv::waitKey(0);
  // }

  // Match descriptors
  // cv::Mat query_descriptor, database_descriptor;
  // database_descriptor = uzh::arma2cv<double>(descriptors(0));
  // query_descriptor = uzh::arma2cv<double>(descriptors(1));
  // std::cout << "query.size" << '\n';
  // std::cout << query_descriptor.size << '\n';
  // cv::Mat matches;
  // MatchDescriptors(query_descriptor, database_descriptor, matches, 4.0);

  // cv::Mat query_keypoints, database_keypoints;
  // cv::drawMatches();

  // arma::mat x(5, 5, arma::fill::randn);
  // x.print("x");
  // arma::abs(x).print("abs");

  return EXIT_SUCCESS;
}
