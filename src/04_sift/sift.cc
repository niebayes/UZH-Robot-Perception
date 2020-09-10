#include "sift.h"

#include <string>
#include <tuple>  // std::tie

#include "armadillo"
#include "feature/matching.h"
#include "google_suite.h"
#include "matlab_port.h"
#include "opencv2/opencv.hpp"
#include "transfer.h"

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string file_path{"data/04_sift/"};

  // Given settings
  const int kNumOctaves = 5;  // Number of levels of the image pyramid.
  const int kNumScales = 3;   // Number of scales per octave.
  const double kBaseSigma =
      1.0;  // Sigma used to do Gaussian blurring on images. This value is
            // multiplied to generate sequences of sigmas.
  const double kKeypointsThreshold =
      0.05;  // Exceed which the keypoints is selected as a potential keypoints

  // Color images to be rendered with detected keypoints
  cv::Mat img_1_show = cv::imread(file_path + "img_1.jpg", cv::IMREAD_COLOR);
  cv::Mat img_2_show = cv::imread(file_path + "img_2.jpg", cv::IMREAD_COLOR);

  // Decimate the images for speed.
  // The original images are [3024 x 4032 x 3] color images.
  const double kRescaleFactor = 0.3;
  cv::Mat left_image =
      uzh::GetImageSIFT(file_path + "img_1.jpg", kRescaleFactor);
  cv::Mat right_image =
      uzh::GetImageSIFT(file_path + "img_2.jpg", kRescaleFactor);
  const int kResizedRows = left_image.rows, kResizedCols = left_image.cols;
  arma::field<cv::Mat> imgs_show(2);
  imgs_show(0) = uzh::imresize(img_1_show, kResizedRows, kResizedCols);
  imgs_show(1) = uzh::imresize(img_2_show, kResizedRows, kResizedCols);

  // Degrees to which the right image is rotated to test SIFT rotation
  // invariance. Positive -> counter-clockwise and negative -> clockwise.
  const double kDegree = 0;
  if (kDegree != 0) {
    // TODO(bayes) Implement imrotate to do general rotation.
    right_image = uzh::imrotate(right_image, kDegree);
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
    // resolutions that are later on fed into the ComputeBlurredImages
    // function to generate images of five octaves with each octave containing 6
    // images blurred with different sigma values.
    const arma::field<cv::Mat> image_pyramid =
        uzh::ComputeImagePyramid(images(i), kNumOctaves);
    const arma::field<arma::cube> blurred_images =
        uzh::ComputeBlurredImages(image_pyramid, kNumScales, kBaseSigma);
    const arma::field<arma::cube> DoGs = uzh::ComputeDoGs(blurred_images);
    const arma::field<arma::umat> keypoints_tmp =
        uzh::ExtractKeypoints(DoGs, kKeypointsThreshold);
    std::tie(descriptors(i), keypoints(i)) =
        uzh::ComputeDescriptors(blurred_images, keypoints_tmp, false);
    LOG(INFO) << "Detected " << keypoints(i).n_cols << " keypoints on img_"
              << i + 1;
  }

  // Display detected keypoints
  for (int img_idx = 0; img_idx < 2; ++img_idx) {
    arma::urowvec kpts_x = keypoints(img_idx).row(1);
    arma::urowvec kpts_y = keypoints(img_idx).row(0);
    const int num_kpts = kpts_x.size();
    for (int i = 0; i < num_kpts; ++i) {
      cv::circle(imgs_show(img_idx),
                 {static_cast<int>(kpts_x(i)), static_cast<int>(kpts_y(i))}, 4,
                 {50, 50, 255}, cv::FILLED);
    }
    cv::imshow(cv::format("Detected SIFT keypoints on img_%d", img_idx + 1),
               imgs_show(img_idx));
    cv::waitKey(0);
  }

  // Match descriptors
  // TODO Implement matchFeatures extending simple thresholding based matching
  // to distance ratio rejection based matching.
  // FIXME Possible errors.
  cv::Mat query_descriptor, database_descriptor;
  query_descriptor = uzh::arma2cv<double>(descriptors(0));
  database_descriptor = uzh::arma2cv<double>(descriptors(1));
  cv::Mat matches;
  uzh::MatchDescriptors(query_descriptor, database_descriptor, matches, 3);
  arma::umat match_indices = uzh::cv2arma<arma::uword>(matches);
  // FIXME Will the arma::find discard 0 index which is matched though?
  LOG(INFO) << "Number of matched keypoint pairs: "
            << arma::size(arma::find(match_indices)).n_rows;

  // Display matched keypoints
  //! Not intended to reinvent the wheel.

  return EXIT_SUCCESS;
}
