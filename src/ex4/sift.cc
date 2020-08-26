#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <tuple>
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

//@brief Compute difference of Gaussians from adjacent images in each octave.
//@param blurred_images Blurred images obtained from ComputeBlurredImages.
//@return Difference of Gaussians of all octaves.
arma::field<arma::cube> ComputeDoGs(
    const arma::field<arma::cube>& blurred_images) {
  const int kNumOctaves = blurred_images.size();
  // DoGs contains DoGs of all octaves.
  arma::field<arma::cube> DoGs(kNumOctaves);

  //! The data layout:
  //! image pyramid ->
  //! num_octaves * octave ->
  //! num_images_per_octave * image ->
  //! one image -> one blurred image ->
  //! two adjacent blurred images -> one DoG
  // Hence the minus one in the size of DoG as below.
  for (int o = 0; o < kNumOctaves; ++o) {
    // DoG contains kNumDoGsPerOctave DoGs in a certain octave.
    arma::cube DoG = arma::zeros<arma::cube>(arma::size(blurred_images(o)) -
                                             arma::size(0, 0, 1));
    const int kNumDoGsPerOctave = DoG.n_slices;
    for (int d = 0; d < kNumDoGsPerOctave; ++d) {
      // arma::abs to ensure every element is not negative.
      DoG.slice(d) = arma::abs(blurred_images(o).slice(d + 1) -
                               blurred_images(o).slice(d));
    }

    DoGs(o) = DoG;
  }
  return DoGs;
}

//@brief Extract keypoints from the maximums both in scale and space.
//@param DoGs Difference of Gaussians computed from ComputeDoGs.
//@param keypoints_threshold Below which the points are suppressed before
// selection of keypoitns to attenuate effect of noise.
//@return Coordinates of the selected putative keypoints. The coordinates are 3D
// vectors where the first two dimension correspond to space and the last one
// corresponds to scale.
arma::field<arma::umat> ExtractKeypoints(const arma::field<arma::cube>& DoGs,
                                         const double keypoints_threshold) {
  const int kNumOctaves = DoGs.size();
  arma::field<arma::umat> keypoints(kNumOctaves);

  // For each octave, locate the maximums both in scale and space from the DoGs
  // in this octave.
  for (int o = 0; o < kNumOctaves; ++o) {
    // Copy the DoG
    arma::cube DoG = DoGs(o);

    // Locate the maximums
    //! A neat trick: use dilation in place of moving max filter.
    //@ref
    // https://en.wikipedia.org/wiki/Dilation_(morphology)#Flat_structuring_functions
    //! Here, by making use of armadillo's subview, we can achieve this in an
    //! efficient manner.
    // TODO(bayes) Modularize the codes below, wrapping into the imdilate
    // function.
    // First pre-pad the cube along each dimension to avoid boundary issues.
    const arma::cube kMaxFilter = arma::ones<arma::cube>(3, 3, 3);
    // TODO(bayes) Modularize the codes below, wrappint into the padarray
    // function.
    const int kPadSize = static_cast<int>(
        std::floor(kMaxFilter.n_rows / 2));  // Padding size equals to the
                                             // radius of the moving max filter.
    arma::cube padded_DoG = arma::zeros<arma::cube>(
        arma::size(DoG) + arma::size(kPadSize * 2, kPadSize * 2, kPadSize * 2));
    // Copy the elements from DoG to the padded_DoG
    padded_DoG(1, 1, 1, arma::size(DoG)) = DoG;
    // Apply moving max filter.
    // Iterate based on the unpadded DoG.
    arma::cube DoG_max = arma::zeros<arma::cube>(arma::size(DoG));
    for (int slice = 0; slice < DoG.n_slices; ++slice) {
      for (int col = 0; col < DoG.n_cols; ++col) {
        for (int row = 0; row < DoG.n_rows; ++row) {
          DoG_max(row, col, slice) =
              padded_DoG(row, col, slice, arma::size(kMaxFilter)).max();
        }
      }
    }

    // Filter out all points but the ones survived in the max filtering which
    // are then be thresholded.
    //! This is equivalent to doing a non-maximum suppression around the
    //! keypoints selected as above followed by a thresholding.
    //! Actually, based on the function provided by armadillo, we do
    //! thresholding followed by non-maximum suppression.
    arma::ucube is_kept_kpts(arma::size(DoG), arma::fill::zeros);
    // Thresholding
    std::cout << "before th: " << arma::size(arma::find(DoG.slice(o))) << '\n';
    // arma::cube DoG, DoG_max;  // Size: 907 x 1210 x 5.
    arma::ucube is_valid(arma::size(DoG), arma::fill::zeros);
    DoG.clean(keypoints_threshold);
    for (int s = 0; s < is_valid.n_slices; ++s) {
      is_valid.slice(s) = (DoG.slice(s) == DoG_max.slice(s));
    }
    for (int s = 0; s < is_kept_kpts.n_slices; ++s) {
      for (int c = 0; c < is_kept_kpts.n_cols; ++c) {
        for (int r = 0; r < is_kept_kpts.n_rows; ++r) {
          // Non-maximum suppression
          //! Consider using arma::approx_equal when the mat type is dmat (mat)
          //! or fmat, due to the necessarily limited precision of the
          //! underlying element types.
          if (DoG(r, c, s) == DoG_max(r, c, s) && DoG(r, c, s) != 0)
            is_kept_kpts(r, c, s) = 1;
        }
      }
      std::cout << arma::size(arma::find(is_kept_kpts.slice(s))) << '\n';
    }

    // Discard the keypoints found in the lowest and highest layers of the
    // current DoG because the two layers are involved with padded borders.
    is_kept_kpts.slice(0).zeros();
    is_kept_kpts.slice(is_kept_kpts.n_slices - 1).zeros();

    // Obtain the corresponding 3D coordinates of each putative keypoints.
    std::cout << "is_kept_kpts size: " << arma::size(is_kept_kpts) << '\n';
    arma::umat coordinates_3d =
        arma::ind2sub(arma::size(is_kept_kpts), arma::find(is_kept_kpts));

    // Store them into the keypoints.
    keypoints(o) = coordinates_3d;
  }

  return keypoints;
}

//@brief Compute descriptors from the patches around the putative keypoints.
//@param blurred_images Blurred images computed from the ComputeBlurredImages
// function. These images are used to locate the patches around the putative
// keypoints.
//@param keypoints Putative keypoints computed from the ExtractKeypoints
// function. These keypoints are then refined to find the final keypoints.
//@param descriptors containing the returned descriptors where each column is a
// descriptor vector.
//@param final_keypoints TODO
//@param rotation_invariant Boolean value used to denote whether the computed
// descriptors are invariant to rotation or not. If true, a dominant orientation
// will be assigned to each descriptor.
void ComputeDescriptors(const arma::field<arma::cube>& blurred_images,
                        const arma::field<arma::umat>& keypoints,
                        arma::mat& descriptors,
                        arma::field<arma::umat>& final_keypoints,
                        const bool rotation_invariant = false) {
  if (blurred_images.size() != keypoints.size())
    LOG(ERROR) << "The number of octaves are not consistent.";
  const int kNumOctaves = blurred_images.size();

  // The magic number 1.5 is taken from Lowe's paper.
  const arma::mat kGaussianWindow =
      uzh::cv2arma<double>(uzh::fspecial(uzh::GAUSSIAN, 16, 16.0 * 1.5)).t();

  for (int o = 0; o < kNumOctaves; ++o) {
    // Get the blurred images and keypoints in o-th octave.
    const arma::cube oct_blurred_images = blurred_images(o);
    const arma::umat oct_keypoints = keypoints(o);

    // Only consider relevant images involved in the extraction of the
    // keypoints we detected.

    // Extract the scale indices of the coordinates of the keypoints and
    // unique them. These indices are indicators from which we can tell which
    // images in the blurred images of the current octave have contributed to
    // the extraction of keypoints.
    const arma::urowvec kImageIndices = arma::unique(oct_keypoints.row(2));
    for (arma::uword img_idx : kImageIndices) {
      // Filter out irrelevant keypoints based on the image index
      const arma::uvec is_kept_in_image = (oct_keypoints.row(2) == img_idx);
      const arma::umat kept_keypoints = oct_keypoints(is_kept_in_image);
      const arma::umat kept_keypoints_xy = kept_keypoints.head_rows(2);

      // Compute image gradient for use of Histogram of Oriented Gradients.
      const arma::mat image = oct_blurred_images.slice(img_idx);
      const arma::cube gradient = uzh::imgradient(image);
      const arma::mat grad_magnitude = gradient.slice(0);
      const arma::mat grad_direction = gradient.slice(1);

      // Construct descriptor matrix to be populated.
      arma::mat descs;
      const int kNumKeypoints = kept_keypoints_xy.n_cols;
      // Mask to mask out all invalid keypoints that are out of the boundary.
      arma::uvec is_valid(kNumKeypoints, arma::fill::zeros);
      for (int corner_idx = 0; corner_idx < kNumKeypoints; ++corner_idx) {
        const int x = kept_keypoints_xy(0, corner_idx);
        const int y = kept_keypoints_xy(0, corner_idx);
        // Ensure all the pixels inside the patch are within the image boundary.
        // The patch is 16 x 16 with the radius not being odd. And we take the
        // point 8 pixels away from the upper left and 7 pixels aways from the
        // lower right as the anchor point.
        if (x >= 8 && x < image.n_cols - 7 && y >= 8 && y < image.n_rows - 7) {
          is_valid(corner_idx) = 1;
          // Convolve the patch with Gaussian window.
          // Gmag_loc = Gmag(row - 8 : row + 7, col - 8 : col + 7);
          // Gmag_loc_w = Gmag_loc.*gausswindow;
          // Gdir_loc = Gdir(row - 8 : row + 7, col - 8 : col + 7);

          // Gmag_loc_derotated_w = Gmag_loc_w;
          // Gdir_loc_derotated = Gdir_loc;
        }
      }
    }
  }
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
    std::cout << "image_pyramid:\n";
    for (auto& img : image_pyramid) std::cout << img.size << '\n';
    arma::field<arma::cube> blurred_images =
        ComputeBlurredImages(image_pyramid, kNumScales, kBaseSigma);
    std::cout << "blurred_images:\n";
    for (auto& imgs : blurred_images) std::cout << arma::size(imgs) << '\n';
    arma::field<arma::cube> DoGs = ComputeDoGs(blurred_images);
    std::cout << "DoGs:\n";
    for (auto& d : DoGs) std::cout << arma::size(d) << '\n';
    arma::field<arma::umat> keypoints_tmp =
        ExtractKeypoints(DoGs, kKeypointsThreshold);
    std::cout << "keypoints_tmp:\n";
    for (auto& k : keypoints_tmp) std::cout << arma::size(k) << '\n';
    arma::mat descriptors;
    arma::field<arma::umat> keypoints;
    ComputeDescriptors(blurred_images, keypoints_tmp, descriptors, keypoints,
                       false);

    std::cout << "descriptors:\n";
    std::cout << "final keypoints:\n";
  }

  arma::mat a{1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12};
  arma::mat b(3, 4, arma::fill::ones);
  a = arma::reshape(a, 3, 4);

  arma::umat comp = a == b;
  std::cout << comp << '\n';
  arma::umat is_valid = arma::ind2sub(arma::size(a), arma::find(comp));
  std::cout << is_valid << '\n';

  std::cout << arma::size(arma::find(a)) << '\n';
  std::cout << arma::size(arma::find(b)) << '\n';

  arma::mat m(907, 1210, arma::fill::randn);
  arma::cube g = uzh::imgradient(m);

  return EXIT_SUCCESS;
}
