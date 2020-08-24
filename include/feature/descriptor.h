#ifndef UZH_FEATURE_DESCRIPTOR_H_
#define UZH_FEATURE_DESCRIPTOR_H_

#include "Eigen/Core"
#include "feature/convolution.h"
#include "opencv2/core.hpp"
#include "opencv2/core/eigen.hpp"

//@brief Describe keypoints using the intensities around the keypoints.
//@param image Input image to be filtered by a patch.
//@param keypoints Input [2 x n] matrix where each column contains the x and y
// coordinates of the keypoints detected and n is the number of keypoints.
//@param descriptors Output [(2*r+1)^2 x n] matrix where each column contains
// the intensities of the pixels around the patch center and r is the
// patch_radius.
//@param patch_radius Radius of the filter.
void DescribeKeypoints(const cv::Mat& image, const cv::Mat& keypoints,
                       cv::Mat& descriptors, const int patch_radius) {
  // Pre-padding to avoid boundary issues.
  const cv::Scalar_<int> pad_size{patch_radius, patch_radius, patch_radius,
                                  patch_radius};
  cv::Mat padded = image.clone();
  PadArray(padded, pad_size);

  // Convert to eigen to speed up computation
  Eigen::MatrixXi img;
  cv::cv2eigen(padded, img);

  // Construct descriptors matrix to be populated
  const int kPatchSize = 2 * patch_radius + 1;
  const int kNumKeypoints = keypoints.cols;
  Eigen::MatrixXi d(kPatchSize * kPatchSize, kNumKeypoints);

  // Collect intensities inside the patch centered around each keypoint and
  // unroll it to a column vector.
  //! As the keypoints are stored in descending order wrt. the response, the
  //! added descriptors are as well sorted based on the response.
  Eigen::Index x, y;

  for (int i = 0; i < kNumKeypoints; ++i) {
    // Add patch_radius to compensate the pre-padding.
    x = keypoints.at<int>(0, i) + patch_radius;
    y = keypoints.at<int>(1, i) + patch_radius;

    // Stack the intensities
    Eigen::MatrixXi patch =
        img.block(y - patch_radius, x - patch_radius, kPatchSize, kPatchSize);
    patch.resize(kPatchSize * kPatchSize, 1);
    d.col(i) = patch;
  }

  // Convert back to cv::Mat
  cv::eigen2cv(d, descriptors);

  // Normalize descriptors to 8-bit range, i.e. [0, 255]
  //! The "normalization" is actually refering to the bits convertion, that is
  //! assure the range of values of the descriptors is in 8-bit. This is
  //! accomplished by cv::convertTo rather than cv::normalize which is designed
  //! to normalize the scale and shift the values to accomodate some rules.
  //! This convertion is redundant since this function does not scale of shift
  //! the intensities of the pixels.
  cv::Mat normalized_descriptors;
  descriptors.convertTo(normalized_descriptors, CV_8U);
  descriptors = normalized_descriptors;
}

#endif  // UZH_FEATURE_DESCRIPTOR_H_