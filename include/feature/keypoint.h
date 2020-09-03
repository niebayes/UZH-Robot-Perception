#ifndef UZH_FEATURE_KEYPOINTS_H_
#define UZH_FEATURE_KEYPOINTS_H_

#include "Eigen/Core"
#include "feature/convolution.h"
#include "opencv2/core.hpp"
#include "opencv2/core/eigen.hpp"

namespace uzh {

//@brief Select keypoints according to the response matrix.
//@param response Input matrix containing the responses for each pixel computed
// from a certian response function.
//@param keypoints Output [2 x num_keypoints] matrix where each column contains
// the x and y coordinates(indices) of the selected keypoints.
//@param num_keypoints Number of keypoints to be selected.
//@param non_maximum_radius Within a circle of radius non_maximum_radius, the
// non-maximum suppresion is applied to even the distribution of the selected
// keypoints. The radius is computed as the distance between the border pixels
// and the center of the non-maximum patch inclusively wrt. to the end pixels.
//
//@note In practice, care has to be taken to select a threshold for suppressing
// keypoints not so strong. Despite the simplicity, the strategy we use - choose
// the top num_keypoints ranked by response strength - works generally well.
void SelectKeypoints(const cv::Mat& response, cv::Mat& keypoints,
                     const int num_keypoints, const int non_maximum_radius) {
  // Pad the response matrix for use in the non-maximum suppression stage later
  // on.
  const cv::Scalar_<int> kPadSize{non_maximum_radius, non_maximum_radius,
                                  non_maximum_radius, non_maximum_radius};
  cv::Mat_<double> response_tmp = response.clone();
  PadArray(response_tmp, kPadSize);

  // Use eigen to speed up the computation
  Eigen::MatrixXd R;
  Eigen::Matrix2Xi kpts(2, num_keypoints);
  cv::cv2eigen(response_tmp, R);

  // Select the top num_keypoints based on their responses and store the
  // corresponding x and y coordinates(indices) to the k matrix in order.
  //! By convention, x -> col, y -> row.
  Eigen::Index x, y;
  for (int i = 0; i < num_keypoints; ++i) {
    R.maxCoeff(&y, &x);
    //! Minus the non_maximum_radius to compensate for the pre-padding.
    kpts.col(i) = Eigen::Vector2i((int)x - non_maximum_radius,
                                  (int)y - non_maximum_radius);

    // Perform non-maximum suppresion: set the pixels within the circle of
    // non_maximum_radius radius to 0 including the keypoints we previously
    // selected. This assures the next keypoint won't be coincident with the
    // ones selected before.
    //! We adopt a box filter to simplify the codes.
    //! Due the pre-padding, no need to worry about boundary issues.
    // FIXME Error occurs here!
    R.block(y - non_maximum_radius, x - non_maximum_radius,
            2 * non_maximum_radius + 1, 2 * non_maximum_radius + 1)
        .setZero();
  }

  // Convert back to cv::Mat
  cv::eigen2cv(kpts, keypoints);
}

}  // namespace uzh

#endif  // UZH_FEATURE_KEYPOINTS_H_