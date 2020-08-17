//@ref https://stackoverflow.com/a/10781165/14007680

#include "opencv2/opencv.hpp"

using namespace cv;

void cameraPoseFromHomography(const Mat& H, Mat& pose) {
  pose = Mat::eye(3, 4, CV_32FC1);  // 3x4 matrix, the camera pose
  float norm1 = (float)norm(H.col(0));
  float norm2 = (float)norm(H.col(1));
  float tnorm = (norm1 + norm2) / 2.0f;  // Normalization value

  Mat p1 = H.col(0);     // Pointer to first column of H
  Mat p2 = pose.col(0);  // Pointer to first column of pose (empty)

  cv::normalize(p1,
                p2);  // Normalize the rotation, and copies the column to pose

  p1 = H.col(1);     // Pointer to second column of H
  p2 = pose.col(1);  // Pointer to second column of pose (empty)

  cv::normalize(p1,
                p2);  // Normalize the rotation and copies the column to pose

  p1 = pose.col(0);
  p2 = pose.col(1);

  Mat p3 = p1.cross(p2);  // Computes the cross-product of p1 and p2
  Mat c2 = pose.col(2);   // Pointer to third column of pose
  p3.copyTo(c2);  // Third column is the crossproduct of columns one and two

  pose.col(3) = H.col(2) / tnorm;  // vector t [R|t] is the last column of pose
}