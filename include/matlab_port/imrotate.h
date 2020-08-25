#ifndef UZH_MATLAB_PORT_IMROTATE_H_
#define UZH_MATLAB_PORT_IMROTATE_H_

#include <cmath>

#include "opencv2/core.hpp"

//@ref
// https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html
//@ref https://stackoverflow.com/a/16159670/14007680
//@ref
// https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
cv::Mat ImRotate(const cv::Mat& image, const double degrees) {
  //@note Difference between std::remainder and std::fmod
  //@ref http://www.cplusplus.com/reference/cmath/remainder/
  //@note Difference between matlab's mod and rem.
  //@ref
  // https://www.mathworks.com/matlabcentral/answers/403870-difference-between-mod-and-rem-functions
  if (std::fmod(degrees, 90.0) == 0) {
    // Invoke OpenCV's cv::rotate function to speed up rotations of multiples of
    // 90 degrees.
    const int kMultipleOfNinety = std::fmod(std::floor(degrees / 90.0), 4.0);
    //! OpenCV's RotateFlags.
    //@ref
    // https://docs.opencv.org/4.3.0/d2/de8/group__core__array.html#ga6f45d55c0b1cc9d97f5353a7c8a7aac2
  } else {
    // Perform general rotation.
    // cv::getRotationMatrix2D()
    // cv::warpAffine()
  }
}

#endif  // UZH_MATLAB_PORT_IMROTATE_H_