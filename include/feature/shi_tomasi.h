#ifndef UZH_FEATURE_SHI_TOMASI_H_
#define UZH_FEATURE_SHI_TOMASI_H_

#include "opencv2/core.hpp"

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
 
#endif  // UZH_FEATURE_SHI_TOMASI_H_