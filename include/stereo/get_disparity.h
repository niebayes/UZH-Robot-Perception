#ifndef UZH_STEREO_GET_DISPARITY
#define UZH_STEREO_GET_DISPARITY

#include "armadillo"
#include "matlab_port/pdist2.h"

//@brief
arma::mat GetDisparity(const arma::mat& left_img, const arma::mat& right_img,
                       const int patch_radius, const double min_disparity,
                       const double max_disparity) {
  //
}

#endif  // UZH_STEREO_GET_DISPARITY