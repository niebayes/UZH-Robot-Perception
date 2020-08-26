#ifndef UZH_MATLAB_PORT_IMGRADIENT_H_
#define UZH_MATLAB_PORT_IMGRADIENT_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {
enum GradientMethod : int { SOBEL };

arma::field<arma::mat> imgradient() {
  //
}
}  // namespace uzh

#endif  // UZH_MATLAB_PORT_IMGRADIENT_H_