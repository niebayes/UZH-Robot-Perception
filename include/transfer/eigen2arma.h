#ifndef UZH_TRANSFER_EIGEN2ARMA_H_
#define UZH_TRANSFER_EIGEN2ARMA_H_

#include "Eigen/Core"
#include "armadillo"
#include "opencv2/core/eigen.hpp"
#include "transfer/cv2arma.h"
#include "transfer/eigen2cv.h"

namespace uzh {

// arma::mat eigen2arma(Eigen::MatrixXd &E, bool copy = true) {
//   return arma::mat(E.data(), E.rows(), E.cols(), /*copy_aux_mem*/ copy,
//                    /*strict*/ false);
// }

arma::mat eigen2arma(const Eigen::MatrixXd &E, bool copy = true) {
  cv::Mat cv_mat;
  cv::eigen2cv(E, cv_mat);
  return uzh::cv2arma<double>(cv_mat).t();
}

}  // namespace uzh

#endif  // UZH_TRANSFER_EIGEN2ARMA_H_