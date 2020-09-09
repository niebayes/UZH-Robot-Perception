#ifndef UZH_BA_COMPUTE_REPROJECTION_ERROR_H_
#define UZH_BA_COMPUTE_REPROJECTION_ERROR_H_

#include "armadillo"
#include "ba/project_point.h"
#include "ceres/ceres.h"
#include "transform/twist.h"

namespace uzh {

//@brief Compute reprojection error.
struct ReprojectionError {
  ReprojectionError(const arma::vec2& observation, const arma::mat33& K)
      : observation_(observation), K_(K) {}

  bool operator()(const double* const twist, const double* const landmark,
                  double* residuals) const {
    const arma::vec6 twist_W_C(twist);
    const arma::vec3 p_W_landmark(landmark);
    const arma::mat44 SE3 = uzh::twist_to_SE3<double>(twist_W_C).i();
    const arma::vec3 p_C_landmark = SE3(0, 0, arma::size(3, 3)) * p_W_landmark +
                                    SE3(0, 3, arma::size(3, 1));
    const arma::vec2 projection = uzh::ProjectPoint(p_C_landmark, K_);

    residuals[0] = observation_(0) - projection(0);
    residuals[1] = observation_(1) - projection(1);

    return true;
  }

  static ceres::CostFunction* Create(const arma::vec2& observation,
                                     const arma::mat33& K) {
    return (new ceres::NumericDiffCostFunction<ReprojectionError,
                                               ceres::CENTRAL, 2, 6, 3>(
        new ReprojectionError(observation, K)));
  }

 private:
  arma::vec2 observation_;
  arma::mat33 K_;
};

}  // namespace uzh

#endif  // UZH_BA_COMPUTE_REPROJECTION_ERROR_H_