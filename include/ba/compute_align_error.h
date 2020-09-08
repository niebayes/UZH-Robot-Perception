#ifndef UZH_BA_COMPUTE_ALIGN_ERROR_H_
#define UZH_BA_COMPUTE_ALIGN_ERROR_H_

#include "armadillo"
#include "ceres/ceres.h"
#include "transform/twist.h"

namespace uzh {

//@brief Cost functor used to compute the alignment error between the two 3D
// position vectors.
struct AlignError {
  AlignError(const arma::vec3& pp_G_C, const arma::vec3& p_V_C)
      : pp_G_C_(pp_G_C), p_V_C_(p_V_C) {}

  bool operator()(const double* const sim3, double* residuals) const {
    const arma::vec6 twist{sim3[0], sim3[1], sim3[2],
                           sim3[3], sim3[4], sim3[5]};
    const arma::mat44 SE3 = uzh::twist_to_SE3<double>(twist);
    const double& scale = sim3[6];
    const arma::vec3 p_G_C = scale * SE3(0, 0, arma::size(3, 3)) * p_V_C_ +
                             SE3(0, 3, arma::size(3, 1));

    residuals[0] = pp_G_C_(0) - p_G_C(0);
    residuals[1] = pp_G_C_(1) - p_G_C(1);
    residuals[2] = pp_G_C_(2) - p_G_C(2);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const arma::vec3& pp_G_C,
                                     const arma::vec3& p_V_C) {
    return (
        new ceres::NumericDiffCostFunction<AlignError, ceres::CENTRAL, 3, 7>(
            new AlignError(pp_G_C, p_V_C)));
  }

 private:
  arma::vec3 pp_G_C_;
  arma::vec3 p_V_C_;
};

}  // namespace uzh

#endif  // UZH_BA_COMPUTE_ALIGN_ERROR_H_