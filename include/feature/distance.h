#ifndef UZH_FEATURE_DISTANCE_H_
#define UZH_FEATURE_DISTANCE_H_

#include <cmath>

#include "Eigen/Core"

namespace uzh {

// TODO(bayes) Templatize these functions to make it more flexible.
// E.g. Appplicable both on column vectors and row vectors.
// More advanced: return vectors of pairwise distances when the input are
// matrices.

static double Euclidean(const Eigen::Ref<const Eigen::VectorXd>& p,
                        const Eigen::Ref<const Eigen::VectorXd>& q) {
  eigen_assert(p.size() == q.size());
  return std::sqrt((p - q).array().square().sum());
}

static double SquaredEuclidean(const Eigen::Ref<const Eigen::VectorXd>& p,
                               const Eigen::Ref<const Eigen::VectorXd>& q) {
  eigen_assert(p.size() == q.size());
  return (p - q).array().square().sum();
}

}  // namespace uzh

#endif  // UZH_FEATURE_DISTANCE_H_