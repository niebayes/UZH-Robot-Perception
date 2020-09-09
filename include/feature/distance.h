#ifndef UZH_FEATURE_DISTANCE_H_
#define UZH_FEATURE_DISTANCE_H_

#include <cmath>

#include "Eigen/Core"

namespace uzh {

//@brief General distance metrics.
enum Distance : int { EUCLIDEAN, SQUARED_EUCLIDEAN };

inline double Euclidean(const Eigen::Ref<const Eigen::VectorXd>& p,
                        const Eigen::Ref<const Eigen::VectorXd>& q) {
  eigen_assert(p.size() == q.size());
  return std::sqrt((p - q).array().square().sum());
}

inline double SquaredEuclidean(const Eigen::Ref<const Eigen::VectorXd>& p,
                               const Eigen::Ref<const Eigen::VectorXd>& q) {
  eigen_assert(p.size() == q.size());
  return (p - q).array().square().sum();
}

}  // namespace uzh

#endif  // UZH_FEATURE_DISTANCE_H_