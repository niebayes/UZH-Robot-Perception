#ifndef UZH_FEATURE_DISTANCE_H_
#define UZH_FEATURE_DISTANCE_H_

#include <cassert>
#include <cmath>

#include "Eigen/Core"
#include "armadillo"

namespace uzh {

//@brief General distance metrics.
enum Distance : int { EUCLIDEAN, SQUARED_EUCLIDEAN };

inline double Euclidean(const Eigen::Ref<const Eigen::VectorXd>& p,
                        const Eigen::Ref<const Eigen::VectorXd>& q) {
  eigen_assert(p.size() == q.size());
  return std::sqrt((p - q).array().square().sum());
}

//@brief Overloaded for armadillo.
inline double Euclidean(const arma::vec& p, const arma::vec& q) {
  assert(p.n_elem == q.n_elem);
  return std::sqrt(arma::accu(arma::square(p - q)));
}

inline double SquaredEuclidean(const Eigen::Ref<const Eigen::VectorXd>& p,
                               const Eigen::Ref<const Eigen::VectorXd>& q) {
  eigen_assert(p.size() == q.size());
  return (p - q).array().square().sum();
}

//@brief Overloaded for armadillo
inline double SquaredEuclidean(const arma::vec& p, const arma::vec& q) {
  assert(p.n_elem == q.n_elem);
  return arma::accu(arma::square(p - q));
}

}  // namespace uzh

#endif  // UZH_FEATURE_DISTANCE_H_