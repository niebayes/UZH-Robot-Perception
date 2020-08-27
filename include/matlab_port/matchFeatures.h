#ifndef UZH_MATLAB_PORT_MATCHFEATURES_H_
#define UZH_MATLAB_PORT_MATCHFEATURES_H_

#include <algorithm>
#include <numeric>
#include <tuple>

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

enum MatchingMethod : int { EXHAUSTIVE, APPROXIMATE };

enum MatchingMetric : int {
  SSD,
  SAD,
  HAMMING  // Used only for binary features
};

//@brief Find matching features based on matching the two sets of descriptors.
std::tuple<arma::umat /*index_pairs*/, arma::mat /*match_metric*/>
matchFeatures(const arma::mat& query_descriptor,
              const arma::mat& database_descriptor,
              const double max_threshold = 10.0, const double max_ratio = 0.6,
              const int matching_method = uzh::EXHAUSTIVE,
              const int matching_metric = uzh::SSD,
              const bool unique_matches = true) {
  //
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_MATCHFEATURES_H_