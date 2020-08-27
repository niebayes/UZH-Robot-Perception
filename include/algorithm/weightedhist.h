#ifndef UZH_ALGORITHM_WEIGHTEDHIST_H_
#define UZH_ALGORITHM_WEIGHTEDHIST_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

//@brief Compute the histogram, i.e. count the values fall into each bin.
// A weighted histogram shows the weighted distribution of the data. If the
// histogram displays proportions (rather than raw counts), then the heights of
// the bars are the sum of the standardized weights of the observations within
// each bin.
//@param values Vector of values.
//@param weights Vector of weights associated with each element in values. The
// length of weights must be consistent with that of values.
//@param edges Vector of Boundaries based on which the values are delimited.
//@return The weighted histogram. The length of the histogram is the same with
// the that of edges.
//! The value values(i) is in the k-th bin if edges(k) ≤ values(i) < edges(k+1).
//! The last bin also includes the right bin boundary, such that it contains
//! values(i) if edges(end-1) ≤ X(i) ≤ edges(end), where the edges(end) denotes
//! the last element in edges. Values outside the edges(end) are not counted.
arma::vec /*weighted histogram*/
weightedhist(const arma::vec& values, const arma::vec& weights,
             const arma::vec& edges) {
  if (!values.is_vec() || !weights.is_vec() ||
      values.size() != weights.size()) {
    LOG(ERROR) << "values and weights must be the vectors with the same size.";
  }
  if (edges.empty() || !edges.is_vec() || !edges.is_sorted()) {
    LOG(ERROR) << "edges must be a non-empty vector with monotonically "
                  "increasing values.";
  }

  const int kNumEdges = edges.size();
  arma::vec hist = arma::zeros<arma::vec>(kNumEdges);

  // Iterate over each bin.
  for (int k = 0; k < kNumEdges - 1; ++k) {
    const arma::uvec indices =
        arma::find((edges(k) <= values) && (values < edges(k + 1)));

    if (!indices.empty()) {
      // The height of the bin is the sum of weights of the data rather than the
      // raw counts.
      hist(k) = arma::sum(weights(indices));
    }
  }
  
  // Deal with the values just on the right most boundary.
  const arma::uvec indices_boundary = arma::find(values == edges.back());
  if (!indices_boundary.empty()) {
    hist.back() = arma::sum(weights(indices_boundary));
  }

  return hist;
}

}  // namespace uzh

#endif  // UZH_ALGORITHM_WEIGHTEDHIST_H_