#ifndef UZH_MATLAB_PORT_PDIST2_H_
#define UZH_MATLAB_PORT_PDIST2_H_

#include <algorithm>  // std::stable_sort
#include <numeric>    // std::iota
#include <tuple>

#include "armadillo"
#include "feature/distance.h"
#include "glog/logging.h"

namespace uzh {

//@brief Return order used in pdist2 function.
enum ReturnOrder : int { SMALLEST_FIRST, LARGEST_FIRST };

//@brief Imitate matlab's pdist2. Calculate the pair-wise distance between every
// pair of the two sets of observations X and Y.
//@param X [m x p] matrix where m is the dimension of the observations and p is
// the number of observations. When used in feature matching, X is the database
// descriptors in which each column is a data descriptor.
//@param Y [m x q] matrix where m is the dimension of the observations and q is
// the number of observations. When used in feature matching, Y is the query
// descriptors in which each column is a query descriptor.
//@param distances Pointer to a [p x q] matrix where the (i, j)-th entry
// represents the pairwise distance between the i-th observation in X and the
// j-th observation in Y.
//@param distance Distance metric used to compute the pairwise distances. By
// default, Euclidean distance is applied.
//@param indices Optional pointer to a [p x q] matrix where the (i, j)-th entry
// contains the corresponding index of the observation in X with which the
// (i, j)-th distance in matrix distances is computed out.
//@param return_order Ascending or descending order when the parameter indices
// is passed into. Applied to each column of distances matrix, both the entries
// of distances matrix and indices matrix are reordered accordingly.
//@param num_returned Number of entries keeped for each column of distances
// matrix and indices matrix after the reordering.
//! Only when the indices matrix is passed into and the return_order and
//! num_returned are given by the user at the same time, this function would
//! then sort the D matrix and return the expected reduced D and I matrix.

std::tuple<arma::mat /* distances */, arma::umat /* indices */> pdist2(
    const arma::mat& X, const arma::mat& Y, const int metric = uzh::EUCLIDEAN,
    const int return_order = uzh::SMALLEST_FIRST, const int num_returned = -1) {
  if (X.empty() || Y.empty()) LOG(ERROR) << "Empty input.";
  if (X.n_rows != Y.n_rows)
    LOG(ERROR) << "Number of rows of X and Y must be consistent.";

  // Construct distances matrix to be populated.
  arma::mat distances(X.n_cols, Y.n_cols, arma::fill::zeros);

  // Compute pairwise distances.
  for (int j = 0; j < distances.n_cols; ++j) {
    for (int i = 0; i < distances.n_rows; ++i) {
      if (metric == uzh::EUCLIDEAN) {
        distances(i, j) = uzh::Euclidean(X.col(i), Y.col(j));
      } else if (metric == uzh::SQUARED_EUCLIDEAN) {
        distances(i, j) = uzh::SquaredEuclidean(X.col(i), Y.col(j));
      }
    }
  }

  // Initialize indices matrix.
  arma::umat indices(arma::size(distances), arma::fill::zeros);

  // Rearrange distances and indices if queried.
  if (num_returned > 0) {
    for (int c = 0; c < distances.n_cols; ++c) {
      arma::vec dist = distances.col(c);
      // Create the idx vector keeping track of the indices.
      arma::uvec idx =
          arma::linspace<arma::uvec>(0, dist.n_elem - 1, dist.n_elem);
      std::iota(idx.begin(), idx.end(), 0);

      // Sort the indices according to the values.
      if (return_order == uzh::SMALLEST_FIRST) {
        std::stable_sort(idx.begin(), idx.end(),
                         [&dist](int i, int j) { return dist[i] < dist[j]; });
      } else if (return_order == uzh::LARGEST_FIRST) {
        std::stable_sort(idx.begin(), idx.end(),
                         [&dist](int i, int j) { return dist[i] > dist[j]; });
      } else {
        LOG(ERROR) << "Invalid return order.";
      }

      // Populate indices.
      indices.col(c) = idx;
      // Sort distances according to the sorted indices.
      distances.col(c) = dist(idx);
    }

    // Truncate D and I according to the num_returned parameter.
    if (num_returned <= 0 || num_returned > distances.n_rows) {
      LOG(ERROR) << "num_retunred should be in range [1, D.rows]";
    }
    distances = distances.head_rows(num_returned).eval();
    indices = indices.head_rows(num_returned).eval();
  }

  return {distances, indices};
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_PDIST2_H_