#ifndef UZH_FEATURE_MATCHING_H_
#define UZH_FEATURE_MATCHING_H_

#include <algorithm>   // std::stable_sort
#include <functional>  // std::less, std::greater
#include <numeric>     // std::iota
#include <optional>    // std::optional

#include "Eigen/Core"
#include "common/type.h"

//@brief Imitate matlab's pdist. Calculate the pair-wise distance between every
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
void PDist2(const Eigen::Ref<const Eigen::MatrixXd>& X,
            const Eigen::Ref<const Eigen::MatrixXd>& Y,
            Eigen::MatrixXd* distances, int distance = EUCLIDEAN,
            std::optional<Eigen::MatrixXi*> indices = std::nullopt,
            int return_order = -1, int num_returned = -1) {
  // Assure the dimension of the descriptors is consistent.
  eigen_assert(X.rows() == Y.rows());

  // Construct distances matrix D to be populated.
  Eigen::MatrixXd D(X.cols(), Y.cols());
  D.setZero();

  // Compute pairwise distances
  // TODO(bayes) Vectorization technique applicable?
  for (int j = 0; j < D.cols(); ++j) {
    for (int i = 0; i < D.rows(); ++i) {
      if (distance == EUCLIDEAN) {
        D(i, j) = Euclidean(X.col(i), Y.col(j));
      }
      // else if (distance == ...) {...}
    }
  }

  // Rearrange matrix D and I according to the parameters.
  if (indices && return_order != -1 && num_returned != -1) {
    // Construct indices matrix I to be populated.
    Eigen::MatrixXi I(D.rows(), D.cols());
    I.setZero();

    // if (return_order == SMALLEST_FIRST) {
    //! The vec is itself a reference type. Hence no need to qualify it as
    //! reference and it's also forbidden.
    // Variable to keep track of column of matrix I (and D).
    Eigen::Index i = 0;
    for (auto vec : D.colwise()) {
      // Determin sorting rule.
      auto less = [&vec](int i, int j) { return vec[i] < vec[j]; };
      auto greater = [&vec](int i, int j) { return vec[i] > vec[j]; };
      std::function<bool(int, int)> comp;
      if (return_order == SMALLEST_FIRST) {
        comp = less;
      } else if (return_order == LARGEST_FIRST) {
        comp = greater;
      }

      // Create the idx vector keeping track of the indices.
      Eigen::VectorXi idx(vec.size());
      std::iota(idx.begin(), idx.end(), 0);

      //! Sort idx vector rather than vec vector.
      //@note using std::stable_sort instead of std::sort
      // to avoid unnecessary index re-orderings
      // when vec contains elements of equal values.
      std::stable_sort(idx.begin(), idx.end(), comp);

      // Populate I.
      I.col(i++) = idx;

      // Sort vec according to the sorted indices.
      //! New feature introduced since eigen 3.4
      vec = vec(idx).eval();
    }

    // Truncate D and I according to the num_returned parameter.
    if (num_returned <= 0 || num_returned > D.rows()) {
      LOG(ERROR) << "num_retunred should be in range [1, D.rows]";
    }
    D.noalias() = D.topRows(num_returned);
    I.noalias() = I.topRows(num_returned);

    // Output I
    *(indices.value()) = I;
  }
  // Output D
  *distances = D;
}

#endif  // UZH_FEATURE_MATCHING_H_