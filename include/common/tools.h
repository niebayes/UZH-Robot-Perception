#ifndef UZH_COMMON_TOOLS_H_
#define UZH_COMMON_TOOLS_H_

#include <algorithm>  
#include <functional>

#include "Eigen/Core"

//@brief Wrap STL's std::min_element and std::remove_if. Find the minimum
// not satisfying the given rule.
template <typename T, typename Derived>
T find_min_if_not(const Eigen::DenseBase<Derived>& X,
                  std::function<bool(typename Derived::Scalar)> pred) {
  // FIXME template followed dot operator works well on Linux, but not on macOS
  // Derived X_(X.size());
  // std::copy(X.cbegin(), X.cend(), X_.template begin());
  // return (*std::min_element(
  //     X_.template begin(),
  //     std::remove_if(X_.template begin(), X_.template end(), pred)));
  Derived X_(X.size());
  std::copy(X.cbegin(), X.cend(), X_.begin());
  return (*std::min_element(X_.begin(),
                            std::remove_if(X_.begin(), X_.end(), pred)));
}

//@brief Remove zeros in an array.
template <typename T>
T remove_zeros(const T& A) {
  //! Alternative way.
  // std::vector<int> A_(A.cbegin(), A.cend());
  // std::vector<int>::const_iterator last_non_zero =
  //     std::remove_if(A_.begin(), A_.end(), [](int x) { return x != 0; });
  // std::vector<int> A_no_zeros(A_.cbegin(), last_non_zero);
  const auto num_non_zeros =
      std::count_if(A.cbegin(), A.cend(), [](int x) { return x != 0; });
  T A_no_zeros(num_non_zeros);
  std::copy_if(A.cbegin(), A.cend(), A_no_zeros.begin(),
               [](int x) { return x != 0; });
  return A_no_zeros;
}

#endif  // UZH_COMMON_TOOLS_H_