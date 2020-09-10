#ifndef UZH_MATLAB_PORT_UNIQUE_H_
#define UZH_MATLAB_PORT_UNIQUE_H_

#include <algorithm>
#include <set>
#include <tuple>
#include <vector>

#include "armadillo"

namespace uzh {

//@brief Imitate matlab's unique. Unique values in an array and store them in a
// sorted order (by default descending).
//@param A One dimensional array containing the original values.
//@return C One dimensional array containing the unique values of A.
//@return
// ia -- One dimensional array containing indices such that C = A(ia).
// ic -- One dimensional array containing indices such that A = C(ic).
//@note How to implement matlab's unique using C++?
//@ref https://stackoverflow.com/q/63537619/14007680
//@note Return unmovable and uncopyable values with C++17.
//@ref https://stackoverflow.com/a/38531743/14007680
template <typename T>
std::tuple<arma::Col<T> /*C*/, std::vector<int> /*ia*/, std::vector<int> /*ic*/>
unique(const arma::Col<T>& A) {
  // Copy the values of A.
  const std::vector<double> original(A.begin(), A.end());

  // Get uniqued values.
  const std::set<double> uniqued(original.begin(), original.end());
  // Eigen::ArrayXd uniqued;

  // Get ia.
  std::vector<int> indices_ori;
  indices_ori.reserve(uniqued.size());
  std::transform(uniqued.cbegin(), uniqued.cend(),
                 std::back_inserter(indices_ori), [&original](double x) {
                   return std::distance(
                       original.cbegin(),
                       std::find(original.cbegin(), original.cend(), x));
                 });

  // Get ic.
  std::vector<int> indices_uni;
  indices_uni.reserve(original.size());
  std::transform(original.cbegin(), original.cend(),
                 std::back_inserter(indices_uni), [&uniqued](double x) {
                   return std::distance(
                       uniqued.cbegin(),
                       std::find(uniqued.cbegin(), uniqued.cend(), x));
                 });

  // Output C.
  arma::Col<T> C_out(uniqued.size());
  std::copy(uniqued.cbegin(), uniqued.cend(), C_out.begin());

  return {C_out, indices_ori, indices_uni};
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_UNIQUE_H_