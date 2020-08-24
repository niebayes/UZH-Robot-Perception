#ifndef UZH_MATLAB_PORT_UNIQUE_H_
#define UZH_MATLAB_PORT_UNIQUE_H_

#include <algorithm>
#include <set>
#include <tuple>
#include <vector>

#include "Eigen/Core"

//@brief Imitate matlab's unique. Unique values in an array and store them in a
// sorted order (by default descending).
//@param A One dimensional array containing the original values.
//@return C One dimensional array containing the unique values of A.
//@return ia One dimensional array containing indices such that C = A(ia).
//@return ic One dimensional array containing indices such that A = C(ic).
//
//! Eigen provides surprising flexibility and genericity, so you can pass as
//! arguments ArrayXd, VectorXd, RowVectorXd as well as MatrixXd to the
//! parameter A and the returned object C. They all work properly.
//
//@note How to implement matlab's unique using C++?
//@ref https://stackoverflow.com/q/63537619/14007680
//@note Return unmovable and uncopyable values with C++17.
//@ref https://stackoverflow.com/a/38531743/14007680
// TODO(bayes) Templatize this function to make the parameter parsing more
// flexible.
std::tuple<Eigen::ArrayXd /*C*/, std::vector<int> /*ia*/,
           std::vector<int> /*ic*/>
Unique(const Eigen::ArrayXd& A) {
  //! Alternatively, explicitly passing into pointers.
  // Eigen::ArrayXd* C,
  // std::optional<Eigen::ArrayXi*> ia = std::nullopt,
  // std::optional<Eigen::ArrayXi*> ic = std::nullopt) {

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
  Eigen::ArrayXd C_out(uniqued.size());
  std::copy(uniqued.cbegin(), uniqued.cend(), C_out.begin());

  return {C_out, indices_ori, indices_uni};
}

#endif  // UZH_MATLAB_PORT_UNIQUE_H_