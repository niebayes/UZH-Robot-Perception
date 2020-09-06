#ifndef UZH_MATLAB_PORT_CONV2_H_
#define UZH_MATLAB_PORT_CONV2_H_

//! This option is for developers only.
//@ref C++ parfor https://stackoverflow.com/a/45773308/14007680
#ifdef __APPLE__
#define PARFOR 1
#else
#define PARFOR 0
#endif

#include <cmath>

#include "armadillo"
#include "glog/logging.h"
#include "matlab_port/padarray.h"

namespace uzh {

enum ConvolutionOption : int { FULL, SAME, VALID };

//@brief C = conv2(A, B) performs the 2-D convolution of matrices A and B.
// If [ma, na] = size(A), [mb, nb] = size(B), and [mc, nc] = size(C), then
// mc = max([ma+mb-1, ma, mb]) and nc = max([na+nb-1, na, nb]).
//@param A Matrix to be filtered.
//@param B Kernel.
//@param shape Convolution option chosen among FULL, SAME, VALID.
// FULL  - (default) returns the full 2-D convolution as the brief.
// SAME  - returns the central part of the convolution that is the same size as
// A. This accomplished by first computing the FULL convolution and then
// truncate out the central part;
// VALID - returns only those parts of the convolution that are computed without
// the zero-padded edges. size(C) = max([ma-max(0, mb-1), na-max(0, nb-1)], 0).
//@return C -- Filtered matrix.
arma::mat /* C */
conv2(const arma::mat& A, const arma::mat& B, const int shape = uzh::FULL) {
  if (A.empty() || B.empty()) LOG(ERROR) << "Empty input.";

  arma::mat C;
  if (shape == uzh::FULL)
    C = arma::conv2(A, B, "full");
  else if (shape == uzh::SAME)
    C = arma::conv2(A, B, "same");
  else if (shape == uzh::VALID) {
    // Compute the parts of the convolution that are computed without the
    // zero-padded edges.
    // Derive it from SAME.
    const int ma = A.n_rows, na = A.n_cols, mb = B.n_rows, nb = B.n_cols;
    const int radius_x = std::floor(nb / 2.0), radius_y = std::floor(mb / 2.0);
    const int start_row = radius_y, start_col = radius_x;
    const int valid_rows = std::max(ma - std::max(mb - 1, 0), 0),
              valid_cols = std::max(na - std::max(nb - 1, 0), 0);
    const arma::mat same = arma::conv2(A, B, "same");
    const arma::mat valid =
        same(start_row, start_col, arma::size(valid_rows, valid_cols));
    if (valid.empty())
      LOG(WARNING) << "VALID part of the convolution is empty.";
    C = valid;
  }

  return C;
}

//@brief Overloaded conv2 for separable filters.
// C = conv2(u, v, A) first convolve each row of A with the vector
// u and then convolve each column of the result with the vector v.  If
// nu = length(u), mv = length(v), and [mc,nc] = size(C) then
// mc = max([ma+mv-1, ma, mv]) and nc = max([na+nu-1, na, nu])
// conv2(u, v, A) is equivalent to conv2(A, v * u') up to
// round-off error, where ' denotes transpose.
//@param u Filter in x direction, i.e. convolve A row by row.
//@param v Filter in y direction, i.e. convolve A column by column.
//@param A Matrix to be filtered.
//@param shape Convolution option chosen among FULL, SAME, VALID.
// FULL  - (default) returns the full 2-D convolution;
// SAME  - returns the central part of the convolution that is the same size as
// A;
// VALID - returns only those parts of the convolution that are computed
// without the zero-padded edges. size(C) =
// max([ma-max(0, mb-1), na-max(0, nb-1)], 0).
//@param border_type Type of making borders.
// REPLICATE - the border values are the replication of the values on the very
// edges.
// CONSTANT  - the border values are all the same with the value provided by the
// user.
//@param border_value The value filled with borders if border_type is CONSTANT.
// By default, 0.0 is filled, i.e. zero-padding is applied.
//@return C -- Filtered matrix.
//! Note, the u, v are different with thosed used in matlab in that u is the
//! kernel along x direction while matlab's u is the kernel along y direction.
//! Same hold for v.
arma::mat /* C */
conv2(const arma::rowvec& u, const arma::vec& v, const arma::mat& A,
      const int shape = uzh::FULL, const int border_type = uzh::CONSTANT,
      const double border_value = 0.0) {
  if (u.empty() || v.empty() || A.empty()) LOG(ERROR) << "Empty input.";
  if (u.size() % 2 == 0 || v.size() % 2 == 0)
    LOG(ERROR) << "conv2 only supports odd-size kernels at this moment.";

  const int nu = u.size(), mv = v.size();
  const int ma = A.n_rows, na = A.n_cols;
  const int mc = std::max(std::max(ma + mv - 1, ma), mv),
            nc = std::max(std::max(na + nu - 1, na), nu);

  // Prepare for convolution.
  arma::mat full_C(mc, nc, arma::fill::zeros);
  // Convolution needs flipping around the kernel.
  const arma::rowvec flipped_u = arma::fliplr(u);
  const arma::vec flipped_v = arma::flipud(v);
  // Get the padding size for FULL C.
  const int pad_size_x = nc - na, pad_size_y = mc - ma;

  // First convolve with u row by row.
  // Only pad A along x direction.
  arma::mat u_padded =
      uzh::padarray<double>(A, pad_size_x, 0, border_type, border_value);
  arma::mat u_filtered(u_padded.n_rows, nc, arma::fill::zeros);
  for (int row = 0; row < u_filtered.n_rows; ++row) {
    for (int col = 0; col < u_filtered.n_cols; ++col) {
      u_filtered(row, col) =
          arma::accu(u_padded(row, col, arma::size(u)) % flipped_u);
    }
  }

  // TODO(bayes) Use C++17 std::for_each to parallelize the loops.

  // Then convolve the result with v column by column.
  // Only pad the result along y direction.
  arma::mat uv_padded = uzh::padarray<double>(u_filtered, 0, pad_size_y,
                                              border_type, border_value);
  arma::mat uv_filtered(mc, uv_padded.n_cols, arma::fill::zeros);
  for (int row = 0; row < uv_filtered.n_rows; ++row) {
    for (int col = 0; col < uv_filtered.n_cols; ++col) {
      uv_filtered(row, col) =
          arma::accu(uv_padded(row, col, arma::size(v)) % flipped_v);
    }
  }

  full_C = uv_filtered;
  arma::mat C;
  const int radius_u = std::floor(nu / 2.0), radius_v = std::floor(mv / 2.0);
  if (shape == uzh::FULL) {
    C = full_C;

  } else if (shape == uzh::SAME) {
    const int start_row = radius_v, start_col = radius_u;
    const arma::mat same_C = full_C(start_row, start_col, arma::size(A));
    C = same_C;

  } else if (shape == uzh::VALID) {
    const int start_row = 2 * radius_v, start_col = 2 * radius_u;
    const int valid_rows = std::max(ma - std::max(mv - 1, 0), 0),
              valid_cols = std::max(na - std::max(nu - 1, 0), 0);
    const arma::mat valid_C =
        full_C(start_row, start_col, arma::size(valid_rows, valid_cols));
    C = valid_C;

  } else {
    LOG(ERROR) << "Unsupported shape option.";
  }

  return C;
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_CONV2_H_