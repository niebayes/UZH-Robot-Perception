#ifndef UZH_MATLAB_PORT_PADARRAY_H_
#define UZH_MATLAB_PORT_PADARRAY_H_

#include "armadillo"
#include "glog/logging.h"

namespace uzh {

enum BorderType : int { CONSTANT, REPLICATE };

enum PaddingDirection : int { BOTH, POST, PRE };

// TODO(bayes) Extend to arma::Cube. (Easy to implement?)
// TODO(bayes) Implement behaviros with different <direction>.
//@brief Imitate matlab's padarray. Pad matrix.
//@param A Matrix to be padded.
//@param pad_size_x Padding size along x direction, aka. direction along cols.
//@param pad_size_y Padding size along y direction, aka. direction along rows.
//@param border_type Type of making borders.
// REPLICATE - the border values are the replication of the values on the very
// edges.
// CONSTANT  - the border values are all the same with the value provided by the
// user.
//@param border_value The value filled with borders if border_type is CONSTANT.
// By default, 0.0 is filled, i.e. zero-padding is applied.
//@param direction
// BOTH -  Pads before the first element and after the last array element along
// each dimension.
// POST - Pad after the last array element along each dimension.
// PRE  - Pad before the first array element along each dimension.
//@return A_padded Padded matrix.
template <typename T>
arma::Mat<T> /* A_padded */
padarray(const arma::Mat<T>& A, const int pad_size_x, const int pad_size_y,
         const int border_type = uzh::CONSTANT, const T border_value = T(0),
         const int direction = uzh::BOTH) {
  if (A.empty()) LOG(ERROR) << "Empty input.";
  if (pad_size_x < 0 || pad_size_y < 0) LOG(ERROR) << "Invalid input.";

  arma::Mat<T> A_padded(A.n_rows + 2 * pad_size_y, A.n_cols + 2 * pad_size_x,
                        arma::fill::zeros);
  // Number of rows and cols of A_padded.
  const int rows = A_padded.n_rows, cols = A_padded.n_cols;

  // Prepare borders.
  // Left borders.
  arma::Mat<T> left_borders(A.n_rows, pad_size_x, arma::fill::zeros);
  // Right borders.
  arma::Mat<T> right_borders(A.n_rows, pad_size_x, arma::fill::zeros);
  // Top borders.
  arma::Mat<T> top_borders(pad_size_y, A.n_cols, arma::fill::zeros);
  // Bottom borders.
  arma::Mat<T> bottom_borders(pad_size_y, A.n_cols, arma::fill::zeros);

  // Populate border values according to border_type.
  if (border_type == uzh::REPLICATE) {
    // Make borders by replicating values on very edges.
    left_borders = arma::repmat(A.head_cols(1), 1, pad_size_x);
    right_borders = arma::repmat(A.tail_cols(1), 1, pad_size_x);
    top_borders = arma::repmat(A.head_rows(1), pad_size_y, 1);
    bottom_borders = arma::repmat(A.tail_rows(1), pad_size_y, 1);

  } else if (border_type == uzh::CONSTANT) {
    //! By making this fill-in explicit, this function could be easily extended
    //! to take as different border values to be filled in different borders.
    // Fill borders with constant values.
    left_borders.fill(border_value);
    right_borders.fill(border_value);
    top_borders.fill(border_value);
    bottom_borders.fill(border_value);

  } else {
    LOG(ERROR) << "Unsupported border type.";
  }

  // Pad borders.
  if (pad_size_x > 0) {
    A_padded(pad_size_y, 0, arma::size(left_borders)) = left_borders;
    A_padded(pad_size_y, cols - pad_size_x, arma::size(right_borders)) =
        right_borders;
  }
  if (pad_size_y > 0) {
    A_padded(0, pad_size_x, arma::size(top_borders)) = top_borders;
    A_padded(rows - pad_size_y, pad_size_x, arma::size(bottom_borders)) =
        bottom_borders;
  }

  // Fill in the values in the four corners.
  if (pad_size_x > 0 && pad_size_y > 0) {
    const arma::Mat<T> corner_block(pad_size_y, pad_size_x, arma::fill::none);
    if (border_type == uzh::REPLICATE) {
      // Upper left corner block.
      A_padded(0, 0, arma::size(corner_block)).fill(A(0, 0));
      // Upper right corner block.
      A_padded(0, cols - pad_size_x, arma::size(corner_block))
          .fill(A(0, A.n_cols - 1));
      // Bottom left corner block
      A_padded(rows - pad_size_y, 0, arma::size(corner_block))
          .fill(A(A.n_rows - 1, 0));
      // Bottom right corner block
      A_padded(rows - pad_size_y, cols - pad_size_x, arma::size(corner_block))
          .fill(A(A.n_rows - 1, A.n_cols - 1));

    } else {
      // Upper left corner block.
      A_padded(0, 0, arma::size(corner_block)).fill(border_value);
      // Upper right corner block.
      A_padded(0, cols - pad_size_x, arma::size(corner_block))
          .fill(border_value);
      // Bottom left corner block
      A_padded(rows - pad_size_y, 0, arma::size(corner_block))
          .fill(border_value);
      // Bottom right corner block
      A_padded(rows - pad_size_y, cols - pad_size_x, arma::size(corner_block))
          .fill(border_value);
    }
  }

  // Copy A to the central part of A_padded.
  A_padded(pad_size_y, pad_size_x, arma::size(A)) = A;

  return A_padded;
}

}  // namespace uzh

#endif  // UZH_MATLAB_PORT_PADARRAY_H_