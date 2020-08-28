#ifndef UZH_SIFT_EXTRACT_KEYPOINTS_H_
#define UZH_SIFT_EXTRACT_KEYPOINTS_H_

#include "armadillo"

//@brief Extract keypoints from the maximums both in scale and space.
//@param DoGs Difference of Gaussians computed from ComputeDoGs.
//@param keypoints_threshold Below which the points are suppressed before
// selection of keypoitns to attenuate effect of noise.
//@return Coordinate matrices for all octaves where keypoints are stored column
// by columns. The coordinates are 3D vectors where the first two dimension
// correspond to space and the last one corresponds to scale.
arma::field<arma::umat> ExtractKeypoints(const arma::field<arma::cube>& DoGs,
                                         const double keypoints_threshold) {
  const int kNumOctaves = DoGs.size();
  arma::field<arma::umat> keypoints(kNumOctaves);

  // For each octave, locate the maximums both in scale and space from the DoGs
  // in this octave.
  for (int o = 0; o < kNumOctaves; ++o) {
    // Copy the DoG
    arma::cube DoG = DoGs(o);

    // Locate the maximums
    //! A neat trick: use dilation in place of moving max filter.
    //@ref
    // https://en.wikipedia.org/wiki/Dilation_(morphology)#Flat_structuring_functions

    //! Here, by making use of armadillo's subview, we can achieve this in an
    //! efficient manner.
    // TODO(bayes) Modularize the codes below, wrapping into the imdilate
    // function.

    // First pre-pad the cube along each dimension to avoid boundary issues.
    const arma::cube kMaxFilter = arma::ones<arma::cube>(3, 3, 3);
    // TODO(bayes) Modularize the codes below, wrappint into the padarray
    // function.
    const int kPadSize = static_cast<int>(std::floor(
        kMaxFilter.n_rows / 2.0));  // Padding size equals to the
                                    // radius of the moving max filter.
    arma::cube padded_DoG = arma::zeros<arma::cube>(
        arma::size(DoG) + arma::size(kPadSize * 2, kPadSize * 2, kPadSize * 2));
    // Copy the elements from DoG to the padded_DoG
    padded_DoG(1, 1, 1, arma::size(DoG)) = DoG;
    // Apply moving max filter.
    arma::cube DoG_max = arma::zeros<arma::cube>(arma::size(DoG));
    for (int slice = 0; slice < DoG_max.n_slices; ++slice) {
      for (int row = 0; row < DoG_max.n_rows; ++row) {
        for (int col = 0; col < DoG_max.n_cols; ++col) {
          DoG_max(row, col, slice) =
              padded_DoG(row, col, slice, arma::size(kMaxFilter)).max();
        }
      }
    }

    // Filter out all points but the ones survived in the max filtering which
    // are then be thresholded.
    //! This is equivalent to doing a non-maximum suppression around the
    //! keypoints selected as above followed by a thresholding.
    arma::ucube is_kept_kpts =
        ((DoG == DoG_max) && (DoG >= keypoints_threshold));

    // Discard the keypoints found in the lowest and highest layers of the
    // current DoG because the two layers are involved with padded borders.
    is_kept_kpts.head_slices(1).zeros();
    is_kept_kpts.tail_slices(1).zeros();

    // Obtain the corresponding 3D coordinates of each putative keypoints.
    arma::umat coordinates_3d =
        arma::ind2sub(arma::size(is_kept_kpts), arma::find(is_kept_kpts));

    // Store them into the keypoints.
    keypoints(o) = coordinates_3d;
  }

  return keypoints;
}

#endif  // UZH_SIFT_EXTRACT_KEYPOINTS_H_