#ifndef UZH_SIFT_COMPUTE_DOGS_H_
#define UZH_SIFT_COMPUTE_DOGS_H_

#include "armadillo"

//@brief Compute difference of Gaussians from adjacent images in each octave.
//@param blurred_images Blurred images obtained from ComputeBlurredImages.
//@return Difference of Gaussians of all octaves.
arma::field<arma::cube> ComputeDoGs(
    const arma::field<arma::cube>& blurred_images) {
  const int kNumOctaves = blurred_images.size();
  // DoGs contains DoGs of all octaves.
  arma::field<arma::cube> DoGs(kNumOctaves);

  //! The data layout:
  //! image pyramid ->
  //! num_octaves * octave ->
  //! num_images_per_octave * image ->
  //! one image -> one blurred image ->
  //! two adjacent blurred images -> one DoG
  // Hence the minus one in the size of DoG as below.
  for (int o = 0; o < kNumOctaves; ++o) {
    // DoG contains kNumDoGsPerOctave DoGs in a certain octave.
    arma::cube DoG = arma::zeros<arma::cube>(arma::size(blurred_images(o)) -
                                             arma::size(0, 0, 1));
    const int kNumDoGsPerOctave = DoG.n_slices;
    for (int d = 0; d < kNumDoGsPerOctave; ++d) {
      // arma::abs to ensure every element is not negative.
      DoG.slice(d) = arma::abs(blurred_images(o).slice(d + 1) -
                               blurred_images(o).slice(d));
      // DoG.slice(d)(0, 0, arma::size(8, 8)).print("DoG(d)");
    }

    DoGs(o) = DoG;
  }
  return DoGs;
}

#endif  // UZH_SIFT_COMPUTE_DOGS_H_