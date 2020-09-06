TEST() {
  const arma::umat tmp{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  arma::mat kernel{-1, 0, 1};
  // kernel = kernel.t();
  uzh::conv2(arma::conv_to<arma::mat>::from(tmp), kernel, uzh::FULL)
      .print("full");
  uzh::conv2(arma::conv_to<arma::mat>::from(tmp), kernel, uzh::SAME)
      .print("same");
  uzh::conv2(arma::conv_to<arma::mat>::from(tmp), kernel, uzh::VALID)
      .print("valid");
}