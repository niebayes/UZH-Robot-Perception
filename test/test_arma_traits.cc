TEST() {
  arma::vec m(0, arma::fill::randn);
  m.print("m");
  uzh::homogeneous<double>(m).print("h_m");
  uzh::hnormalized<double>(m).print("hn_m");
}