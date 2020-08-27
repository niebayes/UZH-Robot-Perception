TEST() {
  arma::mat vals{-132.1687, -122.8049, -106.4993, -77.6149,
                 -111.6225, -109.2748, -106.0616, -104.9305,
                 -53.8097,  -81.5045,  -87.4018,  -89.2609,
                 -17.9929,  -55.5088,  -75.4922,  -90.6530},
      weights{0.0028, 0.0023, 0.0012, 0.0004, 0.0020, 0.0028, 0.0019, 0.0006,
              0.0015, 0.0026, 0.0021, 0.0007, 0.0022, 0.0023, 0.0017, 0.0006};
  // vals.load(kFilePath + "vals.csv", arma::file_type::csv_ascii, true);
  // weights.load(kFilePath + "weights.csv", arma::file_type::csv_ascii, true);
  arma::vec v = arma::vectorise(vals);
  arma::vec w = arma::vectorise(weights);
  arma::vec edges = arma::linspace<arma::vec>(-180, 180, 9);
  v.print();
  w.print();
  edges.print();
  std::cout << "sorted" << edges.is_sorted() << '\n';
  arma::vec h = uzh::weightedhist(v, w, edges);
  h.print()
}