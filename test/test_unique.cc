TEST() {
  Eigen::VectorXd A(9);
  A << 1, 2, 3, -1, -2, 0, 0, 0, 4;
  Eigen::MatrixXd C;
  std::vector<int> ia, ic;
  std::tie(C, ia, ic) = Unique(A);
  std::cout << C << '\n';
  for (auto e : ia) std::cout << " " << e;
  std::cout << '\n';
  for (auto e : ic) std::cout << " " << e;
  std::cout << '\n'
}