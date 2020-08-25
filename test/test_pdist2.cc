#include "Eigen/Core"
#include "common/type.h"
#include "features.h"
#include "gtest/gtest.h"

TEST() {
  Eigen::MatrixXd X(3, 2), Y(3, 2);
  X << 1, 2, 3, 4, 5, 6;
  Y << 1, 2, 4, 5, 0, 0;
  Eigen::MatrixXd D;
  Eigen::MatrixXi I;
  uzh::pdist2(X.transpose(), Y.transpose(), &D, EUCLIDEAN, &I, SMALLEST_FIRST,
              3);
  std::cout << D << '\n' << '\n';
  std::cout << I << '\n';
}