#ifndef UZH_ALGORITHM_H_
#define UZH_ALGORITHM_H_

#include <algorithm>
#include <iostream>

#include "Eigen/Core"

using namespace std;
using namespace Eigen;

/**对向量进行排序，从大到小
 * vec: 待排序的向量
 * sorted_vec: 排序的结果
 * ind: 排序结果中各个元素在原始向量的位置
 */
void sort_vec(const VectorXd& vec, VectorXd& sorted_vec, VectorXi& ind) {
  ind = VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1);  //[0 1 2 3 ... N-1]
  auto rule = [vec](int i, int j) -> bool {
    return vec(i) > vec(j);
  };  //正则表达式，作为sort的谓词
  std::sort(ind.data(), ind.data() + ind.size(), rule);
  // data成员函数返回VectorXd的第一个元素的指针，类似于begin()
  sorted_vec.resize(vec.size());
  for (int i = 0; i < vec.size(); i++) {
    sorted_vec(i) = vec(ind(i));
  }
}
//测试
int main() {
  VectorXd x(5);
  x << 3, 4, 1, 5, 6;
  VectorXi ind;
  VectorXd sorted_vec;
  sort_vec(x, sorted_vec, ind);
  cout << "原始向量:\n";
  cout << x << endl << endl;
  cout << "排序后:\n";
  cout << sorted_vec << endl << endl;
  cout << "排序后向量各元素对应的原始向量中的位置" << endl;
  cout << ind << endl;

  return 0;
}

//@brief Sort the coefficients of a Eigen::Vector object.
void Sort() {}

#endif  // UZH_ALGORITHM_H_