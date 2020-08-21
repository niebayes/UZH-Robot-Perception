#ifndef UZH_COMMON_PLOT_H_
#define UZH_COMMON_PLOT_H_

#include <vector>

#include "Eigen/Core"
#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// #include "unsupported/Eigen/CXX11/Tensor"

//@brief Imitate matlab's meshgrid.
// TODO Generalize this function to arbitrarily accomodating [low, hight] range
// values. E.g. use OpenCV's cv::Range
template <typename Derived>
static void Meshgrid(const int width, const int height,
                     Eigen::MatrixBase<Derived>* X,
                     Eigen::MatrixBase<Derived>* Y) {
  const Eigen::VectorXi x = Eigen::VectorXi::LinSpaced(width, 0, width - 1),
                        y = Eigen::VectorXi::LinSpaced(height, 0, height - 1);
  *X = x.transpose().replicate(height, 1);
  *Y = y.replicate(1, width);
}

//@brief Imitate matlab's meshgrid operating on 3D grid though.
// You could also use eigen's Tensor module which is not supported yet though.
//@ref
// http://eigen.tuxfamily.org/dox-devel/unsupported/group__CXX11__Tensor__Module.html
//@warning If using fixed-size eigen objects, care has to be taken on the
// alignment issues.
//@ref https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html
//? Why "template argument deduction failed"?
// template <typename Derived>
static void Meshgrid3D(const cv::Range& x_range, const cv::Range& y_range,
                       const cv::Range& z_range,
                       std::vector<Eigen::MatrixXi>* X,
                       std::vector<Eigen::MatrixXi>* Y,
                       std::vector<Eigen::MatrixXi>* Z) {
  //  std::vector<typename Eigen::MatrixBase<Derived>>* X,
  //  std::vector<typename Eigen::MatrixBase<Derived>>* Y,
  //  std::vector<typename Eigen::MatrixBase<Derived>>* Z) {
  const int width = x_range.size() + 1, height = y_range.size() + 1,
            depth = z_range.size() + 1;
  const Eigen::VectorXi x = Eigen::VectorXi::LinSpaced(width, x_range.start,
                                                       x_range.end),
                        y = Eigen::VectorXi::LinSpaced(height, y_range.start,
                                                       y_range.end),
                        z = Eigen::VectorXi::LinSpaced(depth, z_range.start,
                                                       z_range.end);
  // const Eigen::MatrixBase<Derived>
  const Eigen::MatrixXi X_any_depth = x.transpose().replicate(height, 1),
                        Y_any_depth = y.replicate(1, width);
  for (int d = 0; d < depth; ++d) {
    X->push_back(X_any_depth);
    Y->push_back(Y_any_depth);
    Eigen::MatrixXi Z_d_depth(height, width);
    Z_d_depth.fill(z(d));
    Z->push_back(Z_d_depth);
  }
}

//@brief Imitate matlab's scatter.
// TODO(bayes) Templatize this function to make the parameter parsing more
// flexible.
static void Scatter(cv::InputOutputArray image,
                    const Eigen::Ref<const Eigen::VectorXi>& x,
                    const Eigen::Ref<const Eigen::VectorXi>& y,
                    const int radius, const cv::Scalar& color,
                    const int thickness = 1) {
  if (x.size() <= 0 || y.size() <= 0) {
    LOG(ERROR) << "Invalid input vectors";
    return;
  } else if (x.size() != y.size()) {
    LOG(ERROR)
        << "YOU_MIXED_DIFFERENT_SIZED_VECTORS";  // Mimic eigen's behavior
    return;
  }

  const int kNumPoints = x.size();
  for (int i = 0; i < kNumPoints; ++i) {
    cv::circle(image, {x(i), y(i)}, radius, color, thickness);
  }
  LOG(INFO) << "Created a scatter plot with " << kNumPoints
            << " points rendered";
}

#endif  // UZH_COMMON_PLOT_H_