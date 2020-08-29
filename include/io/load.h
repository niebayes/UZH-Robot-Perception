#ifndef UZH_COMMON_LOAD_H_
#define UZH_COMMON_LOAD_H_

#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "armadillo"
#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"
#include "armadillo"

namespace uzh {

//@brief Load the data file utilizing armadillo library.
//! Recommended load function.
//@ref https://stackoverflow.com/a/39146048/14007680
//@ref http://arma.sourceforge.net
template <typename M>
M armaLoad(const std::string& file_name) {
  arma::mat arma_mat;
  arma_mat.load(file_name, arma::file_type::auto_detect, true);
  return Eigen::Map<const M>(arma_mat.memptr(), arma_mat.n_rows,
                             arma_mat.n_cols);
}

template <typename T> 
arma::Mat<T> LoadArma(const std::string& file_name) {
  arma::Mat<T> mat;
  mat.load(file_name, arma::file_type::auto_detect, true);
  return mat;
}

}

//@brief Generic load function.
//@template param M, complete type of the returned matrix, e.g. Eigen::MatrixXd
template <typename M>
M Load(const std::string& file_name,
       const std::optional<const char>& entry_delimiter = std::nullopt) {
  std::vector<typename M::Scalar> data;
  int rows = 0;

  char delimiter = entry_delimiter ? entry_delimiter.value() : ' ';

  std::ifstream fin(file_name, std::ios::in);
  if (fin.is_open()) {
    std::string line;
    while (std::getline(fin, line)) {
      std::istringstream line_stream(line);
      std::string entry;
      while (std::getline(line_stream, entry, delimiter)) {
        data.push_back(static_cast<typename M::Scalar>(std::stod(entry)));
      }
      ++rows;
    }
    fin.close();
  } else {
    LOG(ERROR) << "Error opening file " << file_name;
  }
  //! If simply return Eigen::Map<const M>(...), the data order is not expected.
  return Eigen::Map<
      const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime,
                          M::ColsAtCompileTime, Eigen::RowMajor>>(
      data.data(), rows, data.size() / rows);
}

// @brief Load 6dof poses from text file and store them into a 2d vector.
//! Deprecated
std::vector<std::vector<double>> LoadPoses(const std::string& file_name) {
  std::vector<std::vector<double>> poses;

  std::ifstream fin(file_name, std::ios::in);
  if (fin.is_open()) {
    std::string pose;
    while (std::getline(fin, pose)) {
      double w_x, w_y, w_z, t_x, t_y, t_z;
      std::istringstream iss(pose);
      if (iss.good() && iss >> w_x >> w_y >> w_z >> t_x >> t_y >> t_z) {
        poses.push_back(std::vector<double>{w_x, w_y, w_z, t_x, t_y, t_z});
      }
    }
    fin.close();
    LOG(INFO) << "Loaded " << poses.size() << " poses";
  } else {
    LOG(ERROR) << "Fail loading file " << file_name;
  }
  return poses;
}

//@brief Load camera calibration matrix K and store it into a vector.
//! Deprecated
std::vector<double> LoadK(const std::string& file_name) {
  std::vector<double> K;

  std::ifstream fin(file_name, std::ios::in);
  if (fin.is_open()) {
    std::string row;
    while (std::getline(fin, row)) {
      double coeff_1, coeff_2, coeff_3;
      std::istringstream iss(row);
      if (iss.good() && iss >> coeff_1 >> coeff_2 >> coeff_3) {
        K.push_back(coeff_1);
        K.push_back(coeff_2);
        K.push_back(coeff_3);
      }
    }
    fin.close();
  } else {
    LOG(ERROR) << "Fail loading file " << file_name;
  }
  return K;
}

//@brief Overloaded version of LoadK.
// Directly store the data into the eigen matrix object.
//! Deprecated
template <typename Derived>
void LoadK(const std::string& file_name,
           Eigen::MatrixBase<Derived>* calibration_matrix) {
  Eigen::Matrix3d K;

  std::ifstream fin(file_name, std::ios::in);
  if (fin.is_open()) {
    fin.close();
  }
}

//@brief Load lens distortion coefficients D and store them into a vector.
//! Deprecated
std::vector<double> LoadD(const std::string& file_name) {
  std::vector<double> D;

  std::ifstream fin(file_name, std::ios::in);
  if (fin.is_open()) {
    std::string coeffs;
    if (std::getline(fin, coeffs)) {
      double k1, k2;
      std::istringstream iss(coeffs);
      if (iss.good() && iss >> k1 >> k2) {
        D.push_back(k1);
        D.push_back(k2);
      }
    }
    fin.close();
  } else {
    LOG(ERROR) << "Fail loading file " << file_name;
  }
  return D;
}

//@brief Load 3D scene points from the file where each line stores the X, Y, Z
// coordinates of a reference 3D scene point.
//! Deprecated
//@warning The coordinates are comma separated.
// FIXME Reduce the number of generated temporary objects
void LoadObjectPoints(const std::string& file_name,
                      Eigen::Matrix3Xd* object_points) {
  std::vector<double> points_3d;
  int num_object_points = 0;

  std::ifstream fin(file_name, std::ios::in);
  if (fin.is_open()) {
    std::string line;
    while (std::getline(fin, line)) {
      std::istringstream point(line);
      if (point.good()) {
        std::string coordinate;
        while (std::getline(point, coordinate, ',')) {
          std::istringstream iss(coordinate);
          double X;
          if (iss.good() && iss >> X) {
            points_3d.push_back(X);
          }
        }
        num_object_points++;
      } else {
        LOG(ERROR) << "Error loading file " << file_name;
        break;
      }
    }
    fin.close();
  } else {
    LOG(ERROR) << "Error opening file " << file_name;
  }
  LOG(INFO) << "Loaded " << num_object_points << " 3D scene points";
  cv::Mat_<double> cv_mat(points_3d);
  cv_mat = cv_mat.reshape(1, num_object_points).t();
  Eigen::Matrix3Xd eigen_mat(3, num_object_points);
  cv::cv2eigen(cv_mat, eigen_mat);
  *object_points = eigen_mat;
}

#endif  // UZH_COMMON_LOAD_H_