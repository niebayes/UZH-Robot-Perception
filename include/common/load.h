#ifndef UZH_COMMON_LOAD_H_
#define UZH_COMMON_LOAD_H_

#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"

template <typename T>
struct Data {
  int rows;
  int cols;
  std::vector<T> data;
};

//@brief Load everything into a std::vector object.
template <typename T>
static Data<T> Load(const std::string& file_name,
                    const std::optional<const char>& delimiter = std::nullopt) {
  std::vector<T> data;
  int rows = 0, cols = 0;

  std::ifstream fin(file_name, std::ios::in);
  if (fin.is_open()) {
    std::string line;
    while (std::getline(fin, line)) {
      std::istringstream iss_outer(line);
      if (iss_outer.good()) {
        T x;
        std::string entry;
        if (delimiter) {
          while (std::getline(iss_outer, entry, delimiter.value())) {
            std::istringstream iss_inner(entry);
            if (iss_inner.good() && iss_inner >> x) {
              data.push_back(x);
              cols++;
            }
          }
        } else {
          while (std::getline(iss_outer, entry, ' ')) {
            std::istringstream iss_inner(entry);
            if (iss_inner.good() && iss_inner >> x) {
              data.push_back(x);
              cols++;
            }
          }
        }
        rows++;
      } else {
        LOG(ERROR) << "Error loading file " << file_name;
      }
    }
    fin.close();
  } else {
    LOG(ERROR) << "Error opening file " << file_name;
  }
  LOG(INFO) << cv::format("Loaded (rows %d x cols %d) data from file %s", rows,
                          cols, file_name);
  Data<T> d;
  d.rows = rows;
  d.cols = cols;
  d.data = data;
  return d;
}

// @brief Load 6dof poses from text file and store them into a 2d vector.
static std::vector<std::vector<double>> LoadPoses(
    const std::string& file_name) {
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
static std::vector<double> LoadK(const std::string& file_name) {
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
template <typename Derived>
static void LoadK(const std::string& file_name,
                  Eigen::MatrixBase<Derived>* calibration_matrix) {
  Eigen::Matrix3d K;

  std::ifstream fin(file_name, std::ios::in);
  if (fin.is_open()) {
    fin.close();
  }
}

//@brief Load lens distortion coefficients D and store them into a vector.
static std::vector<double> LoadD(const std::string& file_name) {
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
//@warning The coordinates are comma separated.
// FIXME Reduce the number of generated temporary objects
static void LoadObjectPoints(const std::string& file_name,
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

//@brief Load observations from the file where each line stores 12 2D
// observations corresponding to 12 3D reference points.
static void LoadObservations(const std::string& file_name,
                             Eigen::Matrix2Xd* observations) {
  //
}

#endif  // UZH_COMMON_LOAD_H_