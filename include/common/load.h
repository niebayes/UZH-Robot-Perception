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
class Data {
 public:
  Data() = default;

  // TODO Overload or modify this function to directly return a dynamic
  // Eigen::Matrix object whose Scalar, rows, and cols are all dynamic.
  cv::Mat_<T> matrix() const {
    cv::Mat_<T> cv_mat(data);
    cv_mat = cv_mat.reshape(1, rows);
    return cv_mat;
  }

 public:
  int rows;
  int cols;
  int num_entries;
  std::vector<T> data;
};

// TODO Add file extensions provided by the armadillo library.
//@ref https://stackoverflow.com/a/39146048/14007680
//@ref http://arma.sourceforge.net/

//@brief Generic load function.
//@template param M, complete type of the returned matrix, e.g. Eigen::MatrixXd
template <typename M>
static M Load(const std::string& file_name,
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
  }
  return Eigen::Map<Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime,
                                  M::ColsAtCompileTime, Eigen::RowMajor>>(
      data.data(), rows, data.size() / rows);
}

//@brief Load everything into a std::vector object.
// template <typename T>
// static Data<T> Load(
//     const std::string& file_name,
//     const std::optional<const char>& entry_delimiter = std::nullopt) {
//   // Returned Data
//   int rows = 0, cols = 0;
//   int num_entries = 0;
//   std::vector<T> data;

//   // Entry delimiter inside each line of the text file
//   char delimiter = entry_delimiter ? entry_delimiter.value() : ' ';

//   std::ifstream fin(file_name, std::ios::in);
//   if (fin.is_open()) {
//     std::string line;
//     while (std::getline(fin, line)) {
//       std::istringstream iss_outer(line);
//       if (iss_outer.good()) {
//         T x;
//         std::string entry;
//         while (!iss_outer.eof()) {
//           // std::istringstream iss_inner(entry);
//           // if (iss_inner.good() && iss_inner >> x) {
//           if (iss_outer >> x) {
//             data.push_back(x);
//             num_entries++;
//           }
//         }
//         rows++;
//       } else {
//         LOG(ERROR) << "Error loading file " << file_name;
//       }
//     }
//     fin.close();
//   } else {
//     LOG(ERROR) << "Error opening file " << file_name;
//   }
//   Data<T> d;
//   d.rows = rows;
//   d.cols = rows > 0 ? (num_entries / rows) : 0;
//   d.num_entries = num_entries;
//   d.data = data;
//   LOG(INFO) << cv::format("Loaded %d (rows %d x cols %d) data from file %s",
//                           d.num_entries, d.rows, d.cols, file_name.c_str());
//   return d;
// }

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