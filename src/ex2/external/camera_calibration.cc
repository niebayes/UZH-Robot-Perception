// ETHZ Computer Vision Ex1: Camera Calibration
//
// Objective: calibrate your own camera to estimate the intrinsic parameters and
// the distortion coefficients.
//
// Tasks:
//
// 1) Implement the DLT algorithm
//    [K, R, t, error] = runDLT(xy, XYZ)
//
// 2) Implement the Gold Standard algorithm
//    [K, R, t, error] = runGoldStandard(xy, XYZ)
//
// 3) Bonus: Implement the Gold Standard algorithm with radial distortion
// estimation
//    [K, R, t, error] = runGoldStandardRadial(xy, XYZ)

#include <cmath>  // std::sqrt
#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "ceres/ceres.h"
#include "const_iter.h"
#include "google_suite.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//@brief Translate the points so that their centroid, i.e. mean average along
// each dimension, is at the origin (0, 0) for image points and (0, 0, 0) for
// object points; and isotropically scale the points so that their RMS
// (root-mean-squared) distance, aka. Eculidean distance, from the origin is
// 2^(1/2) for image points, i.e. the "average" point has the coordinate (1, 1,
// 1), and 3^(1/2) for object points, i.e. the "average" point has the
// coordinate (1, 1, 1, 1).
//? What does "average" point mean?
//
// Data normalization is an essential part for estimation. It makes the computed
// tranformation invariant to different scale and coordinate origin.
//
//@note When there's a substantial portion of points lie at infinity or very
// close to the camera, the translation technique used in normalization may not
// work very well.
void Normalize(Eigen::Matrix2Xd* image_points, Eigen::Matrix3Xd* object_points,
               Eigen::Affine2d* T, Eigen::Affine3d* U) {
  // First transfer to canonical homogeneous form, i.e. the last entry is 1.
  // Skipped because eigen will automatically promote the points to the
  // homogeneous representation during transformation. So this function only
  // takes as input non-homogeneous points.

  // Compute centroid, aka. the mean average. The negative of that will be
  // applied
  const Eigen::RowVector2d image_centroid = image_points->rowwise().mean();
  const Eigen::RowVector3d object_centroid = object_points->rowwise().mean();

  // Compute scale. The reciprocal of that will be applied.
  const double image_scale =
      std::sqrt((image_points->colwise() - Eigen::Vector2d(image_centroid))
                    .squaredNorm() /
                (2.0 * image_points->cols()));
  const double object_scale =
      std::sqrt((object_points->colwise() - Eigen::Vector3d(object_centroid))
                    .squaredNorm() /
                (3.0 * object_points->cols()));

  // Construct T and U similarity transformation matrices to:
  // translate so that the mean is zero for each dimension
  // scale so that the RMS distance is 2^(1/2) and 3^(1/2) respectively.
  T->setIdentity();
  U->setIdentity();
  *T = (Eigen::Translation2d(Eigen::Vector2d(image_centroid)) *
        Eigen::Scaling(image_scale))
           .inverse();
  *U = (Eigen::Translation3d(Eigen::Vector3d(object_centroid)) *
        Eigen::Scaling(object_scale))
           .inverse();

  // Normalize the points according to the transformation matrices
  *image_points = *T * image_points->colwise().homogeneous();
  *object_points = *U * object_points->colwise().homogeneous();
}

//@brief Estimate the camera matrix using DLT algorithm from the given point
// correspondences.
void DLT(const Eigen::Ref<const Eigen::Matrix2Xd>& image_points,
         const Eigen::Ref<const Eigen::Matrix3Xd>& object_points,
         Eigen::Matrix<double, 3, 4>* camera_matrix) {
  // Construct the coefficient matrix A containing the constraints introduced by
  // the 2D-3D points correspondences
  const int kNumCorrespondences = static_cast<int>(image_points.cols());
  Eigen::MatrixXd A(2 * kNumCorrespondences, 12);
  for (int i = 0; i < kNumCorrespondences; ++i) {
    double x, y, w;
    x = image_points.col(i).x();
    y = image_points.col(i).y();
    w = 1;

    double X, Y, Z, W;
    X = object_points.col(i).x();
    Y = object_points.col(i).y();
    Z = object_points.col(i).z();
    W = 1;

    Eigen::RowVectorXd constraint_xw(12), constraint_yw(12);
    constraint_xw << w * X, w * Y, w * Z, w * W, 0, 0, 0, 0, -x * X, -x * Y,
        -x * Z, -x * W;
    constraint_yw << 0, 0, 0, 0, -w * X, -w * Y, -w * Z, -w * W, y * X, y * Y,
        y * Z, y * W;

    A.row(2 * i) = constraint_xw;
    A.row(2 * i + 1) = constraint_yw;
  }

  // Solve the linear system equations A*P = 0 using SVD A = USV* to get the
  // approximate camera matrix \hat{P} with which the algebraic error is
  // minimized, where \hat{P} is the right null vector of A corresponding to the
  // smallest singular value.
  //
  //@note If the being decomposed matrix has dynamic-size columns, you can
  // compute the thin U and V if you'd like. Eigen does the checking based on
  // the _MatrixType template parameter passed to it. In our case, full or thin,
  // both are okay and we choose full here in spirit of the fixed-size columns
  // here.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(2 * kNumCorrespondences, 12);
  //? the computed V is transposed or not? So the smallest singular vector is
  // the last row or last col?
  // Eigen::VectorXd smallest_singular_vector = svd.matrixV().rightCols(1);
  svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Index smallest_singular_value_index;
  svd.singularValues().minCoeff(&smallest_singular_value_index);
  const Eigen::VectorXd smallest_singular_vector =
      svd.matrixV().col(smallest_singular_value_index);
  // TODO Make the code neater (possibly) using Eigen::Map or (conservative)
  // resizing technique.
  camera_matrix->row(0) =
      Eigen::RowVector4d(smallest_singular_vector.head<4>());
  camera_matrix->row(1) =
      Eigen::RowVector4d(smallest_singular_vector.segment<4>(4));
  camera_matrix->row(2) =
      Eigen::RowVector4d(smallest_singular_vector.tail<4>());

  // Normalize the camera matrix P
  camera_matrix->array() /= camera_matrix->coeff(2, 3);
}

//@brief Denormalize camera matrix using the transformations computed from
// Normalize.
//
// The reason we do this because the un-denormalized camera matrix is computed
// from the normalized points. But the points to be applied by this camera
// matrix are not normalized.
//
//@warning Could not use Eigen::Ref with Eigen::Transform. At this moment,
// Eigen::Ref only supports "Matrix" type. Use plain const reference instead,
// despite the inefficiency and temporary issues.
//
//@note It seems there's no left multiplication operator overloaded between
// plain matrix and transformation object. Convert the transformations to plain
// matrices first and then evaluate the plain matrix product.
void Denormalize(Eigen::Matrix<double, 3, 4>* camera_matrix,
                 const Eigen::Affine2d& T, const Eigen::Affine3d& U) {
  *camera_matrix = T.inverse().matrix() * (*camera_matrix) * U.matrix();
}

//@brief Decompose the camera matrix into the calibration matrix K, and the
// camera pose composed of the rotation matrix R and the translation vector t
// which is also the camera center C.
//
// By construction, P = [M|-MC] = K[R|-RC], where M = KR can be decomposed with
// an RQ-decomposition, and C = t satisfies PC = 0 which means C is the
// (smallest) right null vector of P and can be found with the SVD.
void Decompose(
    const Eigen::Ref<const Eigen::Matrix<double, 3, 4>>& camera_matrix,
    Eigen::Matrix3d* calibration_matrix, Eigen::Matrix3d* rotation,
    Eigen::Vector3d* translation) {
  // Find K and R from M using QR decomposition, where M is the left three cols
  // of P.
  //@note
  //
  // For more about QR:
  //@ref https://en.wikipedia.org/wiki/QR_decomposition
  //@ref https://en.wikipedia.org/wiki/Householder_transformation
  //@ref https://en.wikipedia.org/wiki/Pivot_element
  Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 3, 3>> qr(3, 3);
  qr.compute(camera_matrix.leftCols(3));
  *calibration_matrix = qr.matrixQ();
  *rotation = qr.matrixR();

  // Find t (C) from PC = 0 using SVD on P.
  //
  //@note C is 4d vector in homogeneous representation and t is assumed a 3d
  // vector in non-homogeneous representation for this function. So the
  // hnormalized() function is applied.
  Eigen::JacobiSVD<Eigen::Matrix<double, 3, 4>> svd(3, 4);
  svd.compute(camera_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  //@warning Eigen is heavily dependent on the underlying type. hnormalized() is
  // only callable for Vector type, so you have to in advance use the colwise on
  // the expression resulted from rightCols(1) which is Matrix type.
  // *translation = svd.matrixV().rightCols(1).colwise().hnormalized();
  //
  //@note To avoid the ambiguous introduced by using Eigen::ComputeThinV or
  // Eigen::ComputeFullV, the most unambiguous method is used here: use the
  // index to fetch the singular vector corresponding to the smallest singular
  // value. Hence, we'll always use Eigen::ComputeFullV from now on.
  Eigen::Index smallest_singular_value_index;
  svd.singularValues().minCoeff(&smallest_singular_value_index);
  *translation = svd.matrixV().col(smallest_singular_value_index).hnormalized();
}

//@brief Compute the reprojection error between the measured image points and
// the projections of the measured object points with the given camera matrix.
//
// The reprojection error is given by: E(x, X, P) =  sum_i  |x_i - P X_i|^2.
//
//@warning The reprojection error should always be "averaged", i.e. the RMS of
// the reprojection error is returned.
//
// Templated for use in both multiple points case and single point case.
// const Eigen::Ref<const Eigen::Matrix2Xd>& image_points,
// const Eigen::Ref<const Eigen::Matrix3Xd>& object_points,
template <typename DerivedA, typename DerivedB>
double ComputeReprojectionError(
    const Eigen::MatrixBase<DerivedA>& image_points,
    const Eigen::MatrixBase<DerivedB>& object_points,
    const Eigen::Ref<const Eigen::Matrix<double, 3, 4>>& camera_matrix) {
  const Eigen::Matrix2Xd image_points_hat =
      (camera_matrix * object_points.colwise().homogeneous())
          .colwise()
          .hnormalized();
  //? The error computed this way is the rms reprojection error?
  double reprojection_error = std::sqrt(
      (image_points - image_points_hat).squaredNorm() / image_points.cols());

  return reprojection_error;
}

struct Measurements {
  cv::Mat checkerboard;

  std::vector<cv::Point2d> image_points;
  std::vector<cv::Point3d> object_points;

  int num_correspondences = 0;
};

void mouse_event_handler(int event, int x, int y, int flags, void* param) {
  Measurements* m = static_cast<Measurements*>(param);
  std::vector<cv::Point2d>& image_points = m->image_points;
  std::vector<cv::Point3d>& object_points = m->object_points;
  if (event == cv::EVENT_LBUTTONDOWN) {
    // When pressed left button, capture a image point and store them to
    // image_points. In the meantime, offer user an interface to input the
    // corresponding object point and store it to object_points.
    //? (x, y) or (y, x)? Consistent with Eigen?
    image_points.emplace_back(x, y);
    LOG(INFO) << cv::format("A image point in (%d, %d) is captured", x, y);

    double X, Y, Z;
    std::cout << "Typing ...\n";
    if (std::cin >> X >> Y >> Z) {
      object_points.emplace_back(X, Y, Z);
      cv::circle(m->checkerboard, {x, y}, 8, {0, 255, 0}, 3);
      LOG(INFO) << cv::format("A object point in (%.2f, %.2f, %.2f) is input",
                              X, Y, Z);
      m->num_correspondences++;
    } else {
      LOG(ERROR) << "Error inputing object point coordinates";
      image_points.erase(image_points.end());  //? Never reach this statement?
    }

  } else if (event == cv::EVENT_RBUTTONDOWN) {
    // When pressed right button, return with success if number of point
    // correspondences is suffcient and the points are pairwisely matched, or
    // return with failure otherwise.
    if (m->num_correspondences >= 6 &&
        image_points.size() == object_points.size()) {
      LOG(INFO) << "Captured " << image_points.size()
                << " point correspondences";
      return;
    } else {
      LOG(FATAL) << "Insufficient number of point correspondences. Capture 6 "
                    "at least.";
      return;
    }
  }
}

Measurements GetPoints(const std::string& file_name) {
  if (file_name.empty()) {
    LOG(FATAL) << "Error reading file " << file_name;
  }

  Measurements m;
  m.checkerboard = cv::imread(file_name, cv::IMREAD_COLOR);
  cv::Mat& checkerboard = m.checkerboard;
  cv::String instruction1{"Left click to capture a 2D point"};
  cv::String instruction2{"Then type in the corresponding 3D point"};
  cv::putText(checkerboard, instruction1, {15, 50}, cv::FONT_HERSHEY_PLAIN, 3,
              {0, 255, 0}, 2);
  cv::putText(checkerboard, instruction2, {15, 90}, cv::FONT_HERSHEY_PLAIN, 3,
              {0, 255, 0}, 2);
  cv::putText(checkerboard, "Right click or press 'ESC' to exit", {15, 150},
              cv::FONT_HERSHEY_PLAIN, 3, {0, 0, 255}, 2);
  cv::String winname{"Capture points from clicking"};
  cv::namedWindow(winname);
  cv::setMouseCallback(winname, mouse_event_handler, &m);
  while (m.num_correspondences < 6) {
    cv::imshow(winname, checkerboard);
    char key = cv::waitKey(1);
    if (key == 27) break;  // 'ESC' key -> exit
  }
  cv::destroyWindow(winname);
  if (m.num_correspondences >= 6 &&
      m.image_points.size() == m.object_points.size()) {
    LOG(INFO) << "Successfully read " << m.num_correspondences
              << " point correspondences";
    return m;
  } else {
    LOG(FATAL) << "Failed to get point correspondences";
  }
}

//@brief Initialize Eigen matrices with the read measurements.
void InitializePoints(const Measurements& m, Eigen::Matrix2Xd* image_points,
                      Eigen::Matrix3Xd* object_points) {
  for (int c = 0; c < m.num_correspondences; ++c) {
    image_points->col(c) =
        Eigen::Vector2d(m.image_points[c].x, m.image_points[c].y);
    object_points->col(c) = Eigen::Vector3d(
        m.object_points[c].x, m.object_points[c].y, m.object_points[c].z);
  }
}

struct Result {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix<double, 3, 4> camera_matrix;

  Eigen::Matrix3d calibration_matrix;
  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;

  double reprojection_error;
};

//@brief Wrap the principal procedures of DLT algorithms.
//
// TODO Don't pass matrices by copying
Result RunDLT(Eigen::Matrix2Xd image_points, Eigen::Matrix3Xd object_points) {
  Eigen::Affine2d T;
  Eigen::Affine3d U;
  Normalize(&image_points, &object_points, &T, &U);

  Eigen::Matrix<double, 3, 4> camera_matrix;
  DLT(image_points, object_points, &camera_matrix);

  Denormalize(&camera_matrix, T, U);

  Eigen::Matrix3d calibration_matrix, rotation;
  Eigen::Vector3d translation;
  Decompose(camera_matrix, &calibration_matrix, &rotation, &translation);

  const double reprojection_error =
      ComputeReprojectionError(image_points, object_points, camera_matrix);

  // Store the calibration result
  Result res;
  res.calibration_matrix = calibration_matrix;
  res.rotation = rotation;
  res.translation = translation;
  res.reprojection_error = reprojection_error;
  return res;
}

struct ReprojectionError {
  ReprojectionError(const cv::Point2d& observed, const cv::Point3d& object)
      : observed_(observed), object_(object) {}

  template <typename T>
  bool operator()(const T* const camera_matrix, T* residual) {
    // Reassemble camera matrix
    const Eigen::Matrix<T, 3, 4> camera_matrix_(camera_matrix);

    // Compute reprojection error
    const Eigen::Matrix<T, 2, 1> image_point(T(observed_.x), T(observed_.y));
    const Eigen::Matrix<T, 3, 1> object_point(T(object_.x), T(object_.y),
                                              T(object_.z));

    //@warning When interfacing ceres autodifferentiation, care has to be taken
    // on the external functions that are not templated. Because ceres will use
    // Jet type in addition to the other general type, e.g. double.
    //
    //@ref http://ceres-solver.org/interfacing_with_autodiff.html
    residual[0] =
        ComputeReprojectionError(image_point, object_point, camera_matrix_);
    return true;
  }

  static ceres::CostFunction* Create(const cv::Point2d& observed,
                                     const cv::Point3d& object) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 1, 12>(
        new ReprojectionError(observed, object)));
  }

 private:
  const cv::Point2d observed_;
  const cv::Point3d object_;
};

Result RunGoldStandard(Eigen::Matrix2Xd image_points,
                       Eigen::Matrix3Xd object_points) {}

enum DistortionCoefficientsIndex : int { INDEX_RADIAL_K1, INDEX_RADIAL_K2 };

struct ReprojectionErrorRadial {
  ReprojectionErrorRadial(const cv::Point2d& observed,
                          const cv::Point3d& object)
      : observed_(observed), object_(object) {}

  //@brief Apply simple radial distortion to normalized image point in camera
  // coordinate system to get distorted image point in pixel coordinate system
  // under the distortion factor L = 1 + k_1 * r^2 + k_2 * r^4.
  template <typename T>
  static void ApplySimpleRadialDistortion(const T& k1, const T& k2,
                                          const T& normalized_x,
                                          const T& normalized_y, T* distorted_x,
                                          T* distorted_y) {
    const T r2 = normalized_x * normalized_x + normalized_y * normalized_y;
    const T r4 = r2 * r2;
    const T distort_factor = 1.0 + k1 * r2 + k2 * r4;

    *distorted_x = distort_factor * normalized_x;
    *distorted_y = distort_factor * normalized_y;
  }

  template <typename T>
  bool operator()(const T* const camera_matrix,
                  const T* const distortion_coefficients, T* residual) {
    const Eigen::Matrix<T, 3, 4> camera_matrix_(camera_matrix);

    // Unpack the distortion coefficients according to the indices
    T k1 = distortion_coefficients[INDEX_RADIAL_K1];
    T k2 = distortion_coefficients[INDEX_RADIAL_K2];

    // Apply radial distortion
    T normalized_x = observed_.x, normalized_y = observed_.y;
    T distorted_x, distorted_y;
    ApplySimpleRadialDistortion(k1, k2, normalized_x, normalized_y,
                                &distorted_x, &distorted_y);

    // Compute reprojection error
    const Eigen::Matrix<T, 2, 1> image_point(distorted_x, distorted_y);
    const Eigen::Matrix<T, 3, 1> object_point(object_.x, object_.y, object_.z);
    residual[0] =
        ComputeReprojectionError(image_point, object_point, camera_matrix_);
    return true;
  }

  template <typename T>
  static ceres::CostFunction* Create(const cv::Point2d& observed,
                                     const cv::Point3d& object) {
    return (new ceres::AutoDiffCostFunction<ReprojectionErrorRadial, 1, 12, 2>(
        new ReprojectionErrorRadial(observed, object)));
  }

 private:
  const cv::Point2d observed_;
  const cv::Point3d object_;
};

Result RunGoldStandardRadial(Eigen::Matrix2Xd image_points,
                             Eigen::Matrix3Xd object_points) {}

//@brief Visualize the reprojected points of all checkerboard corners on the
// calibration object with the computed camera matrix.
void Visualize(const Eigen::Ref<const Eigen::Matrix2Xd>& image_points,
               const Eigen::Ref<const Eigen::Matrix3Xd>& object_points,
               const Eigen::Ref<Eigen::Matrix<double, 3, 4>>& camera_matrix,
               const std::string& file_name) {
  cv::Mat image = cv::imread(file_name, cv::IMREAD_COLOR);
  const Eigen::Matrix2Xd image_points_hat =
      (camera_matrix * object_points.colwise().homogeneous())
          .colwise()
          .hnormalized();
  std::cout << image_points << '\n' << '\n';
  std::cout << image_points_hat << '\n';
  const int kNumPoints = static_cast<int>(image_points.cols());
  for (int i = 0; i < kNumPoints; ++i) {
    int x = static_cast<int>(image_points.col(i).x()),
        y = static_cast<int>(image_points.col(i).y());
    int x_hat = static_cast<int>(image_points_hat.col(i).x()),
        y_hat = static_cast<int>(image_points_hat.col(i).y());
    cv::circle(image, {x, y}, 8, {0, 255, 0}, 3);
    cv::circle(image, {x_hat, y_hat}, 8, {0, 0, 255}, 3);
  }
  cv::String winname{"Rreprojected points vs. Observations"};
  cv::imshow(winname, image);
  cv::waitKey(0);
}

//@brief Undistort a single point according to the estimated distortion
// coefficients.
void UnDistortPoint() {}

//@brief Warp the whole image according to the estimated distortion
// coefficients.
void WarpImage() {}

DEFINE_string(input, "", "file path for calibration rig");
DEFINE_int32(method, -1,
             "method used in the camera calibration; Options: 0 -> DLT, 1 -> "
             "GOLD, 2 -> GOLD_RADIAL");

enum class Methods : int { DLT, GOLD_STANDARD, GOLD_STANDARD_RADIAL };

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  // Test data:
  // Eigen::Matrix2Xd image_points(2, 6);
  // image_points << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  // Eigen::Matrix3Xd object_points(3, 6);
  // object_points << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  //     18;

  const std::string& file_name =
      FLAGS_input.empty() ? "data/ex1/images/checkerboard.jpg" : FLAGS_input;

  Measurements m = GetPoints(file_name);
  Eigen::Matrix2Xd image_points(2, m.num_correspondences);
  Eigen::Matrix3Xd object_points(3, m.num_correspondences);
  InitializePoints(m, &image_points, &object_points);
  Result res;

  switch (FLAGS_method) {
    case static_cast<int>(Methods::DLT):
      // DLT: direct linear transform
      //
      // 1) Normalize data points
      //
      // 2) Compute DLT
      //
      // 3) Denormalize camera matrix
      //
      // 4) Decompose camera matrix to K, R, t
      //
      // 5) Compute reprojection error
      //
      // 6) Visualize the reprojected points
      LOG(INFO) << "Running DLT";
      res = RunDLT(image_points, object_points);
      break;
    case static_cast<int>(Methods::GOLD_STANDARD):
      LOG(INFO) << "Running GOLD_STANDARD";
      res = RunGoldStandard(image_points, object_points);
      break;
    case static_cast<int>(Methods::GOLD_STANDARD_RADIAL):
      LOG(INFO) << "Running GOLD_STANDARD_RADIAL";
      res = RunGoldStandardRadial(image_points, object_points);
    default:
      LOG(ERROR) << "Please pass the method flag. Usage: --method=0, 1, 2";
      break;
  }
  std::cout << "Reprojection error: " << res.reprojection_error << '\n';
  Visualize(image_points, object_points, res.camera_matrix, file_name);

  return EXIT_SUCCESS;
}
