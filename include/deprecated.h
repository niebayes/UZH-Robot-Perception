
//! May be deprecated
struct ResultDLT {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix<double, 3, 4> camera_matrix;

  Eigen::Matrix3d calibration_matrix;
  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;

  double reprojection_error;
};

//! May be deprecated
//@brief Wrap the principal procedures of DLT algorithms.
//
// TODO Don't pass matrices by copying
ResultDLT EstimateCameraPoseDLT(Eigen::Matrix2Xd& image_points,
                                Eigen::Matrix3Xd& object_points,
                                const bool do_normalization = true) {
  Eigen::Matrix<double, 3, 4> camera_matrix;

  if (do_normalization) {
    Eigen::Affine2d T;
    Eigen::Affine3d U;
    Normalize(&image_points, &object_points, &T, &U);
    DLT(image_points, object_points, &camera_matrix);
    Denormalize(&camera_matrix, T, U);
  } else {
    DLT(image_points, object_points, &camera_matrix);
  }

  Eigen::Matrix3d calibration_matrix, rotation;
  Eigen::Vector3d translation;
  Decompose(camera_matrix, &calibration_matrix, &rotation, &translation);

  const double reprojection_error =
      ComputeReprojectionError(image_points, object_points, camera_matrix);

  // Store the estimation result
  ResultDLT res;
  res.camera_matrix = camera_matrix;
  res.calibration_matrix = calibration_matrix;
  res.rotation = rotation;
  res.translation = translation;
  res.reprojection_error = reprojection_error;
  return res;
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
static double ComputeReprojectionError(
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

//! Deprecated.
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

//! Deprecated.
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