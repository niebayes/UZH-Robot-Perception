#ifndef UZH_COMMON_NORMALIZE_H_
#define UZH_COMMON_NORMALIZE_H_

#include <cmath>

#include "Eigen/Core"
#include "Eigen/Geometry"

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
#endif  // UZH_COMMON_NORMALIZE_H_