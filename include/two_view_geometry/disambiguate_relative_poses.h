#ifndef UZH_TWO_VIEW_GEOMETRY_DISAMBIGUATE_RELATIVE_POSES_H_
#define UZH_TWO_VIEW_GEOMETRY_DISAMBIGUATE_RELATIVE_POSES_H_

#include <tuple>

#include "armadillo"
#include "glog/logging.h"
#include "two_view_geometry/triangulation.h"

namespace uzh {

//@brief Disambiguate the four possible combinations of R, t by applying a
// positive-depth test.
//@param Rs The two possible rotations.
//@param u The 3D vector encoding the two possible translations where one of
// them is +u and another is -u.
//@param p1 [3 x n] matrix where each column contains the homogeneous
// coordinates of image points p1 on the left camera.
//@param p2 [3 x n] matrix where each column contains the homogeneous
// coordinates of image points p2 on the right camera.
//@param K1 [3 x 3] calibration matrix for the left camera.
//@param K2 [3 x 3] calibration matrix for the right camera.
//@return R -- The final rotation matrix; t -- The final translation vector. The
// rigid transformation formed by R and t transforms 3D scene points from world
// coordinate to the right camera coordinate. And we further assume the world
// coordinate is identical to the left camera coordinate such that M1 = K1[I|0]
// and M2 = K2[R|t].
//! The point correspondences are used to triangulate the 3D scene points which
//! are then used to voting the combination of R, t which has the highest number
//! of points with positive depth.
//! The calibration matrices in conjunction with the relative poses R and t are
//! used to construct the projection marices to be passed in the
//! LinearTriangulation function.
std::tuple<arma::mat /* R */, arma::vec /* t */> DisambiguateRelativePoses(
    const arma::field<arma::mat>& Rs, const arma::vec& u, const arma::mat& p1,
    const arma::mat& p2, const arma::mat& K1, const arma::mat& K2) {
  if (p1.n_cols != p2.n_cols)
    LOG(ERROR) << "Number of points of p1 and p2 must be consistent.";
  if (p1.n_rows != 3 || p2.n_rows != 3)
    LOG(ERROR) << "Points must be represented as homogeneous coordinates.";
  if (K1.n_rows != 3 || K1.n_cols != 3 || K2.n_rows != 3 || K2.n_cols != 3)
    LOG(ERROR) << "Invalid calibration matrix.";

  // The two possible translations.
  arma::field<arma::vec> ts(2);
  ts(0) = u;
  ts(1) = -u;

  // Voting by counting the number of triangulated points with positive depth.
  arma::mat R_final;
  arma::vec t_final;
  int max_num_points_pdepth = 0;
  // M1 = K1 * [I|0] as we assumed.
  const arma::mat M1 = K1 * arma::eye<arma::mat>(3, 4);
  for (arma::mat R : Rs) {
    for (arma::vec t : ts) {
      // Construct projection matrix M2 from R and t.
      const arma::mat M2 = K2 * arma::join_horiz(R, t);

      // Triangulate 3D scene points.
      //! The triangulated 3D scene points are in the left camera frame as we've
      //! assumed the left camera frame is identical to the world frame.
      const arma::mat P_C_1 = LinearTriangulation(p1, p2, M1, M2);
      // Transform to the right camera frame.
      const arma::mat P_C_2 = arma::join_horiz(R, t) * P_C_1;

      // Counting the number points in front of both cameras.
      const int num_points_pdepth_1 = arma::sum(P_C_1.row(2) > 0);
      const int num_points_pdepth_2 = arma::sum(P_C_2.row(2) > 0);
      const int num_points_pdepth_total =
          num_points_pdepth_1 + num_points_pdepth_2;
      // Keep the combination of R, t with highest number of points in front of
      // both cameras.
      if (num_points_pdepth_total > max_num_points_pdepth) {
        max_num_points_pdepth = num_points_pdepth_total;
        R_final = R;
        t_final = t;
      }
    }
  }

  return {R_final, t_final};
}

}  // namespace uzh

#endif  // UZH_TWO_VIEW_GEOMETRY_DISAMBIGUATE_RELATIVE_POSES_H_
