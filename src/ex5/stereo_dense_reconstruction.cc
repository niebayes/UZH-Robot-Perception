#include <string>
#include <tuple>

#include "armadillo"
#include "google_suite.h"
#include "io.h"
#include "matlab_port.h"
#include "opencv2/opencv.hpp"
#include "stereo.h"
#include "transfer.h"

DEFINE_int32(num_image_pairs, 10,
             "Number of image pairs to be accumulated during the computation "
             "of point clouds. Maximum pairs: 100");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  // Load data
  const std::string kFilePath{"data/ex5/"};
  const std::string kLeftImageName{kFilePath + "left/%06d.png"};
  const std::string kRightImageName{kFilePath + "right/%06d.png"};

  const arma::umat left_img =
      stereo::GetImage(cv::format(kLeftImageName.c_str(), 0));
  const arma::umat right_img =
      stereo::GetImage(cv::format(kRightImageName.c_str(), 0));
  arma::mat K = uzh::LoadArma<double>(kFilePath + "K.txt");
  // Rescale K according to the resize factor of images.
  K.head_rows(2) /= 2.0;
  const arma::mat poses = uzh::LoadArma<double>(kFilePath + "poses.txt");

  // Given settings
  const double kBaseLine = 0.54;
  const int kPatchRadius = 5;
  const double kMinDisparity = 5;
  const double kMaxDisparity = 50;
  const arma::vec2 kXLimits{7, 20};
  const arma::vec2 kYLimits{-6, 10};
  const arma::vec2 kZLimits{-5, 5};

  // Part I: calculate pixel disparity
  // Part II: simple outlier removal, through setting reject_outliers to true.
  // Part III: sub-pixel refinement, through setting refine_subpixel to true.
  const arma::mat disparity_map =
      stereo::GetDisparity(left_img, right_img, kPatchRadius, kMinDisparity,
                           kMaxDisparity, true, true);
  uzh::imagesc(arma::conv_to<arma::umat>::from(disparity_map), true,
               "Disparity map produced by the first pair of images");

  // Part IV: Point cloud triangulation.
  arma::mat point_cloud;
  arma::umat intensities;
  std::tie(point_cloud, intensities) =
      stereo::DisparityToPointCloud(disparity_map, left_img, K, kBaseLine);
  arma::mat33 R_C_frame{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
  arma::mat point_cloud_W = R_C_frame.i() * point_cloud;
  // Visualize the point cloud.
  // FIXME Visualization faulties.
  stereo::VisualizePointCloud(point_cloud_W, intensities);

  // Part V: accumulate point clouds over sequence of pairs of images and write
  // them into a .pcd file to be visualized.
  const bool accumulate_seq = true;
  const int kAccumulatedPairs = FLAGS_num_image_pairs > 0
                                    ? FLAGS_num_image_pairs
                                    : 100;  // Max: 100 image pairs.
  if (accumulate_seq) {
    arma::field<arma::mat> all_point_clouds(kAccumulatedPairs);
    arma::field<arma::umat> all_intensities(kAccumulatedPairs);
    for (int i = 0; i < kAccumulatedPairs; ++i) {
      const arma::umat l_img =
          stereo::GetImage(cv::format(kLeftImageName.c_str(), i));
      const arma::umat r_img =
          stereo::GetImage(cv::format(kRightImageName.c_str(), i));
      const arma::mat disp_map = stereo::GetDisparity(
          l_img, r_img, kPatchRadius, kMinDisparity, kMaxDisparity, true, true);
      // Write disparity map to a image file.
      cv::imwrite(
          cv::format((kFilePath + "disp_map/disp_map_%d.png").c_str(), i),
          uzh::imagesc(arma::conv_to<arma::umat>::from(disp_map), false));

      arma::mat p_C_points;
      arma::umat intens;
      std::tie(p_C_points, intens) =
          stereo::DisparityToPointCloud(disp_map, l_img, K, kBaseLine);
      // FIXME What is the derivation of the convertion?
      // Convert the coordinates from camera coordinates to world coordinates.
      // arma::mat33 R_C_frame{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
      arma::mat p_F_points = R_C_frame.i() * p_C_points;

      // Filter out points out of the given limits, as well as the intensities.
      auto a = p_F_points.row(0) > kXLimits(0);
      arma::uvec filter = arma::find((p_F_points.row(0) > kXLimits(0)) &&
                                     (p_F_points.row(0) < kXLimits(1)) &&
                                     (p_F_points.row(1) > kYLimits(0)) &&
                                     (p_F_points.row(1) < kYLimits(1)) &&
                                     (p_F_points.row(2) > kZLimits(0)) &&
                                     (p_F_points.row(2) < kZLimits(1)));
      p_F_points = p_F_points.cols(filter);
      intens = intens.cols(filter);

      // FIXME What does this do? Difference to the R_C_frame?
      // Tranform from camera frame to world frame.
      const arma::mat T_W_C = arma::reshape(poses.row(i), 4, 3).t();
      // TODO(bayes) Replace the ugly codes.
      const arma::mat T_W_F =
          T_W_C * arma::join_horiz(
                      arma::join_vert(R_C_frame, arma::zeros(1, 3)),
                      arma::join_vert(arma::zeros(3, 1), arma::ones(1, 1)));
      all_point_clouds(i) =
          T_W_F(0, 0, arma::size(3, 3)) * p_F_points +
          arma::repmat(T_W_F.head_rows(3).col(3), 1, p_F_points.n_cols);
      all_intensities(i) = intens;
      LOG(INFO) << "Image pair " << i << " contributes "
                << all_point_clouds(i).n_cols << " points.";
    }
    // Gather all point clouds altogether and save it to a .pcd file.
    stereo::WritePointCloud(kFilePath + "cloud.pcd",
                            uzh::cell2mat<double>(all_point_clouds),
                            uzh::cell2mat<arma::uword>(all_intensities));
  }

  return EXIT_SUCCESS;
}