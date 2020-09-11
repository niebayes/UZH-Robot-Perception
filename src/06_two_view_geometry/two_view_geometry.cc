#include "two_view_geometry.h"

#include <string>
#include <tuple>

#include "arma_traits.h"
#include "armadillo"
#include "google_suite.h"
#include "io.h"
#include "matlab_port.h"
#include "opencv2/opencv.hpp"
#include "stereo/visualize_point_cloud.h"
#include "stereo/write_point_cloud.h"
#include "transfer.h"

int main(int /*argc*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string file_path{"data/06_two_view_geometry/"};

  // Load data
  const arma::mat K = uzh::LoadArma<double>(file_path + "K.txt");
  const arma::mat p1 = uzh::LoadArma<double>(file_path + "matches0001.txt");
  const arma::mat p2 = uzh::LoadArma<double>(file_path + "matches0002.txt");

  // Transform to homogeneous 2D coordinates.
  const arma::mat p1_h = uzh::homogeneous<double>(p1);
  const arma::mat p2_h = uzh::homogeneous<double>(p2);

  // Estimate essential matrix E.
  // Assume K1 = K2 = K.
  const arma::mat E = uzh::EstimateEssentialMatrix(p1_h, p2_h, K, K);

  // Decompose E get Rs and u;
  arma::field<arma::mat> Rs;
  arma::vec u;
  std::tie(Rs, u) = uzh::DecomposeEssentialMatrix(E);

  // Disambiguate combinations of R and t.
  arma::mat R;
  arma::mat t;
  std::tie(R, t) = uzh::DisambiguateRelativePoses(Rs, u, p1_h, p2_h, K, K);

  // Triangulate a point cloud from the views.
  const arma::mat M1 = K * arma::eye<arma::mat>(3, 4);
  const arma::mat M2 = K * arma::join_horiz(R, t);
  arma::mat P;

  const bool use_nonlinear_triangulation = true;
  if (use_nonlinear_triangulation) {
    // Use nonlinear triangulation to get a more accurate P.
    LOG(WARNING) << "Nonlinear triangulation risks to overfitting.";
    P = uzh::NonlinearTriangulation(p1_h, p2_h, M1, M2);
  } else {
    P = uzh::LinearTriangulation(p1_h, p2_h, M1, M2);
  }
  // FIXME P is okay in macOS but bad in Ubuntu.

  // Dehomogenize.
  const arma::mat P_hn = uzh::hnormalized<double>(P);

  // Display the 2D image points.
  cv::Mat img1 = cv::imread(file_path + "images/0001.jpg", cv::IMREAD_COLOR);
  cv::Mat img2 = cv::imread(file_path + "images/0002.jpg", cv::IMREAD_COLOR);
  const arma::uvec p1_x = arma::conv_to<arma::uvec>::from(p1.row(0).t());
  const arma::uvec p1_y = arma::conv_to<arma::uvec>::from(p1.row(1).t());
  const arma::uvec p2_x = arma::conv_to<arma::uvec>::from(p2.row(0).t());
  const arma::uvec p2_y = arma::conv_to<arma::uvec>::from(p2.row(1).t());
  uzh::scatter(img1, p1_x, p1_y, 5, {0, 0, 255}, cv::FILLED);
  uzh::scatter(img2, p2_x, p2_y, 5, {0, 0, 255}, cv::FILLED);
  std::vector<cv::Mat> Mat_vec(2);
  Mat_vec[0] = img1;
  Mat_vec[1] = img2;
  cv::imshow("2D image points", uzh::MakeCanvas(Mat_vec, img1.rows, 1));
  cv::waitKey(0);

  // Visualize the reconstructed 3D scene points.
  const arma::umat intensities =
      arma::randi<arma::umat>(1, P_hn.n_cols, arma::distr_param(0, 255));
  uzh::VisualizePointCloud(P_hn, intensities);
  // Save the points to .pcd file.
  uzh::WritePointCloud(file_path + "points_3d.pcd", P_hn, intensities);

  return EXIT_SUCCESS;
}