#include <string>
#include <vector>

#include "Eigen/Dense"
#include "feature.h"
#include "glog/logging.h"
#include "matlab_port.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"

int main(int /*argv*/, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  const std::string kFilePath = "data/ex3/";
  // Image to show the rendered objects.
  cv::Mat image_show =
      cv::imread(kFilePath + "KITTI/000000.png", cv::IMREAD_COLOR);
  cv::Mat image_show_shi = image_show.clone();
  cv::Mat image = image_show.clone();
  cv::cvtColor(image_show, image, cv::COLOR_BGR2GRAY, 1);

  // Part I: compute response.
  // The shi_tomasi_response is computed as comparison whilst the
  // harris_response is used through out the remainder of this program.
  const int kPatchSize = 9;
  const double kHarrisKappa = 0.08;
  cv::Mat harris_response, shi_tomasi_response;
  HarrisResponse(image, harris_response, kPatchSize, kHarrisKappa);
  ShiTomasiResponse(image, shi_tomasi_response, kPatchSize);
  // Compare the colormaps to see the detail of differences.
  uzh::imagesc(harris_response, true, "Harris response");
  uzh::imagesc(shi_tomasi_response, true, "Shi-Tomasi response");

  // Part II: select keypoints
  const int kNumKeypoints = 200;
  const int kNonMaximumRadius = 8;
  cv::Mat keypoints;
  SelectKeypoints(harris_response, keypoints, kNumKeypoints, kNonMaximumRadius);
  // Superimpose the selected keypoins to the original image.
  Eigen::MatrixXd k;
  cv::cv2eigen(keypoints, k);
  const Eigen::VectorXi x = k.row(0).cast<int>(), y = k.row(1).cast<int>();
  uzh::scatter(image_show, x, y, 4, {0, 0, 255}, cv::FILLED);
  cv::imshow("Harris keypoints", image_show);
  cv::waitKey(0);

  // Show the Shi-Tomasi keypoints for comparison.
  cv::Mat shi_keypoints;
  SelectKeypoints(shi_tomasi_response, shi_keypoints, kNumKeypoints,
                  kNonMaximumRadius);
  Eigen::MatrixXd shi_k;
  cv::cv2eigen(shi_keypoints, shi_k);
  const Eigen::VectorXi shi_x = shi_k.row(0).cast<int>(),
                        shi_y = shi_k.row(1).cast<int>();
  uzh::scatter(image_show_shi, shi_x, shi_y, 4, {0, 255, 0}, cv::FILLED);
  cv::imshow("Shi-Tomasi keypoints", image_show_shi);
  cv::waitKey(0);

  // Part III: describe keypoints
  const int kPatchRadius = 9;
  cv::Mat descriptors;
  DescribeKeypoints(image, keypoints, descriptors, kPatchRadius);

  // Show the top 16 descriptors ranked by strengh of response.
  std::vector<cv::Mat> Mat_vec;
  bool show_descriptors = true;
  for (int i = 0; i < 16; ++i) {
    cv::Mat descriptor = descriptors.col(i);
    //@note When dealing with ROI, the Mat may be not continious and the
    // reshape operation is disabled. To solve this, simply clone itself.
    if (!descriptor.isContinuous()) {
      descriptor = descriptor.clone();
    }
    cv::Mat desc_patch = descriptor.reshape(1, 2 * kPatchRadius + 1);
    Mat_vec.push_back(uzh::imagesc(desc_patch, false));
  }
  cv::Mat top_sixteen_patches;
  if (show_descriptors) {
    // FIXME The images composited by MakeCanvas are not keeping the original
    // orientation.
    top_sixteen_patches = MakeCanvas(Mat_vec, image.rows, 4);
    cv::namedWindow("Top 16 descriptors", cv::WINDOW_NORMAL);
    cv::imshow("Top 16 descriptors", top_sixteen_patches);
    cv::waitKey(0);
  }

  // Part IV: match descriptors
  const double kDistanceRatio = 4;
  cv::Mat match_show =
      cv::imread(kFilePath + "KITTI/000001.png", cv::IMREAD_COLOR);
  cv::Mat query_image;
  cv::cvtColor(match_show, query_image, cv::COLOR_BGR2GRAY, 1);

  cv::Mat query_harris_response;
  HarrisResponse(query_image, query_harris_response, kPatchSize, kHarrisKappa);
  cv::Mat query_keypoints;
  SelectKeypoints(query_harris_response, query_keypoints, kNumKeypoints,
                  kNonMaximumRadius);
  cv::Mat query_descriptors;
  DescribeKeypoints(query_image, query_keypoints, query_descriptors,
                    kPatchRadius);

  cv::Mat matches;
  MatchDescriptors(query_descriptors, descriptors, matches, kDistanceRatio);
  PlotMatches(matches, query_keypoints, keypoints, match_show);
  cv::namedWindow("Matches between the first two frames", cv::WINDOW_AUTOSIZE);
  cv::imshow("Matches between the first two frames", match_show);
  cv::waitKey(0);

  // Part V: match descriptors for all 200 images in the reduced KITTI
  // dataset.
  // Prepare database containers.
  cv::Mat database_kps, database_descs;
  const int kNumImages = 200;
  bool plot_matches = true;
  if (plot_matches) {
    for (int i = 0; i < kNumImages; ++i) {
      cv::Mat img_show =
          cv::imread(cv::format((kFilePath + "KITTI/%06d.png").c_str(), i),
                     cv::IMREAD_COLOR);
      cv::Mat query_img;
      cv::cvtColor(img_show, query_img, cv::COLOR_BGR2GRAY, 1);

      // Prepare query containers.
      cv::Mat query_harris, query_kps, query_descs;
      cv::Mat matches_qd;

      HarrisResponse(query_img, query_harris, kPatchSize, kHarrisKappa);
      SelectKeypoints(query_harris, query_kps, kNumKeypoints,
                      kNonMaximumRadius);
      DescribeKeypoints(query_img, query_kps, query_descs, kPatchRadius);

      // Match query and database after the first iteration.
      if (i >= 1) {
        MatchDescriptors(query_descs, database_descs, matches_qd,
                         kDistanceRatio);
        PlotMatches(matches_qd, query_kps, database_kps, img_show);
        cv::putText(img_show,
                    cv::format("Matches / Totabl: %d / %d",
                               cv::countNonZero(matches_qd) + 1, kNumKeypoints),
                    {50, 30}, cv::FONT_HERSHEY_PLAIN, 2, {0, 0, 255}, 2);
        cv::imshow("Matches", img_show);
        char key = cv::waitKey(10);  // Pause 10 ms.
        if (key == 27)
          break;  // 'ESC' key -> exit.
        else if (key == 32)
          cv::waitKey(0);  // 'Space' key -> pause.
      }

      database_kps = query_kps;
      database_descs = query_descs;
    }
  }

  // -------------------------------------------------------------------
  // Optional: profile the program
  // TODO(bayes) Profile the program.

  return EXIT_SUCCESS;
}
