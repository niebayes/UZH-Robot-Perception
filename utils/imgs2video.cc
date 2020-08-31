#include <iostream>

#include "google_suite.h"
#include "opencv2/opencv.hpp"

DEFINE_string(n, "", "Pattern of the image names. For example, ./data/imgs_%d");
DEFINE_int32(k, 0, "Number of images.");
DEFINE_int32(i, 0,
             "Start id of images. For example, imgs_00.png. Default is 0.");
DEFINE_int32(fps, 30, "Frame per second. Default is 30.");
DEFINE_string(o, "", "Name of the output video, including path.");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  if (argc < 5) {
    std::cout << "Usage: imgs2video -n=<image_file_pattern> "
                 "-k=<number_of_images> -i=<start_id_of_images> "
                 "-fps=<frame_per_second> -o=<output_video_name>"
              << '\n';
    return EXIT_FAILURE;
  }
  if (FLAGS_n.empty()) {
    LOG(FATAL) << "Invalid image name.";
  }

  const std::string& kFileName = FLAGS_n;
  const int kNumImages = FLAGS_k;
  int id = FLAGS_i;
  const int kFps = FLAGS_fps;
  const std::string& kVideoName = FLAGS_o;

  cv::Mat sample_image = cv::imread(cv::format(kFileName.c_str(), id),
                                    cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  const cv::Size frame_size = sample_image.size();
  cv::VideoWriter video(kVideoName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                        kFps, frame_size, true);
  if (video.isOpened()) {
    for (; id < kNumImages; ++id) {
      cv::Mat frame = cv::imread(cv::format(kFileName.c_str(), id),
                                 cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
      if (frame.empty()) {
        LOG(INFO) << "Wrote " << id << " images to the video";
        break;
      }
      video << frame;
    }
    video.release();
    if (id == kNumImages) {
      LOG(INFO) << "Successfully wrote " << kNumImages
                << " images into the video file: " << kVideoName;
    } else {
      LOG(ERROR) << "Internal error occurs.";
    }
  } else {
    LOG(FATAL) << "Unable to create video file: " << kVideoName;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}