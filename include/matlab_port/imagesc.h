#ifndef UZH_MATLAB_PORT_IMAGESC_H_
#define UZH_MATLAB_PORT_IMAGESC_H_

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//@brief Imitate matlab's imagesc. Apply colormap to the input image to display
// image with scaled colors.
void ImageSC(const cv::Mat& image,
             const std::string& winname = "Colormapped image",
             const int colormap = cv::COLORMAP_PARULA, const int delay = 0) {
  cv::Mat img;
  cv::normalize(image, img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::applyColorMap(img, img, colormap);
  cv::namedWindow(winname, cv::WINDOW_NORMAL);
  cv::imshow(winname, img);
  cv::waitKey(delay);
}
#endif  // UZH_MATLAB_PORT_IMAGESC_H_