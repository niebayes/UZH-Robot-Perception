#ifndef UZH_FEATURE_SUBPLOT_H_
#define UZH_FEATURE_SUBPLOT_H_

#include <cmath>  // std::floor, std::ceil
#include <cstdarg>
#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

//! Recommended one!
//@ref https://stackoverflow.com/a/22858549/14007680
/**
 * @brief MakeCanvas Makes composite image from the given images
 * @param Mat_vec Vector of Images.
 * @param window_height The height of the new composite image to be formed.
 * @param n_rows Number of rows of images. (Number of columns will be calculated
 *              depending on the value of total number of images).
 * @return new composite image.
 */
cv::Mat MakeCanvas(std::vector<cv::Mat>& Mat_vec, int window_height,
                   int n_rows) {
  int N = Mat_vec.size();
  n_rows = n_rows > N ? N : n_rows;
  int edgeThickness = 10;
  int imagesPerRow = std::ceil(double(N) / n_rows);
  int resizeHeight =
      std::floor(2.0 *
                 ((std::floor(double(window_height - edgeThickness) / n_rows)) /
                  2.0)) -
      edgeThickness;
  int maxRowLength = 0;

  std::vector<int> resizeWidth;
  for (int i = 0; i < N;) {
    int thisRowLen = 0;
    for (int k = 0; k < imagesPerRow; k++) {
      double aspectRatio = double(Mat_vec[i].cols) / Mat_vec[i].rows;
      int temp = int(std::ceil(resizeHeight * aspectRatio));
      resizeWidth.push_back(temp);
      thisRowLen += temp;
      if (++i == N) break;
    }
    if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
      maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
    }
  }
  int windowWidth = maxRowLength;
  cv::Mat canvasImage(window_height, windowWidth, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int k = 0, i = 0; i < n_rows; i++) {
    int y = i * resizeHeight + (i + 1) * edgeThickness;
    int x_end = edgeThickness;
    for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
      int x = x_end;
      cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
      cv::Size s = canvasImage(roi).size();
      // change the number of channels to three
      cv::Mat target_ROI(s, CV_8UC3);
      if (Mat_vec[k].channels() != canvasImage.channels()) {
        if (Mat_vec[k].channels() == 1) {
          cv::cvtColor(Mat_vec[k], target_ROI, cv::COLOR_GRAY2BGR);
        }
      } else {
        Mat_vec[k].copyTo(target_ROI);
      }
      cv::resize(target_ROI, target_ROI, s);
      if (target_ROI.type() != canvasImage.type()) {
        target_ROI.convertTo(target_ROI, canvasImage.type());
      }
      target_ROI.copyTo(canvasImage(roi));
      x_end += resizeWidth[k] + edgeThickness;
    }
  }
  return canvasImage;
}

#endif  // UZH_FEATURE_SUBPLOT_H_