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
 * @param vecMat Vector of Images.
 * @param windowHeight The height of the new composite image to be formed.
 * @param nRows Number of rows of images. (Number of columns will be calculated
 *              depending on the value of total number of images).
 * @return new composite image.
 */
cv::Mat MakeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows) {
  int N = vecMat.size();
  nRows = nRows > N ? N : nRows;
  int edgeThickness = 10;
  int imagesPerRow = std::ceil(double(N) / nRows);
  int resizeHeight =
      std::floor(
          2.0 *
          ((std::floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) -
      edgeThickness;
  int maxRowLength = 0;

  std::vector<int> resizeWidth;
  for (int i = 0; i < N;) {
    int thisRowLen = 0;
    for (int k = 0; k < imagesPerRow; k++) {
      double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
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
  cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int k = 0, i = 0; i < nRows; i++) {
    int y = i * resizeHeight + (i + 1) * edgeThickness;
    int x_end = edgeThickness;
    for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
      int x = x_end;
      cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
      cv::Size s = canvasImage(roi).size();
      // change the number of channels to three
      cv::Mat target_ROI(s, CV_8UC3);
      if (vecMat[k].channels() != canvasImage.channels()) {
        if (vecMat[k].channels() == 1) {
          cv::cvtColor(vecMat[k], target_ROI, cv::COLOR_GRAY2BGR);
        }
      } else {
        vecMat[k].copyTo(target_ROI);
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

//* Forked from OpenCV wiki.
//! Too old. Not recommended!
//@ref https://github.com/opencv/opencv/wiki/DisplayManyImages
void DisplayManyImages(const std::string& winname, int nArgs, ...) {
  int size;
  int i;
  int m, n;
  int x, y;

  // w - Maximum number of images in a row
  // h - Maximum number of images in a column
  int w, h;

  // scale - How much we have to resize the image
  double scale;
  int max;

  // If the number of arguments is lesser than 0 or greater than 12
  // return without displaying
  if (nArgs <= 0) {
    LOG(ERROR) << "Number of arguments too small....";
    return;
  } else if (nArgs > 14) {
    LOG(ERROR) << "Number of arguments too large, can only handle maximally 12 "
                  "images at "
                  "a time ...";
    return;
  }
  // Determine the size of the image,
  // and the number of rows/cols
  // from number of arguments
  else if (nArgs == 1) {
    w = h = 1;
    size = 300;
  } else if (nArgs == 2) {
    w = 2;
    h = 1;
    size = 300;
  } else if (nArgs == 3 || nArgs == 4) {
    w = 2;
    h = 2;
    size = 300;
  } else if (nArgs == 5 || nArgs == 6) {
    w = 3;
    h = 2;
    size = 200;
  } else if (nArgs == 7 || nArgs == 8) {
    w = 4;
    h = 2;
    size = 200;
  } else {
    w = 4;
    h = 3;
    size = 150;
  }

  // Create a new 3 channel image
  cv::Mat DispImage =
      cv::Mat::zeros(cv::Size(100 + size * w, 60 + size * h), CV_8UC3);

  // Used to get the arguments passed
  va_list args;
  va_start(args, nArgs);

  // Loop for nArgs number of arguments
  for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
    // Get the Pointer to the IplImage
    cv::Mat img = va_arg(args, cv::Mat);

    // Check whether it is NULL or not
    // If it is NULL, release the image, and return
    if (img.empty()) {
      LOG(ERROR) << "Invalid arguments";
      return;
    }

    // Find the width and height of the image
    x = img.cols;
    y = img.rows;

    // Find whether height or width is greater in order to resize the image
    max = (x > y) ? x : y;

    // Find the scaling factor to resize the image
    scale = (double)((double)max / size);

    // Used to Align the images
    if (i % w == 0 && m != 20) {
      m = 20;
      n += 20 + size;
    }

    // Set the image ROI to display the current image
    // Resize the input image and copy the it to the Single Big Image
    cv::Rect ROI(m, n, (int)(x / scale), (int)(y / scale));
    cv::Mat temp;
    cv::resize(img, temp, cv::Size(ROI.width, ROI.height));
    temp.copyTo(DispImage(ROI));
  }

  // Create a new window, and show the Single Big Image
  cv::namedWindow(winname, 1);
  cv::imshow(winname, DispImage);
  cv::waitKey(0);

  // End the number of arguments
  va_end(args);
}

#endif  // UZH_FEATURE_SUBPLOT_H_