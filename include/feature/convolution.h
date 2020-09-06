#ifndef UZH_FEATURE_CONV2_H_
#define UZH_FEATURE_CONV2_H_

#include <optional>

#include "Eigen/Core"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

// TODO(bayes) Remove OpenCV dependency or (recommended) make a overloaded
// version for Eigen::Matrix
//@ref http://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html#
//@see the Custom index list section at the bottom of the page.
// E.g.
// struct pad {
//   Index size() const { return out_size; }
//   Index operator[] (Index i) const { return
//   std::max<Index>(0,i-(out_size-in_size)); } Index in_size, out_size;
// };

// Matrix3i A;
// A.reshaped() = VectorXi::LinSpaced(9,1,9);
// cout << "Initial matrix A:\n" << A << "\n\n";
// MatrixXi B(5,5);
// B = A(pad{3,5}, pad{3,5});
// cout << "A(pad{3,N}, pad{3,N}):\n" << B << "\n\n";

//@brief Imitate matlab's padarray. Pad the input image using the given pad_size
// and pad_value.
//@param image Input image to be padded.
//@param pad_size Vector contains size of padded values corresponding to
// different directions.
//@param pad_value Vector contains values to be padded in certain directions
// according to the pad_size. By default, the values are zeros in each dimenion.
// The size(pad_size) must be equal to size(pad_value)
//
// E.g. PadArray(A, [1, 2, 3, 4], 1) pads 1 ones to top, 2 ones to
// bottom, 3 ones to left and 4 ones to right, where the 1, 2, 3, 4 denotes the
// width of the padded borders in different directions.
void PadArray(cv::Mat& image, const cv::Scalar_<int> pad_size,
              const double pad_value = 0) {
  cv::copyMakeBorder(image, image, pad_size(0), pad_size(1), pad_size(2),
                     pad_size(3), cv::BORDER_CONSTANT, {pad_value});
}

// enum ConvolutionType {
//   /* Return the full convolution, including border */
//   CONVOLUTION_FULL,

//   /* Return only the part that corresponds to the original image */
//   CONVOLUTION_SAME,

//   /* Return only the submatrix containing elements that were not
//   influenced by
//    * the border
//    */
//   CONVOLUTION_VALID
// };

// void conv2(const Mat& img, const Mat& kernel, ConvolutionType type, Mat&
// dest) {
//   Mat source = img;
//   if (CONVOLUTION_FULL == type) {
//     source = Mat();
//     const int additionalRows = kernel.rows - 1,
//               additionalCols = kernel.cols - 1;
//     copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows
//     / 2,
//                    (additionalCols + 1) / 2, additionalCols / 2,
//                    BORDER_CONSTANT, Scalar(0));
//   }

//   Point anchor(kernel.cols - kernel.cols / 2 - 1,
//                kernel.rows - kernel.rows / 2 - 1);
//   int borderMode = BORDER_CONSTANT;
//   // filter2D(source, dest, img.depth(), /*flip(kernel)*/, anchor, 0,
//   // borderMode);

//   if (CONVOLUTION_VALID == type) {
//     dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols /
//     2)
//                .rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows /
//                2);
//   }
// }

//@brief Imitate matlab's conv2. Convolve the image with the given kernel.
// TODO(bayes) Remove OpenCV dependency.
void Conv2D(cv::InputArray src, cv::OutputArray dst, int ddepth,
            cv::InputArray kernel, cv::Point anchor = cv::Point(-1, -1),
            double delta = 0.0, int border_type = 4) {
  cv::filter2D(src, dst, ddepth, kernel, anchor, delta, border_type);
}

#endif  // UZH_FEATURE_CONV2_H_