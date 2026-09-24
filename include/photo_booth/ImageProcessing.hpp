#pragma once

#include <opencv2/core.hpp>

namespace photo_booth {

/**
 * @brief Calculates the histogram of each channel of an 8-bit BGR image.
 *
 * Channel histograms are returned in the rows of a 3x256 CV_32S.
 */
cv::Mat calcHist(const cv::Mat& image);

/**
 * @brief Swaps the blue and red channels of an 8-bit BGR image.
 */
cv::Mat swapRedBlueChannels(const cv::Mat& image);

/**
 * @brief Inverts every channel of an 8-bit BGR image.
 *
 * Each output channel value is 255 minus the corresponding input value.
 */
cv::Mat invertImage(const cv::Mat& image);

/**
 * @brief Quantizes an 8-bit BGR image to a specified number of levels.
 */
cv::Mat quantizeImage(const cv::Mat& image, int levels);

/**
 * @brief Applies histogram equalization to an 8-bit BGR image.
 */
cv::Mat equalizeHistogram(const cv::Mat& image);

/**
 * @brief Applies a 3x3 mean smoothing filter to an 8-bit BGR image.
 */
cv::Mat meanFilter(const cv::Mat& image);
}  // namespace photo_booth
