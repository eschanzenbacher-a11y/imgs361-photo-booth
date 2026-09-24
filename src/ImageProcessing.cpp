#include "photo_booth/ImageProcessing.hpp"

#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace photo_booth {

namespace {

void validateImage(const cv::Mat& image, const char* function_name) {
  if (image.empty()) {
    throw std::invalid_argument(std::string(function_name) +
                                ": input image is empty");
  }

  if (image.type() != CV_8UC3) {
    throw std::invalid_argument(std::string(function_name) +
                                ": input image must be CV_8UC3");
  }
}

}  // namespace

cv::Mat calcHist(const cv::Mat& image) {
  validateImage(image, "calcHist()");

  cv::Mat histogram = cv::Mat_<int>::zeros(3, 256);
  for (int row = 0; row < image.rows; ++row) {
    for (int column = 0; column < image.cols; ++column) {
      auto value = image.at<cv::Vec3b>(row, column);
      histogram.at<int>(0, value[0])++;
      histogram.at<int>(1, value[1])++;
      histogram.at<int>(2, value[2])++;
    }
  }

  return histogram;
}

cv::Mat swapRedBlueChannels(const cv::Mat& image) {
  validateImage(image, "swapRedBlueChannels()");

  cv::Mat output;

  cv::cvtColor(image, output, cv::COLOR_BGR2RGB);

  return output;
}

cv::Mat invertImage(const cv::Mat& image) {
  validateImage(image, "invertImage()");

  cv::Mat output;

  cv::bitwise_not(image, output);

  return output;
}

cv::Mat quantizeImage(const cv::Mat& image, int levels) {
  validateImage(image, "quantizeImage()");
    if (levels < 1 || levels > 256) {
  throw std::invalid_argument("quantizeImage(): levels must be between 1 and 256");
}

  cv::Mat output = image.clone();

  int step = 256 / levels;

  for (int row = 0; row < image.rows; ++row) {
    for (int column = 0; column < image.cols; ++column) {
      cv::Vec3b& pixel = output.at<cv::Vec3b>(row, column);
        for (int channel = 0; channel < 3; ++channel) {
          pixel[channel] = (pixel[channel] / step) * step;
      }
    }
  }

  return output;
}

cv::Mat equalizeHistogram(const cv::Mat& image) {
  validateImage(image, "equalizeHistogram()");

  cv::Mat ycrcb;
  cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);

  std::vector<cv::Mat> channels;
  cv::split(ycrcb, channels);

  int histogram[256] = {0};

  for (int row = 0; row < channels[0].rows; ++row) {
   for (int column = 0; column < channels[0].cols; ++column) {
    unsigned char value = channels[0].at<unsigned char>(row, column);
    histogram[value]++;
   }
 }
int cdf[256] = {0};

cdf[0] = histogram[0];

for (int i = 1; i < 256; ++i) {
  cdf[i] = cdf[i - 1] + histogram[i];
}
int total_pixels = channels[0].rows * channels[0].cols;

int cdf_min = 0;
for (int i = 0; i < 256; ++i) {
  if (histogram[i] != 0) {
    cdf_min = cdf[i];
    break;
  }
}

if (total_pixels == cdf_min) {
  return image.clone();
}

unsigned char lut[256];

for (int i = 0; i < 256; ++i) {
  if (cdf[i] < cdf_min) {
    lut[i] = 0;
  } else {
    lut[i] = static_cast<unsigned char>(
        ((cdf[i] - cdf_min) * 255.0) / (total_pixels - cdf_min));
  }
}

for (int row = 0; row < channels[0].rows; ++row) {
  for (int column = 0; column < channels[0].cols; ++column) {
    unsigned char& value =
        channels[0].at<unsigned char>(row, column);

    value = lut[value];
  }
}
cv::merge(channels, ycrcb);

cv::Mat output;
cv::cvtColor(ycrcb, output, cv::COLOR_YCrCb2BGR);

return output;
}


cv::Mat meanFilter(const cv::Mat& image) {
  validateImage(image, "meanFilter()");

  cv::Mat output = image.clone();

  for (int row = 1; row < image.rows - 1; ++row) {
    for (int column = 1; column < image.cols - 1; ++column) {
      int blue_sum = 0;
      int green_sum = 0;
      int red_sum = 0;

      for (int row_offset = -1; row_offset <= 1; ++row_offset) {
        for (int column_offset = -1; column_offset <= 1; ++column_offset) {
          cv::Vec3b neighbor =
              image.at<cv::Vec3b>(row + row_offset,
                                  column + column_offset);

          blue_sum += neighbor[0];
          green_sum += neighbor[1];
          red_sum += neighbor[2];
        }
      }

      cv::Vec3b& output_pixel = output.at<cv::Vec3b>(row, column);

      output_pixel[0] = blue_sum / 9;
      output_pixel[1] = green_sum / 9;
      output_pixel[2] = red_sum / 9;
    }
  }

  return output;
}

}  // namespace photo_booth
