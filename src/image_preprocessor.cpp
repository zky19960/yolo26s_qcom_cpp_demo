#include "yolo26s/image_preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo26s {

LetterboxInput ImagePreprocessor::run(const std::string& image_path,
                                      int input_size) const {
  if (input_size <= 0) {
    throw std::invalid_argument("input_size must be positive");
  }

  LetterboxInput result;
  result.image_bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (result.image_bgr.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }

  auto& meta = result.meta;
  meta.source_width = result.image_bgr.cols;
  meta.source_height = result.image_bgr.rows;
  meta.input_size = input_size;
  meta.scale = std::min(
      static_cast<float>(input_size) / meta.source_width,
      static_cast<float>(input_size) / meta.source_height);
  meta.resized_width =
      static_cast<int>(std::lround(meta.source_width * meta.scale));
  meta.resized_height =
      static_cast<int>(std::lround(meta.source_height * meta.scale));
  meta.pad_left = (input_size - meta.resized_width) / 2;
  meta.pad_top = (input_size - meta.resized_height) / 2;

  cv::Mat resized;
  cv::resize(result.image_bgr, resized,
             cv::Size(meta.resized_width, meta.resized_height), 0.0, 0.0,
             cv::INTER_LINEAR);

  cv::Mat canvas(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(
      canvas(cv::Rect(meta.pad_left, meta.pad_top, meta.resized_width,
                      meta.resized_height)));

  cv::Mat rgb;
  cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

  result.tensor_nchw.resize(static_cast<std::size_t>(3) * input_size *
                            input_size);
  const std::size_t plane_size =
      static_cast<std::size_t>(input_size) * input_size;
  for (int y = 0; y < input_size; ++y) {
    const auto* row = rgb.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input_size; ++x) {
      const std::size_t pixel = static_cast<std::size_t>(y) * input_size + x;
      for (int channel = 0; channel < 3; ++channel) {
        result.tensor_nchw[static_cast<std::size_t>(channel) * plane_size +
                           pixel] =
            static_cast<float>(row[x][channel]) / 255.0F;
      }
    }
  }

  return result;
}

}  // namespace yolo26s
