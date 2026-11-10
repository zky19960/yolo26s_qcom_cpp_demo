#include "yolo26s/result_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo26s {

cv::Mat ResultRenderer::draw(
    const cv::Mat& image_bgr,
    const std::vector<Detection>& detections) const {
  cv::Mat output = image_bgr.clone();
  const int thickness =
      std::max(2, std::max(output.cols, output.rows) / 500);
  const double font_scale = std::max(0.6, output.cols / 4700.0);

  for (const auto& detection : detections) {
    std::vector<cv::Point> polygon;
    if (std::abs(detection.angle_rad) > 1e-6F) {
      const cv::RotatedRect rect(
          cv::Point2f(detection.box.x + detection.box.width / 2.0F,
                      detection.box.y + detection.box.height / 2.0F),
          cv::Size2f(detection.box.width, detection.box.height),
          detection.angle_rad * 180.0F / static_cast<float>(CV_PI));
      cv::Point2f points[4];
      rect.points(points);
      for (const auto& point : points) {
        polygon.emplace_back(cvRound(point.x), cvRound(point.y));
      }
    } else {
      const cv::Rect rect = detection.box;
      polygon = {
          {rect.x, rect.y},
          {rect.x + rect.width, rect.y},
          {rect.x + rect.width, rect.y + rect.height},
          {rect.x, rect.y + rect.height},
      };
    }

    cv::polylines(output, polygon, true, cv::Scalar(0, 255, 0), thickness,
                  cv::LINE_AA);
    const std::string label =
        "class" + std::to_string(detection.class_id) + " " +
        cv::format("%.3f", detection.score);
    const cv::Point origin(
        std::max(0, polygon[0].x),
        std::max(cvRound(font_scale * 30), polygon[0].y - 12));
    cv::putText(output, label, origin, cv::FONT_HERSHEY_SIMPLEX, font_scale,
                cv::Scalar(0, 255, 0), std::max(2, thickness / 2),
                cv::LINE_AA);
  }
  return output;
}

void ResultRenderer::save(const std::string& path,
                          const cv::Mat& image_bgr) const {
  if (!cv::imwrite(path, image_bgr,
                   {cv::IMWRITE_JPEG_QUALITY, 95})) {
    throw std::runtime_error("Failed to save result image: " + path);
  }
}

}  // namespace yolo26s
