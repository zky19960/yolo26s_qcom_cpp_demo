#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "yolo26s/types.hpp"

namespace yolo26s {

class ResultRenderer {
 public:
  cv::Mat draw(const cv::Mat& image_bgr,
               const std::vector<Detection>& detections) const;

  void save(const std::string& path, const cv::Mat& image_bgr) const;
};

}  // namespace yolo26s
