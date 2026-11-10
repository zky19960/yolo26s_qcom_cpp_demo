#pragma once

#include "yolo26s/types.hpp"

namespace yolo26s {

class DetectionPostprocessor {
 public:
  PostprocessResult run(const RawOutputs& outputs,
                        const LetterboxMeta& meta,
                        float score_threshold,
                        float nms_iou) const;

 private:
  static float iou(const cv::Rect2f& first, const cv::Rect2f& second);
};

}  // namespace yolo26s
