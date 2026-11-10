#include "yolo26s/postprocessor.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace yolo26s {

namespace {

struct Candidate {
  int anchor = 0;
  float score = 0.0F;
  float angle_rad = 0.0F;
  cv::Rect2f box;
};

}  // namespace

float DetectionPostprocessor::iou(const cv::Rect2f& first,
                                  const cv::Rect2f& second) {
  const float x1 = std::max(first.x, second.x);
  const float y1 = std::max(first.y, second.y);
  const float x2 = std::min(first.x + first.width, second.x + second.width);
  const float y2 = std::min(first.y + first.height, second.y + second.height);
  const float intersection =
      std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
  const float union_area =
      first.width * first.height + second.width * second.height -
      intersection;
  return intersection / (union_area + 1e-9F);
}

PostprocessResult DetectionPostprocessor::run(const RawOutputs& outputs,
                                              const LetterboxMeta& meta,
                                              float score_threshold,
                                              float nms_iou) const {
  if (outputs.boxes.size() % 4 != 0) {
    throw std::runtime_error("output_boxes is not divisible by 4");
  }

  const std::size_t anchors = outputs.boxes.size() / 4;
  if (anchors == 0 || outputs.scores.size() % anchors != 0) {
    throw std::runtime_error("output_scores does not match anchor count");
  }
  const std::size_t score_channels = outputs.scores.size() / anchors;

  std::vector<Candidate> candidates;
  for (std::size_t anchor = 0; anchor < anchors; ++anchor) {
    const float score = outputs.scores[anchor];
    if (score < score_threshold) {
      continue;
    }

    const float center_x = outputs.boxes[anchor];
    const float center_y = outputs.boxes[anchors + anchor];
    const float width = outputs.boxes[2 * anchors + anchor];
    const float height = outputs.boxes[3 * anchors + anchor];
    if (width <= 0.0F || height <= 0.0F) {
      continue;
    }

    const cv::Rect2f box(center_x - width / 2.0F,
                         center_y - height / 2.0F, width, height);

    // Match the Python reference: only keep boxes fully inside the
    // non-padded vertical content region.
    if (box.y < meta.pad_top ||
        box.y + box.height > meta.pad_top + meta.resized_height) {
      continue;
    }

    const float angle =
        score_channels > 1 ? outputs.scores[anchors + anchor] : 0.0F;
    candidates.push_back(
        {static_cast<int>(anchor), score, angle, box});
  }

  PostprocessResult result;
  result.candidates = candidates.size();
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& left, const Candidate& right) {
              return left.score > right.score;
            });

  std::vector<Candidate> kept;
  kept.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    bool suppressed = false;
    for (const auto& previous : kept) {
      if (iou(candidate.box, previous.box) > nms_iou) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(candidate);
    }
  }

  for (const auto& candidate : kept) {
    const float x1 =
        (candidate.box.x - meta.pad_left) / meta.scale;
    const float y1 =
        (candidate.box.y - meta.pad_top) / meta.scale;
    const float x2 =
        (candidate.box.x + candidate.box.width - meta.pad_left) /
        meta.scale;
    const float y2 =
        (candidate.box.y + candidate.box.height - meta.pad_top) /
        meta.scale;

    Detection detection;
    detection.class_id = 0;
    detection.score = candidate.score;
    detection.angle_rad = candidate.angle_rad;
    detection.box.x =
        std::clamp(x1, 0.0F, static_cast<float>(meta.source_width - 1));
    detection.box.y =
        std::clamp(y1, 0.0F, static_cast<float>(meta.source_height - 1));
    detection.box.width =
        std::clamp(x2, 0.0F, static_cast<float>(meta.source_width - 1)) -
        detection.box.x;
    detection.box.height =
        std::clamp(y2, 0.0F, static_cast<float>(meta.source_height - 1)) -
        detection.box.y;
    result.detections.push_back(detection);
  }

  return result;
}

}  // namespace yolo26s
