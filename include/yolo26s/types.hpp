#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace yolo26s {

struct Config {
  std::string model_path = "/workspace/yolo26s_split_quantized.dlc";
  std::string image_path =
      "/workspace/DJI_20251029164538_0027_V_   .JPG";
  std::string output_path = "/workspace/yolo26s_split_quantized_cpp.jpg";
  std::string qnn_net_run = "qnn-net-run";
  std::string backend =
      "/opt/qairt-embedded/lib/x86_64-linux-clang/libQnnHtp.so";
  std::string dlc_loader =
      "/opt/qairt-embedded/lib/x86_64-linux-clang/libQnnModelDlc.so";
  int input_size = 1024;
  int expected_anchors = 21504;
  float score_threshold = 0.60F;
  float nms_iou = 0.45F;
};

struct LetterboxMeta {
  int source_width = 0;
  int source_height = 0;
  int resized_width = 0;
  int resized_height = 0;
  int pad_left = 0;
  int pad_top = 0;
  int input_size = 0;
  float scale = 1.0F;
};

struct LetterboxInput {
  cv::Mat image_bgr;
  std::vector<float> tensor_nchw;
  LetterboxMeta meta;
};

struct RawOutputs {
  std::vector<float> boxes;
  std::vector<float> scores;
};

struct Detection {
  int class_id = 0;
  float score = 0.0F;
  float angle_rad = 0.0F;
  cv::Rect2f box;
};

struct PostprocessResult {
  std::size_t candidates = 0;
  std::vector<Detection> detections;
};

}  // namespace yolo26s
