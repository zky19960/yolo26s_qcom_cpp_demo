#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "yolo26s/image_preprocessor.hpp"
#include "yolo26s/postprocessor.hpp"
#include "yolo26s/qnn_runner.hpp"
#include "yolo26s/result_renderer.hpp"
#include "yolo26s/types.hpp"

namespace {

void printUsage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --model <path>             Default: "
         "/workspace/yolo26s_split_quantized.dlc\n"
      << "  --image <path>             Input JPEG/PNG\n"
      << "  --output <path>            Annotated output JPEG\n"
      << "  --backend <path>           QNN backend, default libQnnHtp.so\n"
      << "  --dlc-loader <path>        QNN DLC loader, default libQnnModelDlc.so\n"
      << "  --qnn-net-run <path>       qnn-net-run executable or command name\n"
      << "  --score-threshold <float>  Default: 0.60\n"
      << "  --nms-iou <float>          Default: 0.45\n";
}

bool consumeValue(int argc, char** argv, int& index, std::string& output) {
  if (index + 1 >= argc) {
    return false;
  }
  output = argv[++index];
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    yolo26s::Config config;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--help" || argument == "-h") {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
      }
      if (argument == "--model") {
        if (!consumeValue(argc, argv, i, config.model_path)) {
          throw std::invalid_argument("--model requires a value");
        }
      } else if (argument == "--image") {
        if (!consumeValue(argc, argv, i, config.image_path)) {
          throw std::invalid_argument("--image requires a value");
        }
      } else if (argument == "--output") {
        if (!consumeValue(argc, argv, i, config.output_path)) {
          throw std::invalid_argument("--output requires a value");
        }
      } else if (argument == "--backend") {
        if (!consumeValue(argc, argv, i, config.backend)) {
          throw std::invalid_argument("--backend requires a value");
        }
      } else if (argument == "--dlc-loader") {
        if (!consumeValue(argc, argv, i, config.dlc_loader)) {
          throw std::invalid_argument("--dlc-loader requires a value");
        }
      } else if (argument == "--qnn-net-run") {
        if (!consumeValue(argc, argv, i, config.qnn_net_run)) {
          throw std::invalid_argument("--qnn-net-run requires a value");
        }
      } else if (argument == "--score-threshold") {
        std::string value;
        if (!consumeValue(argc, argv, i, value)) {
          throw std::invalid_argument("--score-threshold requires a value");
        }
        config.score_threshold = std::stof(value);
      } else if (argument == "--nms-iou") {
        std::string value;
        if (!consumeValue(argc, argv, i, value)) {
          throw std::invalid_argument("--nms-iou requires a value");
        }
        config.nms_iou = std::stof(value);
      } else {
        throw std::invalid_argument("Unknown argument: " + argument);
      }
    }

    yolo26s::ImagePreprocessor preprocessor;
    auto prepared =
        preprocessor.run(config.image_path, config.input_size);

    yolo26s::QnnRunner runner(config);
    const auto raw_outputs = runner.run(prepared.tensor_nchw);

    yolo26s::DetectionPostprocessor postprocessor;
    const auto result = postprocessor.run(
        raw_outputs, prepared.meta, config.score_threshold, config.nms_iou);

    yolo26s::ResultRenderer renderer;
    const auto output = renderer.draw(prepared.image_bgr, result.detections);
    renderer.save(config.output_path, output);

    std::cout << "Image: " << config.image_path << '\n'
              << "Letterbox: scale=" << prepared.meta.scale
              << ", left=" << prepared.meta.pad_left
              << ", top=" << prepared.meta.pad_top
              << ", content=" << prepared.meta.resized_width << "x"
              << prepared.meta.resized_height << '\n'
              << "Candidates >= " << config.score_threshold << ": "
              << result.candidates << ", after NMS: "
              << result.detections.size() << '\n';

    for (std::size_t i = 0; i < result.detections.size(); ++i) {
      const auto& detection = result.detections[i];
      std::cout << "#" << (i + 1) << " class=" << detection.class_id
                << " score=" << detection.score << " box=("
                << detection.box.x << ", " << detection.box.y << ", "
                << detection.box.width << ", " << detection.box.height
                << ")\n";
    }
    std::cout << "Saved: " << config.output_path << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
