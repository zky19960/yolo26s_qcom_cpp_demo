#include "yolo26s/qnn_runner.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace yolo26s {
namespace {

namespace fs = std::filesystem;

class TempDirectory {
 public:
  TempDirectory() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = fs::temp_directory_path() /
            ("yolo26s_cpp_" + std::to_string(getpid()) + "_" +
             std::to_string(stamp));
    fs::create_directories(path_);
  }

  ~TempDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }

  const fs::path& path() const { return path_; }

 private:
  fs::path path_;
};

void writeFloats(const fs::path& path, const std::vector<float>& values) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Failed to create input file: " + path.string());
  }
  stream.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!stream) {
    throw std::runtime_error("Failed to write input file: " + path.string());
  }
}

std::vector<float> readFloats(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("Missing QNN output: " + path.string());
  }
  const auto bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
    throw std::runtime_error("Invalid float32 output: " + path.string());
  }
  stream.seekg(0);
  std::vector<float> values(
      static_cast<std::size_t>(bytes / static_cast<std::streamoff>(sizeof(float))));
  stream.read(reinterpret_cast<char*>(values.data()), bytes);
  if (!stream) {
    throw std::runtime_error("Failed to read QNN output: " + path.string());
  }
  return values;
}

void runProcess(const std::vector<std::string>& args) {
  if (args.empty()) {
    throw std::invalid_argument("Command must not be empty");
  }

  std::cout << "Running:";
  for (const auto& arg : args) {
    std::cout << ' ' << arg;
  }
  std::cout << '\n';

  const pid_t child = fork();
  if (child < 0) {
    throw std::runtime_error("fork() failed");
  }
  if (child == 0) {
    std::vector<char*> raw_args;
    raw_args.reserve(args.size() + 1);
    for (const auto& arg : args) {
      raw_args.push_back(const_cast<char*>(arg.c_str()));
    }
    raw_args.push_back(nullptr);
    execvp(raw_args[0], raw_args.data());
    std::cerr << "Failed to execute " << args[0] << ": "
              << std::strerror(errno) << '\n';
    _exit(127);
  }

  int status = 0;
  if (waitpid(child, &status, 0) < 0) {
    throw std::runtime_error("waitpid() failed");
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error(args[0] + " failed");
  }
}

}  // namespace

QnnRunner::QnnRunner(Config config) : config_(std::move(config)) {}

RawOutputs QnnRunner::run(const std::vector<float>& input_nchw) const {
  TempDirectory temp;
  const fs::path input_raw = temp.path() / "input.raw";
  const fs::path input_list = temp.path() / "input_list.txt";
  const fs::path qnn_output = temp.path() / "qnn_output";

  writeFloats(input_raw, input_nchw);
  {
    std::ofstream stream(input_list);
    if (!stream) {
      throw std::runtime_error("Failed to create input list");
    }
    stream << input_raw.string() << '\n';
  }

  runProcess({
      config_.qnn_net_run,
      "--backend",
      config_.backend,
      "--model",
      config_.dlc_loader,
      "--dlc_path",
      config_.model_path,
      "--input_list",
      input_list.string(),
      "--output_dir",
      qnn_output.string(),
      "--log_level",
      "error",
  });

  const fs::path result_dir = qnn_output / "Result_0";
  RawOutputs outputs;
  outputs.boxes = readFloats(result_dir / "output_boxes.raw");
  outputs.scores = readFloats(result_dir / "output_scores.raw");

  const std::size_t expected_box_values =
      static_cast<std::size_t>(4) * config_.expected_anchors;
  if (outputs.boxes.size() != expected_box_values) {
    throw std::runtime_error(
        "Unexpected output_boxes count: " + std::to_string(outputs.boxes.size()) +
        ", expected: " + std::to_string(expected_box_values));
  }
  if (outputs.scores.empty() ||
      outputs.scores.size() % config_.expected_anchors != 0) {
    throw std::runtime_error(
        "Unexpected output_scores count: " +
        std::to_string(outputs.scores.size()));
  }
  return outputs;
}

}  // namespace yolo26s
