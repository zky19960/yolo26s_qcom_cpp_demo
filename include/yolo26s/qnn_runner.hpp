#pragma once

#include <vector>

#include "yolo26s/types.hpp"

namespace yolo26s {

class QnnRunner {
 public:
  explicit QnnRunner(Config config);

  RawOutputs run(const std::vector<float>& input_nchw) const;

 private:
  Config config_;
};

}  // namespace yolo26s
