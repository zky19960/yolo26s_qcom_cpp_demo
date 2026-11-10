#pragma once

#include "yolo26s/types.hpp"

namespace yolo26s {

class ImagePreprocessor {
 public:
  LetterboxInput run(const std::string& image_path, int input_size) const;
};

}  // namespace yolo26s
