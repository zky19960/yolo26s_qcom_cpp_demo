cmake_minimum_required(VERSION 3.14)
project(yolo26s_htp_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)

if(OpenCV_FOUND)
  set(YOLO26S_OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
  set(YOLO26S_OPENCV_LIBRARIES ${OpenCV_LIBS})
else()
  # Ubuntu's split OpenCV development packages provide the headers and
  # unversioned link libraries without OpenCVConfig.cmake.
  find_path(YOLO26S_OPENCV_INCLUDE_DIR
    NAMES opencv2/core.hpp
    HINTS /usr/include/opencv4
    REQUIRED
  )
  find_library(YOLO26S_OPENCV_CORE_LIBRARY
    NAMES opencv_core
    REQUIRED
  )
  find_library(YOLO26S_OPENCV_IMGPROC_LIBRARY
    NAMES opencv_imgproc
    REQUIRED
  )
  find_library(YOLO26S_OPENCV_IMGCODECS_LIBRARY
    NAMES opencv_imgcodecs
    REQUIRED
  )
  set(YOLO26S_OPENCV_INCLUDE_DIRS ${YOLO26S_OPENCV_INCLUDE_DIR})
  set(YOLO26S_OPENCV_LIBRARIES
    ${YOLO26S_OPENCV_CORE_LIBRARY}
    ${YOLO26S_OPENCV_IMGPROC_LIBRARY}
    ${YOLO26S_OPENCV_IMGCODECS_LIBRARY}
  )
endif()

add_executable(yolo26s_htp_demo
  src/main.cpp
  src/image_preprocessor.cpp
  src/qnn_runner.cpp
  src/postprocessor.cpp
  src/result_renderer.cpp
)

target_include_directories(
  yolo26s_htp_demo
  PRIVATE
    include
    ${YOLO26S_OPENCV_INCLUDE_DIRS}
)
target_link_libraries(
  yolo26s_htp_demo
  PRIVATE
    ${YOLO26S_OPENCV_LIBRARIES}
)

target_compile_options(yolo26s_htp_demo PRIVATE -Wall -Wextra -Wpedantic)
