# CMakeLists.txt — JetPack 5.1 / TensorRT 8.5 / CUDA 11.4 / OpenCV 4.5

cmake_minimum_required(VERSION 3.16)
project(yolo_usbcam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- OpenCV ---
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "OpenCV include : ${OpenCV_INCLUDE_DIRS}")

# --- CUDA (JetPack 5.1 = CUDA 11.4 in /usr/local/cuda) ---
find_package(CUDA REQUIRED)
message(STATUS "CUDA version   : ${CUDA_VERSION}")

# --- TensorRT (JetPack 기본 설치 경로) ---
find_path(TRT_INCLUDE_DIR NvInfer.h
    HINTS
        /usr/include/aarch64-linux-gnu
        /usr/include
)
find_library(TRT_NVINFER nvinfer
    HINTS /usr/lib/aarch64-linux-gnu
)
find_library(TRT_NVINFER_PLUGIN nvinfer_plugin
    HINTS /usr/lib/aarch64-linux-gnu
)

if(NOT TRT_INCLUDE_DIR OR NOT TRT_NVINFER)
    message(FATAL_ERROR "TensorRT not found. Check /usr/include/aarch64-linux-gnu/NvInfer.h")
endif()
message(STATUS "TensorRT header: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib   : ${TRT_NVINFER}")

# --- 실행 파일 ---
add_executable(yolo_demo main.cpp)

target_include_directories(yolo_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(yolo_demo PRIVATE
    ${OpenCV_LIBS}
    ${TRT_NVINFER}
    ${TRT_NVINFER_PLUGIN}
    ${CUDA_LIBRARIES}
)

target_compile_options(yolo_demo PRIVATE -Wall -Wno-deprecated-declarations)

