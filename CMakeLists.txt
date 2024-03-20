cmake_minimum_required(VERSION 2.8)
project(ObjSLAM)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

#SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE relwithdebinfo)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# ======  start c++ compile  ======

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
add_definitions(-DCOMPILEDWITHC14)
message(STATUS "Using flag -std=c++14.")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

ADD_DEFINITIONS(-DUSE_ONNX=true )

find_package(OpenCV 4.0 QUIET)

find_package(Eigen3 3.1.0 REQUIRED)
find_package(CSparse REQUIRED)
find_package(Pangolin REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.16.3")

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/YOLOv8/include
${EIGEN3_INCLUDE_DIR}
${CSPARSE_INCLUDE_DIR}
${ONNXRUNTIME_DIR}/include
${Pangolin_INCLUDE_DIRS}
)


add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/ORBextractor.cc
src/Converter.cc
src/Map.cc
src/Optimizer.cc
src/Frame.cc
src/Viewer.cc

src/cvplot/color.cc
src/cvplot/figure.cc
src/cvplot/highgui.cc
src/cvplot/window.cc
src/cvplot/internal.h

YOLOv8/src/detector_onnxruntime.cpp
YOLOv8/src/detector_opencv_dnn.cpp
YOLOv8/src/segmentor_onnxruntime.cpp
YOLOv8/src/segmentor_opencv_dnn.cpp

YOLOv8/include/data_struct.h
YOLOv8/include/detector_onnxruntime.h
YOLOv8/include/detector_opencv_dnn.h
YOLOv8/include/segmentor_onnxruntime.h
YOLOv8/include/segmentor_opencv_dnn.h
)

target_link_libraries(
${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${CSPARSE_LIBRARY}
${Pangolin_LIBRARIES}
# for mac os (default)
#${PROJECT_SOURCE_DIR}/dependencies/g2o/lib/libg2o.dylib
# for linux (ubuntu)
${PROJECT_SOURCE_DIR}/dependencies/g2o/lib/libg2o.so
${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
-lboost_serialization
-lcrypto
)

message("***** CUDA")

find_package(CUDA)
if(CUDA_FOUND)
    message("********** CUDA " ${CUDA_VERSION} " founded in: " ${CUDA_INCLUDE_DIRS})
    ADD_DEFINITIONS( -DCUDA_STATUS=true )
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
else()
    message("********** CUDA not founded")
    ADD_DEFINITIONS( -DCUDA_STATUS=false )
endif(CUDA_FOUND)

# Build example

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)

add_executable(vdo_slam
example/vdo_slam.cc)
target_link_libraries(vdo_slam ${PROJECT_NAME})


