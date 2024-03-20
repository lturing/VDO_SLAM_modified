cmake_minimum_required(VERSION 3.0.0)
project(yolov8)
message("build for ${PROJECT_NAME}")
#set(CMAKE_CXX_STANDARD 14)


STRING(REGEX MATCH "-std=c\\+\\+[0-9]+" result ${CMAKE_CXX_FLAGS})
IF(NOT ${result} MATCHES "-std=c\\+\\+[0-9]+")
    set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
ENDIF()


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

#SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE relwithdebinfo)

MESSAGE("Build type: ${CMAKE_BUILD_TYPE} for ${PROJECT_NAME}")
MESSAGE("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS} in ${PROJECT_NAME}")
MESSAGE("CMAKE_C_FLAGS: ${CMAKE_C_FLAGS} in ${PROJECT_NAME}")
MESSAGE("CMAKE_C_FLAGS_RELEASE: ${CMAKE_C_FLAGS_RELEASE} in ${PROJECT_NAME}")
MESSAGE("CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE} in ${PROJECT_NAME}")

find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
    include_directories("/usr/include/eigen3")
endif()

#add_definitions(-DENABLE_PRECOMPILED_HEADERS=OFF)
#message("EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")

find_package(OpenCV 4.8 REQUIRED)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../onnxruntime-linux-x64-1.16.3")

set(YOLOV8_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/detector_onnxruntime.cpp
    ${PROJECT_SOURCE_DIR}/src/detector_opencv_dnn.cpp
    ${PROJECT_SOURCE_DIR}/src/segmentor_onnxruntime.cpp
    ${PROJECT_SOURCE_DIR}/src/segmentor_opencv_dnn.cpp
    ${PROJECT_SOURCE_DIR}/bytetrack/src/lapjv.cpp
    ${PROJECT_SOURCE_DIR}/bytetrack/src/kalmanFilter.cpp
    ${PROJECT_SOURCE_DIR}/bytetrack/src/BYTETracker.cpp
    ${PROJECT_SOURCE_DIR}/bytetrack/src/STrack.cpp
)

#include_directories(
#    ${OpenCV_INCLUDE_DIRS}
#    ${EIGEN3_INCLUDE_DIR}
#    ${PROJECT_SOURCE_DIR}/include
#    ${PROJECT_SOURCE_DIR}/bytetrack/include
#    ${ONNXRUNTIME_DIR}/include
#)

add_library(${PROJECT_NAME} SHARED ${YOLOV8_SOURCE_FILES})

#target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
target_include_directories(${PROJECT_NAME} PRIVATE
                           ${PROJECT_SOURCE_DIR}/include
                           ${PROJECT_SOURCE_DIR}/bytetrack/include
                           ${EIGEN3_INCLUDE_DIR}
                           ${OpenCV_INCLUDE_DIRS}  
                           ${ONNXRUNTIME_DIR}/include
)
target_link_libraries(${PROJECT_NAME}
                      ${OpenCV_LIBRARIES}
                      ${EIGEN3_LIBS}
                      ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
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

# Force to use CPU
#ADD_DEFINITIONS( -DCUDA_STATUS=false )