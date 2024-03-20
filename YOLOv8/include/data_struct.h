#pragma once

#include <opencv2/core/mat.hpp>
#include <string> 


using MatVector = std::vector<cv::Mat>;

#pragma pack(push, n)
struct SegmentedObject {
    int classID;
    float confidence;
    cv::Rect box;
    cv::Mat boxMask;
    std::vector<std::vector<cv::Point>> maskContoursList;
};
#pragma pack(pop)

using ImagesSegmentedObject = std::vector<SegmentedObject>;
using BatchSegmentedObject = std::vector<ImagesSegmentedObject>;

struct MaskParams {
    //int segChannels = 32;
    //int segWidth = 160;
    //int segHeight = 160;
    int netWidth = 640;
    int netHeight = 640;
    float maskThreshold = 0.5;
    cv::Size srcImgShape;
    cv::Vec4d params;
};

struct DetectedObject_bak {
    int classID;
    float confidence;
    cv::Rect box;
};

using DetectedObject = SegmentedObject;
using ImagesDetectedObject = std::vector<DetectedObject>;
using BatchDetectedObject = std::vector<ImagesDetectedObject>;
