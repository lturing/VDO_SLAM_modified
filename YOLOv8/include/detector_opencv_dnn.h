#pragma once

#include <opencv2/opencv.hpp>
#include "data_struct.h"


class Detector_OpenCV_DNN
{
public:
    Detector_OpenCV_DNN();

    bool LoadModel(std::string& modelPath);
    BatchDetectedObject Run(MatVector& srcImgList);

    void setClassNames(std::vector<std::string> newClassNamesList);
    void setBatchSize(int newBatch);
    void setInputSize(cv::Size newInputSize);
    std::string getClassName(int classId);

    void setDynamicClassNames(std::vector<std::string> classNamesDynamicList);
    bool whetherInDynamicClass(std::string className);
    
private:
    cv::dnn::Net model;
    float _classThreshold = 0.25;
    float _nmsThreshold = 0.45;

    void LetterBox(const cv::Mat& image,
                   cv::Mat& outImage,
                   cv::Vec4d& params,
                   const cv::Size& newShape = cv::Size(640, 640),
                   bool autoShape = false,
                   bool scaleFill = false,
                   bool scaleUp = true,
                   int stride = 32,
                   const cv::Scalar& color = cv::Scalar(114, 114, 114));
    
    std::vector<std::string> _classNamesList;
    std::vector<std::string> _classNamesDynamicList;
    int _batchSize = 1;
    cv::Size _inputSize = cv::Size(640, 640);

};
