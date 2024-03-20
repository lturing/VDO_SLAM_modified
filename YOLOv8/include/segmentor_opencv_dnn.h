#pragma once
#include <opencv2/opencv.hpp>
#include "opencv2/core/mat.hpp"
#include "data_struct.h"



class Segmentor_OpenCV_DNN
{
public:
    Segmentor_OpenCV_DNN();

    bool LoadModel(std::string& modelPath);
    BatchSegmentedObject Run(MatVector& srcImgList);

    void setBatchSize(int newBatch);
    void setInputSize(cv::Size newInputSize);
    void setClassNames(std::vector<std::string> newClassNamesList);
    std::string getClassName(int classId);

    void setDynamicClassNames(std::vector<std::string> classNamesDynamicList);
    bool whetherInDynamicClass(std::string className);
    
private:
    cv::dnn::Net model;
    float _classThreshold = 0.25;
    float _nmsThreshold = 0.45;
    float _maskThreshold = 0.5;
    int _batchSize = 1;
    cv::Size _inputSize = cv::Size(640, 640);
    std::vector<std::string> _classNamesList;
    std::vector<std::string> _classNamesDynamicList;

    void LetterBox(const cv::Mat& image,
                   cv::Mat& outImage,
                   cv::Vec4d& params,
                   const cv::Size& newShape = cv::Size(640, 640),
                   bool autoShape = false,
                   bool scaleFill = false,
                   bool scaleUp = true,
                   int stride = 32,
                   const cv::Scalar& color = cv::Scalar(114, 114, 114));
    void GetMask(const cv::Mat& maskProposals,
                 const cv::Mat& maskProtos,
                 ImagesSegmentedObject& output,
                 const MaskParams& maskParams);
    void GetMask2(const cv::Mat& maskProposals,
                  const cv::Mat& maskProtos,
                  SegmentedObject& output,
                  const MaskParams& maskParams);
    void calcContours(ImagesSegmentedObject& output);

};
