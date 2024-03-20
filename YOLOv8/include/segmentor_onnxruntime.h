#pragma once

#include<onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include <numeric> 
#include "data_struct.h"

class Segmentor_ONNXRUNTIME
{
public:
    Segmentor_ONNXRUNTIME();

    bool LoadModel(std::string& modelPath);
    BatchSegmentedObject Run(MatVector& srcImgList);

    void setBatchSize(int newBatch);
    void setInputSize(cv::Size newInputSize);
    void setClassNames(std::vector<std::string> newClassNamesList);
    std::string getClassName(int classId);

    void setDynamicClassNames(std::vector<std::string> classNamesDynamicList);
    bool whetherInDynamicClass(std::string className);

private:
    void Preprocessing(const std::vector<cv::Mat>& SrcImgs, std::vector<cv::Mat>& OutSrcImgs, std::vector<cv::Vec4d>& params);
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
    template <typename T>
    T VectorProduct(const std::vector<T>& v)
    {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    };
    int _cudaID = 0;
    float _classThreshold = 0.25;
    float _nmsThreshold = 0.45;
    float _maskThreshold = 0.5;
    Ort::Session* _OrtSession = nullptr;
    Ort::Env _OrtEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov8-Seg");
    std::shared_ptr<char> _inputName, _output_name0, _output_name1;
    std::vector<char*> _inputNodeNames, _outputNodeNames;
    std::vector<int64_t> _inputTensorShape, _outputTensorShape, _outputMaskTensorShape;
    bool _isDynamicShape = false; //onnx support dynamic shape
    Ort::MemoryInfo _OrtMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput);

    int _batchSize = 1;
    cv::Size _inputSize = cv::Size(640, 640);
    std::vector<std::string> _classNamesList;
    std::vector<std::string> _classNamesDynamicList;
};
