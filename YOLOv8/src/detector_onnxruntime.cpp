#include "detector_onnxruntime.h"
#include <algorithm>

#if CUDA_STATUS
#define CUDA_Availability true
#else
#define CUDA_Availability false
#endif

Detector_ONNXRUNTIME::Detector_ONNXRUNTIME() {
}

void Detector_ONNXRUNTIME::setBatchSize(int newBatch)
{
    if (newBatch < 1) newBatch = 1;
    _batchSize = newBatch;
}

void Detector_ONNXRUNTIME::setInputSize(cv::Size newInputSize)
{
    _inputSize = newInputSize;
}

void Detector_ONNXRUNTIME::setClassNames(std::vector<std::string> newClassNamesList)
{
    _classNamesList = newClassNamesList;
}

std::string Detector_ONNXRUNTIME::getClassName(int classId)
{
    return _classNamesList[classId];
}

void Detector_ONNXRUNTIME::setDynamicClassNames(std::vector<std::string> classNamesDynamicList)
{
    _classNamesDynamicList = classNamesDynamicList;
}

bool Detector_ONNXRUNTIME::whetherInDynamicClass(std::string className)
{
    return std::find(_classNamesDynamicList.begin(), _classNamesDynamicList.end(), className) != _classNamesDynamicList.end();
}

bool Detector_ONNXRUNTIME::LoadModel(std::string &modelPath)
{
    try {
        std::cout << "modelPath=" << modelPath << std::endl;
        Ort::SessionOptions _OrtSessionOptions = Ort::SessionOptions();
#if CUDA_Availability
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");

        if (cuda_available != available_providers.end()) {
            std::cout << "----- Inference device: CUDA" << std::endl;
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, _cudaID);
        }
        else if (cuda_available == available_providers.end()){
            std::cout << "----- Your ORT build without GPU. Change to CPU." << std::endl;
            std::cout << "----- Inference device: CPU" << std::endl;
        }
#else
        std::cout << "----- Inference device: CPU" << std::endl;
#endif

        _OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
        _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#else
        _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        auto _inputNodesNum = _OrtSession->GetInputCount();
        _inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
        _inputNodeNames.push_back(_inputName.get());

        Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
        auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto _inputNodeDataType = input_tensor_info.GetElementType();
        _inputTensorShape = input_tensor_info.GetShape();

        if (_inputTensorShape[0] == -1)
        {
            _isDynamicShape = true;
            _inputTensorShape[0] = _batchSize;
        }
        if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
            _isDynamicShape = true;
            _inputTensorShape[2] = _inputSize.height;
            _inputTensorShape[3] = _inputSize.width;
        }

        auto _outputNodesNum = _OrtSession->GetOutputCount();
        if (_outputNodesNum != 1) {
            std::cout << "This model has " << _outputNodesNum << "output, which is not a detection model.Please check your model name or path!" << std::endl;
            return false;
        }

        _output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
        _outputNodeNames.push_back(_output_name0.get());

        Ort::TypeInfo type_info_output0(nullptr);
        type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0

        auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
        auto _outputNodeDataType = tensor_info_output0.GetElementType();
        _outputTensorShape = tensor_info_output0.GetShape();
    } catch (const std::exception&) {
        std::cout << "----- Can't load model:" << modelPath << std::endl;
        return false;
    }

    std::cout << "---------- Model is loaded " << std::endl;
    return true;
}

BatchDetectedObject Detector_ONNXRUNTIME::Run(MatVector &srcImgList)
{
    // TODO: just work with bachNumber=1
    if(_batchSize > 1 || srcImgList.size() > 1) {
        std::cout <<"This class just work with batchNumber=1" << std::endl;
        return {};
    }

    BatchDetectedObject batchOutput;

    std::vector<cv::Vec4d> params;
    MatVector input_images;
    //preprocessing
    Preprocessing(srcImgList, input_images, params);
    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, _inputSize, cv::Scalar(0, 0, 0), true, false);

    int64_t input_tensor_length = VectorProduct(_inputTensorShape);
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

    output_tensors = _OrtSession->Run(Ort::RunOptions{nullptr},
                                      _inputNodeNames.data(),
                                      input_tensors.data(),
                                      _inputNodeNames.size(),
                                      _outputNodeNames.data(),
                                      _outputNodeNames.size());

    //post-process
    float* all_data = output_tensors[0].GetTensorMutableData<float>();
    _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
    int net_width = (int)_outputTensorShape[1];

    for (int img_index = 0; img_index < srcImgList.size(); ++img_index) {
        cv::Mat output0 = cv::Mat(cv::Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
        all_data += one_output_length;
        float* pdata = (float*)output0.data;
        int rows = output0.rows;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        for (int r = 0; r < rows; ++r) {
            cv::Mat scores(1, _classNamesList.size(), CV_32F, pdata + 4);
            cv::Point classIdPoint;
            double max_class_socre;
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
            max_class_socre = (float)max_class_socre;
            if (max_class_socre >= _classThreshold) {
                //rect [x,y,w,h]
                float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
                float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
                float w = pdata[2] / params[img_index][0];  //w
                float h = pdata[3] / params[img_index][1];  //h
                int left = MAX(int(x - 0.5 * w + 0.5), 0);
                int top = MAX(int(y - 0.5 * h + 0.5), 0);
                class_ids.push_back(classIdPoint.x);
                confidences.push_back(max_class_socre);
                boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
            }
            pdata += net_width;
        }

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
        std::vector<std::vector<float>> temp_mask_proposals;
        cv::Rect holeImgRect(0, 0, srcImgList[img_index].cols, srcImgList[img_index].rows);
        ImagesDetectedObject imageOutput;
        for (int i = 0; i < nms_result.size(); ++i) {
            int idx = nms_result[i];
            DetectedObject result;
            result.classID = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx] & holeImgRect;
            imageOutput.push_back(result);
        }
        batchOutput.push_back(imageOutput);
    }

    return batchOutput;
}

void Detector_ONNXRUNTIME::Preprocessing(const std::vector<cv::Mat> &SrcImgs, std::vector<cv::Mat> &OutSrcImgs, std::vector<cv::Vec4d> &params)
{
    OutSrcImgs.clear();
    for (int i = 0; i < SrcImgs.size(); ++i) {
        cv::Mat temp_img = SrcImgs[i];
        cv::Vec4d temp_param = { 1,1,0,0 };
        if (temp_img.size() != _inputSize) {
            cv::Mat borderImg;
            LetterBox(temp_img, borderImg, temp_param, _inputSize, false, false, true, 32);
            //cout << borderImg.size() << endl;
            OutSrcImgs.push_back(borderImg);
            params.push_back(temp_param);
        }
        else {
            OutSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }

    int lack_num = SrcImgs.size() % _batchSize;
    if (lack_num != 0) {
        for (int i = 0; i < lack_num; ++i) {
            cv::Mat temp_img = cv::Mat::zeros(_inputSize, CV_8UC3);
            cv::Vec4d temp_param = { 1,1,0,0 };
            OutSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }
}

void Detector_ONNXRUNTIME::LetterBox(const cv::Mat &image, cv::Mat &outImage, cv::Vec4d &params, const cv::Size &newShape, bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar &color)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape) {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1]) {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else {
        outImage = image.clone();
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;

    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}
