#include "detector_opencv_dnn.h"
#include <algorithm>

#if CUDA_STATUS
#define CUDA_Availability true
#else
#define CUDA_Availability false
#endif

Detector_OpenCV_DNN::Detector_OpenCV_DNN() {
}

void Detector_OpenCV_DNN::setBatchSize(int newBatch)
{
    if (newBatch < 1) newBatch = 1;
    _batchSize = newBatch;
}

void Detector_OpenCV_DNN::setInputSize(cv::Size newInputSize)
{
    _inputSize = newInputSize;
}

void Detector_OpenCV_DNN::setClassNames(std::vector<std::string> newClassNamesList)
{
    _classNamesList = newClassNamesList;
}

std::string Detector_OpenCV_DNN::getClassName(int classId)
{
    return _classNamesList[classId];
}

void Detector_OpenCV_DNN::setDynamicClassNames(std::vector<std::string> classNamesDynamicList)
{
    _classNamesDynamicList = classNamesDynamicList;
}

bool Detector_OpenCV_DNN::whetherInDynamicClass(std::string className)
{
    return std::find(_classNamesDynamicList.begin(), _classNamesDynamicList.end(), className) != _classNamesDynamicList.end();
}


bool Detector_OpenCV_DNN::LoadModel(std::string &modelPath)
{
    std::cout << "modelPath=" << modelPath << std::endl;
#if CUDA_Availability
    std::cout << "----- Founded CUDA device info" << std::endl;
    int cuda_devices_count = cv::cuda::getCudaEnabledDeviceCount();
    for (int dev = 0; dev < cuda_devices_count; ++dev) {
        std::cout << " -------------------------------------------------- " << std::endl;
        cv::cuda::printCudaDeviceInfo(dev);
        std::cout << " -------------------------------------------------- " << std::endl;
    }
#endif

    try
    {
        model = cv::dnn::readNetFromONNX(modelPath);

#if CUDA_Availability
        std::cout << "----- Inference device: CUDA" << std::endl;
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); //DNN_TARGET_CUDA or DNN_TARGET_CUDA_FP16
        std::cout << "---------- Model is loaded " << std::endl;
#else
        std::cout << "----- Inference device: CPU" << std::endl;
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "---------- Model is loaded " << std::endl;
#endif
    }
    catch (const std::exception&) {
        std::cout << "----- Can't load model:" << modelPath << std::endl;
        return false;
    }
    return true;
}

BatchDetectedObject Detector_OpenCV_DNN::Run(MatVector &srcImgList)
{
    // TODO: just work with bachNumber=1
    if(_batchSize > 1 || srcImgList.size() > 1) {
        std::cout <<"This class just work with batchNumber=1" << std::endl;
        return {};
    }

    BatchDetectedObject batchOutput;
    ImagesDetectedObject imageOutput;

    auto srcImg = srcImgList[0];
    cv::Mat netInputImg;
    cv::Vec4d params;
    LetterBox(srcImg, netInputImg, params, _inputSize);

    cv::Mat blob;
    cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, _inputSize, cv::Scalar(0, 0, 0), true, false);
    //************************************
    // If there is no problem with other settings, but results are a lot different from  Python-onnx , you can try to use the following two sentences
    //cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, _inputSize, cv::Scalar(104, 117, 123), true, false);
    //cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, _inputSize, cv::Scalar(114, 114,114), true, false);
    //************************************
    model.setInput(blob);

    std::vector<cv::Mat> net_output_img;
    model.forward(net_output_img, model.getUnconnectedOutLayersNames()); //get outputs

    std::vector<int> class_ids;// res-class_id
    std::vector<float> confidences;// res-conf
    std::vector<cv::Rect> boxes;// res-box
    cv::Mat output0=cv::Mat(cv::Size(net_output_img[0].size[2], net_output_img[0].size[1]), CV_32F, (float*)net_output_img[0].data).t();  //[bs,116,8400]=>[bs,8400,116]
    int net_width = output0.cols;
    int rows = output0.rows;
    float* pdata = (float*)output0.data;

    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, _classNamesList.size(), CV_32FC1, pdata + 4);
        cv::Point classIdPoint;
        double max_class_socre;
        minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
        max_class_socre = (float)max_class_socre;
        if (max_class_socre >= _classThreshold) {
            //rect [x,y,w,h]
            float x = (pdata[0] - params[2]) / params[0];
            float y = (pdata[1] - params[3]) / params[1];
            float w = pdata[2] / params[0];
            float h = pdata[3] / params[1];
            int left = MAX(int(x - 0.5 * w + 0.5), 0);
            int top = MAX(int(y - 0.5 * h + 0.5), 0);
            class_ids.push_back(classIdPoint.x);
            confidences.push_back(max_class_socre);
            boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
        }
        pdata += net_width;//next line
    }
    //NMS
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
    std::vector<std::vector<float>> temp_mask_proposals;
    cv::Rect holeImgRect(0, 0, srcImg.cols, srcImg.rows);
    for (int i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];
        DetectedObject result;
        result.classID = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx] & holeImgRect;
        imageOutput.push_back(result);
    }

    batchOutput.push_back(imageOutput);
    return batchOutput;
}

void Detector_OpenCV_DNN::LetterBox(const cv::Mat &image, cv::Mat &outImage, cv::Vec4d &params, const cv::Size &newShape, bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar &color)
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
