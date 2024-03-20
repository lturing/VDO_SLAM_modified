#include "segmentor_opencv_dnn.h"
#include <iostream>
#include <string> 
#include <algorithm>

#if CUDA_STATUS
#define CUDA_Availability true
#else
#define CUDA_Availability false
#endif


Segmentor_OpenCV_DNN::Segmentor_OpenCV_DNN() {}

void Segmentor_OpenCV_DNN::setBatchSize(int newBatch)
{
    if (newBatch < 1) newBatch = 1;
    _batchSize = newBatch;
}

void Segmentor_OpenCV_DNN::setInputSize(cv::Size newInputSize)
{
    _inputSize = newInputSize;
}

void Segmentor_OpenCV_DNN::setClassNames(std::vector<std::string> newClassNamesList)
{
    _classNamesList = newClassNamesList;
}

std::string Segmentor_OpenCV_DNN::getClassName(int classId)
{
    return _classNamesList[classId];
}

void Segmentor_OpenCV_DNN::setDynamicClassNames(std::vector<std::string> classNamesDynamicList)
{
    _classNamesDynamicList = classNamesDynamicList;
}

bool Segmentor_OpenCV_DNN::whetherInDynamicClass(std::string className)
{
    return std::find(_classNamesDynamicList.begin(), _classNamesDynamicList.end(), className) != _classNamesDynamicList.end();
}


bool Segmentor_OpenCV_DNN::LoadModel(std::string &modelPath)
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
#else
        std::cout << "----- Inference device: CPU" << std::endl;
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif
    }
    catch (const std::exception&) {
        std::cout << "----- Can't load model:" << modelPath << std::endl;
        return false;
    }

    std::cout << "---------- Model is loaded " << std::endl;
    return true;
}

BatchSegmentedObject Segmentor_OpenCV_DNN::Run(MatVector &srcImgList)
{
    // TODO: just work with bachNumber=1
    if(_batchSize > 1 || srcImgList.size() > 1) {
        std::cout <<"This class just work with batchNumber=1" << std::endl;
        return {};
    }

    BatchSegmentedObject batchOutput;
    ImagesSegmentedObject imageOutput;

    auto srcImg = srcImgList[0];
    cv::Mat netInputImg;
    cv::Vec4d params;
    LetterBox(srcImg, netInputImg, params, _inputSize);

    cv::Mat blob;
    cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, _inputSize, cv::Scalar(0, 0, 0), true, false);
    //************************************
    //If there is no problem with other settings, but results are a lot different from  Python-onnx, you can try to use the following two sentences
    //cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, _inputSize, cv::Scalar(104, 117, 123), true, false);
    //cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, _inputSize, cv::Scalar(114, 114,114), true, false);
    //************************************

    model.setInput(blob);

    std::vector<cv::Mat> net_output_img;
    std::vector<std::string> output_layer_names{ "output0","output1" };
    model.forward(net_output_img, output_layer_names); //get outputs

    std::vector<int> class_ids;// res-class_id
    std::vector<float> confidences;// res-conf
    std::vector<cv::Rect> boxes;// res-box
    std::vector<std::vector<float>> picked_proposals;  //output0[:,:, 4 + _className.size():net_width]===> for mask
    cv::Mat output0 = cv::Mat(cv::Size(net_output_img[0].size[2], net_output_img[0].size[1]), CV_32F, (float*)net_output_img[0].data).t();  //[bs,116,8400]=>[bs,8400,116]
    int rows = output0.rows;
    int net_width = output0.cols;
    float* pdata = (float*)output0.data;

    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, _classNamesList.size(), CV_32FC1, pdata + 4);
        cv::Point classIdPoint;
        double max_class_socre;
        minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
        max_class_socre = (float)max_class_socre;
        if (max_class_socre >= _classThreshold) {
            std::vector<float> temp_proto(pdata + 4 + _classNamesList.size(), pdata + net_width);
            picked_proposals.push_back(temp_proto);
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
        SegmentedObject result;
        result.classID = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx] & holeImgRect;
        temp_mask_proposals.push_back(picked_proposals[idx]);
        imageOutput.push_back(result);
    }

    MaskParams mask_params;
    mask_params.params = params;
    mask_params.srcImgShape = srcImg.size();
    mask_params.netHeight = _inputSize.height;
    mask_params.netWidth = _inputSize.width;
    mask_params.maskThreshold = _maskThreshold;

    //************************************
    for (int i = 0; i < temp_mask_proposals.size(); ++i) {
        GetMask2(cv::Mat(temp_mask_proposals[i]).t(), net_output_img[1], imageOutput[i], mask_params);
    }
    //************************************
    //If the GetMask2() still reports errors , it is recommended to use GetMask().
    //    cv::Mat mask_proposals;
    //    for (int i = 0; i < temp_mask_proposals.size(); ++i) {
    //        mask_proposals.push_back(cv::Mat(temp_mask_proposals[i]).t());
    //    }
    //    GetMask(mask_proposals, net_output_img[1], imageOutput, mask_params);
    //************************************

    calcContours(imageOutput);

    batchOutput.push_back(imageOutput);
    return batchOutput;
}

void Segmentor_OpenCV_DNN::LetterBox(const cv::Mat &image, cv::Mat &outImage, cv::Vec4d &params, const cv::Size &newShape, bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar &color)
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

void Segmentor_OpenCV_DNN::GetMask(const cv::Mat &maskProposals, const cv::Mat &maskProtos, ImagesSegmentedObject &output, const MaskParams &maskParams)
{
    int net_width = maskParams.netWidth;
    int net_height = maskParams.netHeight;
    int seg_channels = maskProtos.size[1];
    int seg_height = maskProtos.size[2];
    int seg_width = maskProtos.size[3];
    float mask_threshold = maskParams.maskThreshold;
    cv::Vec4f params = maskParams.params;
    cv::Size src_img_shape = maskParams.srcImgShape;

    cv::Mat protos = maskProtos.reshape(0, { seg_channels,seg_width * seg_height });

    cv::Mat matmul_res = (maskProposals * protos).t();
    cv::Mat masks = matmul_res.reshape(output.size(), { seg_width,seg_height });
    std::vector<cv::Mat> maskChannels;
    split(masks, maskChannels);
    for (int i = 0; i < output.size(); ++i) {
        cv::Mat dest, mask;
        //sigmoid
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest);

        cv::Rect roi(int(params[2] / net_width * seg_width), int(params[3] / net_height * seg_height), int(seg_width - params[2] / 2), int(seg_height - params[3] / 2));
        dest = dest(roi);
        resize(dest, mask, src_img_shape, cv::INTER_NEAREST);

        //crop
        cv::Rect temp_rect = output[i].box;
        mask = mask(temp_rect) > mask_threshold;
        output[i].boxMask = mask;
    }
}

void Segmentor_OpenCV_DNN::GetMask2(const cv::Mat &maskProposals, const cv::Mat &maskProtos, SegmentedObject &output, const MaskParams &maskParams)
{
    int net_width = maskParams.netWidth;
    int net_height = maskParams.netHeight;
    int seg_channels = maskProtos.size[1];
    int seg_height = maskProtos.size[2];
    int seg_width = maskProtos.size[3];
    float mask_threshold = maskParams.maskThreshold;
    cv::Vec4f params = maskParams.params;
    cv::Size src_img_shape = maskParams.srcImgShape;

    cv::Rect temp_rect = output.box;
    //crop from mask_protos
    int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
    int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
    int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
    int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

    rang_w = MAX(rang_w, 1);
    rang_h = MAX(rang_h, 1);
    if (rang_x + rang_w > seg_width) {
        if (seg_width - rang_x > 0)
            rang_w = seg_width - rang_x;
        else
            rang_x -= 1;
    }
    if (rang_y + rang_h > seg_height) {
        if (seg_height - rang_y > 0)
            rang_h = seg_height - rang_y;
        else
            rang_y -= 1;
    }

    std::vector<cv::Range> roi_rangs;
    roi_rangs.push_back(cv::Range(0, 1));
    roi_rangs.push_back(cv::Range::all());
    roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
    roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

    //crop
    cv::Mat temp_mask_protos = maskProtos(roi_rangs).clone();
    cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
    cv::Mat matmul_res = (maskProposals * protos).t();
    cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
    cv::Mat dest, mask;

    //sigmoid
    cv::exp(-masks_feature, dest);
    dest = 1.0 / (1.0 + dest);

    int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
    int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
    int width = ceil(net_width / seg_width * rang_w / params[0]);
    int height = ceil(net_height / seg_height * rang_h / params[1]);

    cv::resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
    cv::Rect mask_rect = temp_rect - cv::Point(left, top);
    mask_rect &= cv::Rect(0, 0, width, height);
    mask = mask(mask_rect) > mask_threshold;
    output.boxMask = mask;
}

void Segmentor_OpenCV_DNN::calcContours(ImagesSegmentedObject &output)
{
    for (auto &obj : output) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(obj.boxMask, contours, {}, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        auto idx = 0;
        for (auto i = contours.begin(); i != contours.end(); ++i) {
            // scale from box to image
            for(auto &p : contours[idx])
                p = cv::Point(p.x + obj.box.x, p.y + obj.box.y);

            // remove small objects
            if (contours[idx].size() < 10) {
                contours.erase(i);
                i--;
                idx--;
            }
            idx++;
        }

        obj.maskContoursList = contours;
    }
}
