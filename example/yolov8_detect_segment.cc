#include "detector_opencv_dnn.h"
#include "segmentor_opencv_dnn.h"
#include "detector_onnxruntime.h"
#include "segmentor_onnxruntime.h"


int main(int argc, char *argv[])
{
    std::vector<std::string> _classNamesList = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    auto imgPathList = {
        "Examples/data/0000_rgb_raw.jpg",
        "Examples/data/kitti_2011_10_03_0000000168.png",
        "Examples/data/bus.jpg",
        "Examples/data/zidane.jpg"
    };
    std::vector<cv::Mat> imgList;
    for(auto imgPath : imgPathList){
        auto img = cv::imread(imgPath);
        imgList.push_back(img);
    }

    auto batchSize = 1;
    auto inputSize = cv::Size(640, 640);

    //--------------------------------------------------Detector
    {
        Detector_ONNXRUNTIME* myDetectorOnnxRun = new Detector_ONNXRUNTIME();
        Detector_OpenCV_DNN* myDetectorOnnxCV = new Detector_OpenCV_DNN();

        std::string modelPath = "/home/spurs/x/yolov8/yolov8s.onnx";
        
        myDetectorOnnxRun->LoadModel(modelPath);
        myDetectorOnnxRun->setClassNames(_classNamesList);
        myDetectorOnnxRun->setBatchSize(batchSize);
        myDetectorOnnxRun->setInputSize(inputSize);

        myDetectorOnnxCV->LoadModel(modelPath);
        myDetectorOnnxCV->setClassNames(_classNamesList);
        myDetectorOnnxCV->setBatchSize(batchSize);
        myDetectorOnnxCV->setInputSize(inputSize);

        for (int imgIDX = 0; imgIDX < imgList.size(); ++imgIDX) {
            // make batch of images = 1
            std::vector<cv::Mat> imgBatch;
            //imgBatch.clear();
            imgBatch.push_back(imgList[imgIDX]);

            auto result1 = myDetectorOnnxRun->Run(imgBatch);
            auto result2 = myDetectorOnnxCV->Run(imgBatch);

            auto img = imgList[imgIDX];
            auto color_box = cv::Scalar(0, 0, 255);
            cv::Mat boxImg1 = img.clone();
            for (int i = 0; i < result1[0].size(); ++i) {
                cv::rectangle(boxImg1, result1[0][i].box, color_box, 2, 8);
                cv::putText(boxImg1, _classNamesList[result1[0][i].classID],
                                    cv::Point(result1[0][i].box.x, result1[0][i].box.y),
                                   cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(0,255,0), 1.0);
            }
            cv::imshow("Detection 1", boxImg1);

            cv::Mat boxImg2 = img.clone();
            for (int i = 0; i < result2[0].size(); ++i) {
                cv::rectangle(boxImg2, result2[0][i].box, color_box, 2, 8);
                cv::putText(boxImg2, _classNamesList[result2[0][i].classID],
                                    cv::Point(result2[0][i].box.x, result2[0][i].box.y),
                                   cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(0,255,0), 1.0);
            }
            cv::imshow("Detection 2", boxImg2);

            cv::waitKey(0);
        }
    }

    //--------------------------------------------------Segmentor
    {
        std::string modelPath = "/home/spurs/x/yolov8/yolov8s-seg.onnx";
        Segmentor_ONNXRUNTIME* mySegOnnxRun = new Segmentor_ONNXRUNTIME();
        Segmentor_OpenCV_DNN* mySegOnnxCV = new Segmentor_OpenCV_DNN();

        mySegOnnxRun->LoadModel(modelPath);
        mySegOnnxRun->setClassNames(_classNamesList);
        mySegOnnxRun->setBatchSize(batchSize);
        mySegOnnxRun->setInputSize(inputSize);

        mySegOnnxCV->LoadModel(modelPath);
        mySegOnnxCV->setClassNames(_classNamesList);
        mySegOnnxCV->setBatchSize(batchSize);
        mySegOnnxCV->setInputSize(inputSize);

        std::vector<cv::Mat> imgBatch;
        for (int imgIDX = 0; imgIDX < imgList.size(); ++imgIDX) {
            // make batch of images = 1
            imgBatch.clear();
            imgBatch.push_back(imgList[imgIDX]);

            auto result1 = mySegOnnxRun->Run(imgBatch);
            auto result2 = mySegOnnxCV->Run(imgBatch);

            auto img = imgList[imgIDX];
            auto color_box = cv::Scalar(0, 0, 255);
            auto color_mask = cv::Scalar(0, 255, 0);
            auto color_contours = cv::Scalar(255, 0, 0);

            cv::Mat maskImg1 = img.clone();
            cv::Mat boxImg1 = img.clone();
            cv::Mat contoursImg1 = img.clone();
            for (int i = 0; i < result1[0].size(); ++i) {
                maskImg1(result1[0][i].box).setTo(color_mask, result1[0][i].boxMask);

                cv::rectangle(boxImg1, result1[0][i].box, color_box, 2, 8);
                cv::putText(boxImg1, _classNamesList[result1[0][i].classID],
                            cv::Point(result1[0][i].box.x, result1[0][i].box.y),
                            cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(0,255,0), 1.0);

                for(size_t c = 0; c< result1[0][i].maskContoursList.size(); c++)
                    drawContours(contoursImg1, result1[0][i].maskContoursList, (int)c, color_contours, 2, cv::LINE_8, {}, 0 );
            }
            cv::imshow("Segmentation Mask1", maskImg1);
            cv::imshow("Segmentation Box1", boxImg1);
            cv::imshow("Segmentation Contours1", contoursImg1);

            cv::Mat maskImg2 = img.clone();
            cv::Mat boxImg2 = img.clone();
            cv::Mat contoursImg2 = img.clone();
            for (int i = 0; i < result2[0].size(); ++i) {
                maskImg2(result2[0][i].box).setTo(color_mask, result2[0][i].boxMask);
                cv::rectangle(boxImg2, result2[0][i].box, color_box, 2, 8);
                cv::putText(boxImg2, _classNamesList[result2[0][i].classID],
                            cv::Point(result2[0][i].box.x, result2[0][i].box.y),
                            cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(0,255,0), 1.0);

                for(size_t c = 0; c< result2[0][i].maskContoursList.size(); c++)
                    drawContours(contoursImg2, result2[0][i].maskContoursList, (int)c, color_contours, 2, cv::LINE_8, {}, 0 );
            }
            cv::imshow("Segmentation Mask2", maskImg2);
            cv::imshow("Segmentation Box2", boxImg2);
            cv::imshow("Segmentation Contours2", contoursImg2);

            cv::waitKey(0);
        
        }
        
    }

}
