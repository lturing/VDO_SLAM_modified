/**
* This file is part of VDO-SLAM.
*
* Copyright (C) 2019-2020 Jun Zhang <jun doc zhang2 at anu dot edu doc au> (The Australian National University)
* For more information see <https://github.com/halajun/VDO_SLAM>
*
**/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<unistd.h>

#include<opencv2/core/core.hpp>
#include<opencv2/optflow.hpp>

#include<System.h>

#include "detector_opencv_dnn.h"
#include "segmentor_opencv_dnn.h"
#include "detector_onnxruntime.h"
#include "segmentor_onnxruntime.h"

#include <vector>
#include <glob.h>
#include <iostream>
#include <algorithm>

using namespace std;

void LoadData(const string &strPathToSequence, 
              vector<string> &vstrFilenamesRGB, vector<string> &vstrFilenamesDEP, vector<string> &vstrFilenamesFLO, vector<string> &vstrFilenamesFLOV,
              vector<double> &vTimestamps);


int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./vdo_slam path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrFilenamesRGB;
    vector<string> vstrFilenamesDEP;
    vector<string> vstrFilenamesFLO;
    vector<string> vstrFilenamesFLOV;
    vector<double> vTimestamps;

    LoadData(argv[2], vstrFilenamesRGB, vstrFilenamesDEP, vstrFilenamesFLO, vstrFilenamesFLOV,
                  vTimestamps);

    // Check consistency in the number of images, depth maps, segmentations and flow maps
    int nImages = vstrFilenamesRGB.size()-1;
    if(vstrFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    VDO_SLAM::System SLAM(argv[1],VDO_SLAM::System::RGBD);

    cv::FileStorage fSettings(argv[1], cv::FileStorage::READ);
    string seg_onnx_path = fSettings["seg_onnx_path"].string();

    cout << endl << "--------------------------------------------------------------------------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // namedWindow( "Trajectory", cv::WINDOW_AUTOSIZE);
    cv::Mat imTraj(1000, 1000, CV_8UC3, cv::Scalar(255,255,255));

    // Main loop
    // (799,0007) (802,0009) (293,0010) (836,0020) (338,0018) (1057,0019) (339,0013)
    // (153,0000)(446,0001)(232,0002)(143,0003)(313,0004)(296,0005)(144,0017)(269,0006)
    cv::Mat imRGB, imD, mTcw_gt;

    std::shared_ptr<Segmentor_ONNXRUNTIME> mySegOnnxRun;
    std::shared_ptr<Segmentor_OpenCV_DNN> mySegOnnxCV;

    int batchSize = 1;
    auto inputSize = cv::Size(640, 640);

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

    std::vector<std::string> dynamicObjectClass = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball"
    };

    mySegOnnxRun = std::make_shared<Segmentor_ONNXRUNTIME>();
    mySegOnnxCV = std::make_shared<Segmentor_OpenCV_DNN>();

    mySegOnnxRun->LoadModel(seg_onnx_path);
    mySegOnnxRun->setClassNames(_classNamesList);
    mySegOnnxRun->setDynamicClassNames(dynamicObjectClass);
    mySegOnnxRun->setBatchSize(batchSize);
    mySegOnnxRun->setInputSize(inputSize);

    mySegOnnxCV->LoadModel(seg_onnx_path);
    mySegOnnxCV->setClassNames(_classNamesList);
    mySegOnnxCV->setDynamicClassNames(dynamicObjectClass);
    mySegOnnxCV->setBatchSize(batchSize);
    mySegOnnxCV->setInputSize(inputSize);

    int StopFrame = nImages-1;
    for(int ni=0; ni<StopFrame; ni++)
    {
        cout << endl;
        cout << "=======================================================" << endl;
        cout << "Processing Frame: " << ni << endl;

        // Read imreadmage and depthmap from file
        imRGB = cv::imread(vstrFilenamesRGB[ni],cv::IMREAD_UNCHANGED);
        imD   = cv::imread(vstrFilenamesDEP[ni],cv::IMREAD_UNCHANGED);
        cv::Mat imD_f, imD_r;

        // // For stereo disparity input
        imD.convertTo(imD_f, CV_32F);
        
        if(imRGB.channels() == 1)
        {
            cv::cvtColor(imRGB, imRGB, cv::COLOR_GRAY2RGB);
        }
        
        std::vector<cv::Mat> imgBatch;
        //imgBatch.push_back(bgr_img);
        imgBatch.push_back(imRGB.clone());

        BatchSegmentedObject result;
        #if USE_ONNX
            result = mySegOnnxRun->Run(imgBatch);
        #else
            result = mySegOnnxCV->Run(imgBatch);
        #endif

        int object_cnt = 1;
        cv::Mat imSem = cv::Mat(imRGB.rows, imRGB.cols, CV_32SC1, cv::Scalar(0));
        vector<vector<float> > vObjPose_gt;
        cv::Mat imDynaSem = imRGB.clone();
        for (int i = 0; i < result[0].size(); ++i)
        {
            int classId = result[0][i].classID;
            std::string className = mySegOnnxRun->getClassName(classId);
            if (mySegOnnxRun->whetherInDynamicClass(className))
            {
                size_t h = i * 6364136223846793005u + 1442695040888963407;
                cv::Scalar standardColor(h & 0xFF, (h >> 4) & 0xFF, (h >> 8) & 0xFF);

                imDynaSem(result[0][i].box).setTo(standardColor, result[0][i].boxMask);
                imSem(result[0][i].box).setTo(cv::Scalar(object_cnt), result[0][i].boxMask);
                vector<float> objBox;
                int x1 = result[0][i].box.x;
                int y1 = result[0][i].box.y;
                int x2 = result[0][i].box.x + result[0][i].box.width;
                int y2 = result[0][i].box.y + result[0][i].box.height;
                
                objBox.push_back(object_cnt);
                objBox.push_back(x1);
                objBox.push_back(y1);
                objBox.push_back(x2);
                objBox.push_back(y2);
                
                vObjPose_gt.push_back(objBox);
                object_cnt++;
            }
        }

        // // For monocular depth input
        // cv::resize(imD, imD_r, cv::Size(1242,375));
        // imD_r.convertTo(imD_f, CV_32F);

        // Load flow matrix
        //std::cout << vstrFilenamesFLO[ni] << std::endl;
        cv::Mat imFlow = cv::readOpticalFlow(vstrFilenamesFLO[ni]);
        // FlowShow(imFlow);

        cv::Mat imFlowV = cv::imread(vstrFilenamesFLOV[ni], cv::IMREAD_UNCHANGED);

        double tframe = vTimestamps[ni];
        //mTcw_gt = vPoseGT[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrFilenamesRGB[ni] << endl;
            return 1;
        }

        bool bIsEnd;
        // time costly
        if (ni == (StopFrame-1) && false)
            bIsEnd = true;
        else 
            bIsEnd = false;

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD_f,imFlow, imFlowV, imSem, imDynaSem, mTcw_gt, vObjPose_gt, tframe, bIsEnd);

    }

    SLAM.shutdown();
    // Save camera trajectory
    // SLAM.SaveResults("/Users/steed/work/code/Evaluation/ijrr2020/omd/omd_results/new/");
    // SLAM.SaveResults("/Users/steed/work/code/Evaluation/ijrr2020/00/new/");

    std::cout << "successful" << std::endl;
    return 0;
}

void LoadData(const string &strPathToSequence, 
              vector<string> &vstrFilenamesRGB,vector<string> &vstrFilenamesDEP, vector<string> &vstrFilenamesFLO,  vector<string> &vstrFilenamesFLOV,
              vector<double> &vTimestamps)
{
    // +++ timestamps +++
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }
    fTimes.close();

    // +++ image, depth, semantic and moving object tracking mask +++
    string strPrefixImage = strPathToSequence + "/data/";         // image  image_0
    string strPrefixDepth = strPathToSequence + "/zoedepth_depth/";           // depth_gt  depth  depth_mono_stereo
    string strPrefixFlow = strPathToSequence + "/videoFlow_flow/";             // flow_gt  flow

    const int nTimes = vTimestamps.size();
    vstrFilenamesRGB.resize(nTimes);
    vstrFilenamesDEP.resize(nTimes);
    vstrFilenamesFLO.resize(nTimes);
    vstrFilenamesFLOV.resize(nTimes);


    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(10) << i;
        vstrFilenamesRGB[i] = strPrefixImage + ss.str() + ".png";
        vstrFilenamesDEP[i] = strPrefixDepth + ss.str() + ".png";

        stringstream ss1;
        stringstream ss2;
        ss1 << setfill('0') << setw(6) << i; 
        ss2 << setfill('0') << setw(6) << (i+1); 
        // flow_0271_to_0272.flo
        vstrFilenamesFLO[i] = strPrefixFlow + "flow_" + ss1.str() + "_to_" + ss2.str() + ".flo";
        //std::cout << "vstrFilenamesFLO[i]=" << vstrFilenamesFLO[i] << std::endl;
        vstrFilenamesFLOV[i] = strPrefixFlow + "flow_" + ss1.str() + "_to_" + ss2.str() + ".png";
    }

}





