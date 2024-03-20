#ifndef VIEWER_H_
#define VIEWER_H_

#include "Map.h"

#include <thread>
#include <mutex>
#include <pangolin/pangolin.h>
#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

namespace VDO_SLAM {

    using namespace std;
    class Map;
    //  Visualization for DSO

    /**
     * viewer implemented by pangolin
     */
    class Viewer {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Viewer(int w, int h, bool startRunThread = true);

        ~Viewer();

        void run();

        void shutdown();

        void publishPointPoseFrame(Map* pMap, cv::Mat feat_, cv::Mat seg_, cv::Mat flow_);
        void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw);
        // void pushLiveFrame( shared_ptr<Frame> image);

        /* call on finish */
        void join();
        bool isFinished();

    private:

        thread runThread;
        bool running = true;
        int w, h;

        std::vector<cv::Mat> vmCameraPose;
        std::map<int, std::vector<cv::Mat>> objTrajectory;
        std::map<int, std::vector<cv::Mat>> vp3DPointDyn;
        std::vector<cv::Mat> vp3DPointSta; // staic points

        cv::Mat feat;
        cv::Mat seg;
        cv::Mat flow;

        //bool videoImgChanged = true;
        // 3D model rendering
        std::mutex myMutex;

        // timings
        struct timeval last_track;
        struct timeval last_map;

        std::deque<float> lastNTrackingMs;
        std::deque<float> lastNMappingMs;
        bool mbIsFinished;

    };

}

#endif // LDSO_VIEWER_H_