#include <thread>
#include <pangolin/pangolin.h>
#include <unistd.h>

#include "Viewer.h"
#include "Converter.h"

using namespace std;
namespace VDO_SLAM {

    // =================================================================================

    Viewer::Viewer(int w, int h, bool startRunThread) {

        this->w = w;
        this->h = h;
        running = true;
        mbIsFinished = false;

        if (startRunThread)
            runThread = std::thread(&Viewer::run, this);
    }

    Viewer::~Viewer() {
        //shutdown();
        if (runThread.joinable()) {
            runThread.join();
        }
    }

    bool Viewer::isFinished() {
        return mbIsFinished;
    }

    void Viewer::run() {
        
        int hh = h * 3;
        int ww = w * 1.2;

        pangolin::CreateWindowAndBind("Main", ww, hh);
        std::cout << "Create Pangolin viewer" << std::endl;
        const int UI_WIDTH = 170;

        glEnable(GL_DEPTH_TEST);

        // 3D visualization
        pangolin::OpenGlRenderState Visualization3D_camera(
            pangolin::ProjectionMatrix(ww, hh, 2000, 2000, ww / 2, hh / 2, 0.1, 1000),
            pangolin::ModelViewLookAt(-0, -100.0, -0.1, 0, 0, 0, pangolin::AxisNegY)
        );

        pangolin::View &Visualization3D_display = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -ww / (float) hh)
            .SetHandler(new pangolin::Handler3D(Visualization3D_camera));
        
        float x_right = 0.7;

        pangolin::View &feat_video = pangolin::Display("featVideo")
            .SetBounds(2/3.0f, 1.0f, pangolin::Attach::Pix(UI_WIDTH), x_right, w / (float) h);
            //.SetLock(pangolin::LockLeft, pangolin::LockBottom);
        pangolin::GlTexture feat_texVideo(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
        
        pangolin::View &seg_video = pangolin::Display("segVideo")
            .SetBounds(1/3.0f, 2/3.0f, pangolin::Attach::Pix(UI_WIDTH), x_right, w / (float) h);
            //.SetLock(pangolin::LockLeft, pangolin::LockBottom);
            
        pangolin::GlTexture seg_texVideo(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

        pangolin::View &flow_video = pangolin::Display("flowVideo")
            .SetBounds(0.0f, 1/3.0f, pangolin::Attach::Pix(UI_WIDTH), x_right, w / (float) h);
            //.SetLock(pangolin::LockLeft, pangolin::LockBottom);
        pangolin::GlTexture flow_texVideo(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
        
        /*
        pangolin::View &feat_video = pangolin::Display("featVideo")
            .SetAspect(w / (float) h);
        pangolin::GlTexture feat_texVideo(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
        
        pangolin::View &seg_video = pangolin::Display("segVideo")
            .SetAspect(w / (float) h);
            
        pangolin::GlTexture seg_texVideo(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

        pangolin::View &flow_video = pangolin::Display("flowVideo")
            .SetAspect(w / (float) h);
        pangolin::GlTexture flow_texVideo(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

        pangolin::CreateDisplay()
            .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(feat_video)
            .AddDisplay(seg_video)
            .AddDisplay(flow_video)
            .SetLock(pangolin::LockLeft, pangolin::LockTop);;
        */

        // parameter reconfigure gui
        pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

        pangolin::Var<bool> settings_showFeatVideo("ui.showFeatVideo", true, true);
        pangolin::Var<bool> settings_showSegVideo("ui.showSegVideo", true, true);
        pangolin::Var<bool> settings_showFlowVideo("ui.showFlowVideo", true, true);

        pangolin::Var<bool> settings_followCamera("ui.followCamera", true, true);
        pangolin::Var<bool> settings_showTrajectory("ui.showTrajectory", true, true);
        pangolin::Var<bool> settings_showObjTrajectory("ui.showObjTrajectory", false, true);

        pangolin::Var<bool> settings_showPoints("ui.show3DPoint", true, true);
        pangolin::Var<bool> settings_showObjPoints("ui.showObj3DPoint", true, true);
        

        // Default hooks for exiting (Esc) and fullscreen (tab).
        std::cout << "Looping viewer thread" << std::endl;
        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
        Ow.SetIdentity();

        while (!pangolin::ShouldQuit() && running) {
            // Clear entire screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

            {
                unique_lock<std::mutex> lk3d(myMutex);
                {
                    // allFramePoses
                    //if (allFramePoses.size() > 0) Visualization3D_camera.Follow();
                    
                    Visualization3D_display.Activate(Visualization3D_camera);

                    GetCurrentOpenGLCameraMatrix(Twc, Ow);
                    // draw camera
                    {
                        //const float &w = mCameraSize;
                        //const float h = w*0.75;
                        //const float z = w*0.6;

                        float w = 0.5, h = 0.75, z = 0.8;
                        h = w * h * 1.5;
                        z = w * h * 1.5;

                        w *= 2.0;
                        h *= 2.0;
                        z *= 2.0;

                        glPushMatrix();

                        #ifdef HAVE_GLES
                                glMultMatrixf(Twc.m);
                        #else
                                glMultMatrixd(Twc.m);
                        #endif

                        glLineWidth(3);
                        glColor3f(0.0f,1.0f,0.0f);
                        glBegin(GL_LINES);
                        glVertex3f(0,0,0);
                        glVertex3f(w,h,z);
                        glVertex3f(0,0,0);
                        glVertex3f(w,-h,z);
                        glVertex3f(0,0,0);
                        glVertex3f(-w,-h,z);
                        glVertex3f(0,0,0);
                        glVertex3f(-w,h,z);

                        glVertex3f(w,h,z);
                        glVertex3f(w,-h,z);

                        glVertex3f(-w,h,z);
                        glVertex3f(-w,-h,z);

                        glVertex3f(-w,h,z);
                        glVertex3f(w,h,z);

                        glVertex3f(-w,-h,z);
                        glVertex3f(w,-h,z);
                        glEnd();

                        glPopMatrix();

                    }

                    // trajectory
                    if (settings_showTrajectory)
                    {
                        if (settings_followCamera)
                        {
                            Visualization3D_camera.Follow(Twc);
                        }
                        
                        float colorRed[3] = {1, 0, 0};
                        glColor3f(colorRed[0], colorRed[1], colorRed[2]);
                        glLineWidth(3);
                        glBegin(GL_LINE_STRIP);
                        for (unsigned int i = 0; i < vmCameraPose.size(); i++) {
                            cv::Mat position = vmCameraPose[i].rowRange(0,3).col(3);
                            glVertex3f(position.at<float>(0), position.at<float>(1), position.at<float>(2));
                        }
                        glEnd(); 
                    }

                    if (settings_showPoints)
                    {
                        float pointSize = 0.2;
                        glPointSize(pointSize);
                        glBegin(GL_POINTS);
                        glColor3f(0.0,0.0,0.0);

                        for(size_t i=0, iend=vp3DPointSta.size(); i<iend;i++)
                        {
                            cv::Mat pos = vp3DPointSta[i];
                            //if (pos.rows==3)
                            {
                                glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
                            }
                        }
                        glEnd();
                    }

                    if (settings_showObjPoints)
                    {
                        for (int i = 0; i < vp3DPointDyn.size(); i ++)
                        {
                            size_t h = i * 6364136223846793005u + 1442695040888963407;
                            cv::Scalar standardColor(h & 0xFF, (h >> 4) & 0xFF, (h >> 8) & 0xFF);
                            float r = (h & 0xFF) / 255.;
                            float g = ((h >> 4) & 0xFF) / 255.;
                            float b = ((h >> 8) & 0xFF) / 255.;

                            float pointSize = 0.2;
                            glPointSize(pointSize);
                            glBegin(GL_POINTS);
                            glColor3f(b, g, r);
                            for (int j = 0; j < vp3DPointDyn[i].size(); j++)
                            {
                                cv::Mat pos = vp3DPointDyn[i][j];
                                glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
                            }
                            glEnd();
                        }
                    }

                    if (settings_showObjTrajectory)
                    {
                        for (int i = 0; i < objTrajectory.size(); i++)
                        {
                            size_t h = i * 6364136223846793005u + 1442695040888963407;
                            cv::Scalar standardColor(h & 0xFF, (h >> 4) & 0xFF, (h >> 8) & 0xFF);
                            float r = (h & 0xFF) / 255.;
                            float g = ((h >> 4) & 0xFF) / 255.;
                            float b = ((h >> 8) & 0xFF) / 255.;

                            glColor3f(b, g, r);
                            glLineWidth(3);
                            glBegin(GL_LINE_STRIP);
                            for (unsigned int j = 0; j < objTrajectory[i].size(); i++) {
                                cv::Mat pos = objTrajectory[i][j];
                                if (pos.rows==3)
                                    glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
                            }
                            glEnd(); 
                        }
                    }

                }
                
                if (settings_showFeatVideo) {
                    // https://github.com/stevenlovegrove/Pangolin/issues/682
                    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
                    feat_texVideo.Upload(feat.data, GL_BGR, GL_UNSIGNED_BYTE);
                    feat_video.Activate();
                    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                    feat_texVideo.RenderToViewportFlipY();
                }

                if (settings_showSegVideo) {
                    // https://github.com/stevenlovegrove/Pangolin/issues/682
                    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
                    seg_texVideo.Upload(seg.data, GL_BGR, GL_UNSIGNED_BYTE);
                    seg_video.Activate();
                    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                    seg_texVideo.RenderToViewportFlipY();
                }

                if (settings_showFlowVideo) {
                    // https://github.com/stevenlovegrove/Pangolin/issues/682
                    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
                    flow_texVideo.Upload(flow.data, GL_BGR, GL_UNSIGNED_BYTE);
                    flow_video.Activate();
                    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                    flow_texVideo.RenderToViewportFlipY();
                }
            }
            // Swap frames and Process Events
            pangolin::FinishFrame();

            usleep(5000);
        }

        std::cout << "QUIT Pangolin thread!" << std::endl;
        mbIsFinished = true;
    }

    void Viewer::shutdown() {
        std::cout << "start to stop pangolin" << std::endl;
        running = false;
    }

    void Viewer::join() {
        runThread.join();
        std::cout << "JOINED Pangolin thread!" << std::endl;
    }

    void Viewer::publishPointPoseFrame(Map* pMap, cv::Mat feat_, cv::Mat seg_, cv::Mat flow_) {
        unique_lock<std::mutex> lk3d(myMutex);
        feat = feat_.clone();
        seg = seg_.clone();
        flow = flow_.clone();
        vmCameraPose = pMap->vmCameraPose;

        int current_frame_id = vmCameraPose.size() - 1;

        //std::cout << "publishPointPoseFrame1" << std::endl;
        // get static 3d point
        std::vector<std::vector<std::pair<int, int> > > StaTracks = pMap->TrackletSta;
        vp3DPointSta.clear();
        //vp3DPointSta.resize(StaTracks.size());
        for (int i = 0; i < StaTracks.size(); ++i)
        {
            // filter the tracklets via threshold
            int track_len = StaTracks[i].size();
            if (track_len<4) // 3 the length of track on background.
                continue;
            
            int frameId = StaTracks[i][track_len-1].first;
            int featId = StaTracks[i][track_len-1].second;

            cv::Mat point3d_w = pMap->vp3DPointSta[frameId][featId];
            if (point3d_w.rows == 3)
                vp3DPointSta.push_back(point3d_w.clone());
        }
        
        //std::cout << "publishPointPoseFrame2" << std::endl;
        // get dynamic 3d point and trajectory
        std::vector<std::vector<std::pair<int, int> > > DynTracks = pMap->TrackletDyn;
        vp3DPointDyn.clear();
        //return;
        std::map<int, int> object_update;
        for (int i = 0; i < DynTracks.size(); ++i)
        {
            // filter the tracklets via threshold
            int track_len = DynTracks[i].size();
            if (track_len<4) // 3 the length of track on objects.
                continue;
            
            // vnFeaLabDyn[DynTracks[i][j].first][DynTracks[i][j].second]
            int frameId = DynTracks[i][track_len-1].first;
            int featId = DynTracks[i][track_len-1].second;
            int objectId = pMap->nObjID[i];

            // only the latest object
            if (frameId < current_frame_id)
                continue;

            cv::Mat point3d_w = pMap->vp3DPointDyn[frameId][featId];
            if (point3d_w.rows==3)
            {
                vp3DPointDyn[objectId].push_back(point3d_w.clone());
                //if (frameId == current_frame_id)
                object_update[frameId] = featId;
            }
            
        }

        for (int objId = 0; objId < vp3DPointDyn.size(); objId++)
        {
            cv::Mat ObjCentre3D = (cv::Mat_<float>(3,1) << 0.f, 0.f, 0.f);
            for (int i = 0; i < vp3DPointDyn[objId].size(); ++i)
            {
                ObjCentre3D = ObjCentre3D + vp3DPointDyn[objId][i];
            }
            ObjCentre3D = ObjCentre3D/vp3DPointDyn[objId].size();
            objTrajectory[objId].push_back(ObjCentre3D);

        }

        pMap->vmRigidMotion; // from 1 to n

        pMap->vnFeatLabel; // 点所属的物体, from 1
        pMap->vnRMLabel; // 物体label
        pMap->vbObjStat; // 物体的状态

    }

    void Viewer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
    {
        if(vmCameraPose.size() == 0)
            return; 
        Eigen::Matrix4f Twc;
        {
            //unique_lock<mutex> lock(myMutex); // dead lock
            Twc = Converter::toMatrix4d(vmCameraPose.back()).cast<float>();
        }

        for (int i = 0; i<4; i++) {
            M.m[4*i] = Twc(0,i);
            M.m[4*i+1] = Twc(1,i);
            M.m[4*i+2] = Twc(2,i);
            M.m[4*i+3] = Twc(3,i);
        }

        MOw.SetIdentity();
        MOw.m[12] = Twc(0,3);
        MOw.m[13] = Twc(1,3);
        MOw.m[14] = Twc(2,3);
    }

}