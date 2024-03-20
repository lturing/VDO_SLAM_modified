#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace cv;
using namespace std;

namespace byte_track
{
enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    STrack(vector<float> tlwh_, float score, int classID=-1);
    ~STrack();

    vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
    void static multi_predict(vector<std::shared_ptr<STrack>> &stracks, KalmanFilter &kalman_filter);
    void static_tlwh();
    void static_tlbr();
    vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
    vector<float> to_xyah();
    void mark_lost();
    void mark_removed();
    int next_id();
    int end_frame();
    
    void activate(KalmanFilter &kalman_filter, int frame_id);
    void re_activate(std::shared_ptr<STrack>& new_track, int frame_id, bool new_id = false);
    void update(std::shared_ptr<STrack>& new_track, int frame_id);

public:
    bool is_activated;
    int track_id;
    int state;

    vector<float> _tlwh;
    vector<float> tlwh;
    vector<float> tlbr;
    int frame_id;
    int tracklet_len;
    int start_frame;

    int classID;

    KAL_MEAN mean;
    KAL_COVA covariance;
    float score;

private:
    KalmanFilter kalman_filter;
};

//using STrackPtr = std::shared_ptr<STrack>;
typedef std::shared_ptr<STrack> STrackPtr;
}