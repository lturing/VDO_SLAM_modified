#pragma once

#include "STrack.h"
#include "data_struct.h"
#include <Eigen/Core>
#include <Eigen/Dense>

namespace byte_track
{

class BYTETracker
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    BYTETracker(int frame_rate = 30, int track_buffer = 30);
    ~BYTETracker();

    vector<STrackPtr> update(const ImagesSegmentedObject& objects);
    Scalar get_color(int idx);

private:
    vector<STrackPtr> joint_stracks(vector<STrackPtr> &tlista, vector<STrackPtr> &tlistb);

    vector<STrackPtr> sub_stracks(vector<STrackPtr> &tlista, vector<STrackPtr> &tlistb);
    void remove_duplicate_stracks(vector<STrackPtr> &resa, vector<STrackPtr> &resb, vector<STrackPtr> &stracksa, vector<STrackPtr> &stracksb);

    void linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
        vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
    vector<vector<float> > iou_distance(vector<STrackPtr> &atracks, vector<STrackPtr> &btracks, int &dist_size, int &dist_size_size);
    vector<vector<float> > iou_distance(vector<STrackPtr> &atracks, vector<STrackPtr> &btracks);
    vector<vector<float> > ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs);

    double lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol, 
        bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

    float track_thresh;
    float high_thresh;
    float match_thresh;
    int frame_id;
    int max_time_lost;

    vector<STrackPtr> tracked_stracks;
    vector<STrackPtr> lost_stracks;
    vector<STrackPtr> removed_stracks;
    KalmanFilter kalman_filter;
};
}