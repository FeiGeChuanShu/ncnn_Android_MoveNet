// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NANODET_H
#define NANODET_H

#include <opencv2/core/core.hpp>

#include <net.h>

struct PadInfo
{
    int target_w;
    int target_h;
    int wpad;
    int hpad;
    float scale;
};
struct Keypoint
{
    float x;
    float y;
    float prob;
};

struct Person
{
    std::vector<Keypoint> points;
    int x;
    int y;
    int width;
    int height;
    float score;
};
class NanoDet
{
public:
    NanoDet();
    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals,const float* multipose_scale, bool use_gpu = false);
    int detect(const cv::Mat &rgb, std::vector<Person> &objects);
    int draw(cv::Mat& rgb,std::vector<Person> &objects);
private:
    ncnn::Net poseNet;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
    void postProcess(ncnn::Mat& kpt_regress, ncnn::Mat& center, ncnn::Mat& kpt_heatmap,  ncnn::Mat& kpt_offset,
            ncnn::Mat& center_idx,  ncnn::Mat& box_scale,  ncnn::Mat& box_offset,const std::vector<std::vector<float>>& dist_x,
            const std::vector<std::vector<float>>& dist_y,int feat_w,int feat_h,ncnn::Mat& detect_result);
    void decodeTopkBoxCoord(const ncnn::Mat& box_scale, const ncnn::Mat& box_offset, const std::vector<std::pair<int, int>>& topk,
            ncnn::Mat& box_coords_norm_, std::vector<std::tuple<float, float, float, float>>& box_coords,int feat_w);
    void decodeRegressWithOffset(const ncnn::Mat& kpt_offset, const ncnn::Mat& logit_transpose,ncnn::Mat& scores_transpose,
            int feat_w,int feat_h,std::vector<float>& kpt_regress_scores,std::vector<std::vector<std::tuple<float, float, float>>>& kpt_offsets_all, ncnn::Mat& keypoints);
    void calcTopkKptScores(const ncnn::Mat& kpt_regress,const ncnn::Mat& kpt_heatmap, const std::vector<std::pair<int, int>>& topk,
            const std::vector<std::vector<float>>& dist_x, const std::vector<std::vector<float>>& dist_y,int feat_w,int feat_h, ncnn::Mat& scores);
private:
    int input_w;
    int input_h;
    float mean_vals[3];
    float norm_vals[3];
    float multipose_scale[4];
    float scale;
    std::vector<std::vector<float>> dist_y, dist_x;

};

#endif // NANODET_H
