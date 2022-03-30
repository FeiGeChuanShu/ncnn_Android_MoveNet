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

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

const int num_joints = 17;

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}


NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

}

int NanoDet::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    poseNet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    poseNet.opt = ncnn::Option();

#if NCNN_VULKAN
    poseNet.opt.use_vulkan_compute = use_gpu;
#endif

    poseNet.opt.num_threads = ncnn::get_big_cpu_count();
    poseNet.opt.blob_allocator = &blob_pool_allocator;
    poseNet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    poseNet.load_param(parampath);
    poseNet.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int NanoDet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    poseNet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    poseNet.opt = ncnn::Option();

#if NCNN_VULKAN
    poseNet.opt.use_vulkan_compute = use_gpu;
#endif

    poseNet.opt.num_threads = ncnn::get_big_cpu_count();
    poseNet.opt.blob_allocator = &blob_pool_allocator;
    poseNet.opt.workspace_allocator = &workspace_pool_allocator;
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    poseNet.load_param(mgr,parampath);
    poseNet.load_model(mgr,modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];
    
    if(target_size == 192)
    {
        feature_size = 48;
        kpt_scale = 0.02083333395421505;
    }
    else
    {
        feature_size = 64;
        kpt_scale = 0.015625;
    }
    for (int i = 0; i < feature_size; i++)
    {
        std::vector<float> x, y;
        for (int j = 0; j < feature_size; j++)
        {
            x.push_back(j);
            y.push_back(i);
        }
        dist_y.push_back(y);
        dist_x.push_back(x);
    }
    return 0;
}

int NanoDet::detect(const cv::Mat& rgb)
{
    //TODO:add person detection
    return 0;
}

void NanoDet::detect_pose(cv::Mat &rgb, std::vector<keypoint> &points)
{
    int w = rgb.cols;
    int h = rgb.rows;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows, w, h);
    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    ncnn::Extractor ex = poseNet.create_extractor();
    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    
    ex.input("input", in_pad);

    ncnn::Mat regress, center, heatmap, offset;

    ex.extract("regress", regress);
    ex.extract("offset", offset);
    ex.extract("heatmap", heatmap);
    ex.extract("center", center);

    float* center_data = (float*)center.data;
    float* heatmap_data = (float*)heatmap.data;
    float* offset_data = (float*)offset.data;

    int top_index = 0;
    float top_score = 0;

    top_index = int(argmax(center_data, center_data+center.h));
    top_score = *std::max_element(center_data, center_data + center.h);

    int ct_y = (top_index / feature_size);
    int ct_x = top_index - ct_y * feature_size;

    std::vector<float> y_regress(num_joints), x_regress(num_joints);
    float* regress_data = (float*)regress.channel(ct_y).row(ct_x);
    for (size_t i = 0; i < num_joints; i++)
    {
        y_regress[i] = regress_data[i] + (float)ct_y;
        x_regress[i] = regress_data[i + num_joints] + (float)ct_x;
    }

    ncnn::Mat kpt_scores = ncnn::Mat(feature_size * feature_size, num_joints, sizeof(float));
    float* scores_data = (float*)kpt_scores.data;
    for (int i = 0; i < feature_size; i++)
    {
        for (int j = 0; j < feature_size; j++)
        {
            std::vector<float> score;
            for (int c = 0; c < num_joints; c++)
            {
                float y = (dist_y[i][j] - y_regress[c]) * (dist_y[i][j] - y_regress[c]);
                float x = (dist_x[i][j] - x_regress[c]) * (dist_x[i][j] - x_regress[c]);
                float dist_weight = std::sqrt(y + x) + 1.8;
                scores_data[c* feature_size * feature_size +i* feature_size +j] = heatmap_data[i * feature_size * num_joints + j * num_joints + c] / dist_weight;
            }
        }
    }
    std::vector<int> kpts_ys, kpts_xs;
    for (int i = 0; i < num_joints; i++)
    {
        top_index = 0;
        top_score = 0;
        top_index = int(argmax(scores_data + feature_size * feature_size *i, scores_data + feature_size * feature_size *(i+1)));
        top_score = *std::max_element(scores_data + feature_size * feature_size * i, scores_data + feature_size * feature_size * (i + 1));

        int top_y = (top_index / feature_size);
        int top_x = top_index - top_y * feature_size;
        kpts_ys.push_back(top_y);
        kpts_xs.push_back(top_x);
    }

    points.clear();
    for (int i = 0; i < num_joints; i++)
    {
        float kpt_offset_x = offset_data[kpts_ys[i] * feature_size * num_joints*2 + kpts_xs[i] * num_joints * 2 + i * 2];
        float kpt_offset_y = offset_data[kpts_ys[i] * feature_size * num_joints * 2 + kpts_xs[i] * num_joints * 2 + i * 2+1];

        float x = (kpts_xs[i] + kpt_offset_y) * kpt_scale * target_size;
        float y = (kpts_ys[i] + kpt_offset_x) * kpt_scale * target_size;

        keypoint kpt;
        kpt.x = (x - (wpad / 2)) / scale;
        kpt.y = (y - (hpad / 2)) / scale;
        kpt.score = heatmap_data[kpts_ys[i] * feature_size * num_joints + kpts_xs[i] * num_joints + i];
        points.push_back(kpt);

    }

}

int NanoDet::draw(cv::Mat& rgb)
{
    std::vector<keypoint> points;
    detect_pose(rgb,points);

    int skele_index[][2] = { {0,1},{0,2},{1,3},{2,4},{0,5},{0,6},{5,6},{5,7},{7,9},{6,8},{8,10},{11,12},
                                {5,11},{11,13},{13,15},{6,12},{12,14},{14,16} };
    int color_index[][3] = { {255, 0, 0},
        {0, 0, 255},
        {255, 0, 0},
        {0, 0, 255},
        {255, 0, 0},
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
        {255, 0, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 0, 255},
        {0, 0, 255}, };

    for (int i = 0; i < 18; i++)
    {
        if(points[skele_index[i][0]].score > 0.3 && points[skele_index[i][1]].score > 0.3)
            cv::line(rgb, cv::Point(points[skele_index[i][0]].x,points[skele_index[i][0]].y),
                    cv::Point(points[skele_index[i][1]].x,points[skele_index[i][1]].y), cv::Scalar(color_index[i][0], color_index[i][1], color_index[i][2]), 2);
    }
    for (int i = 0; i < num_joints; i++)
    {
        if (points[i].score > 0.3)
            cv::circle(rgb, cv::Point(points[i].x,points[i].y), 3, cv::Scalar(100, 255, 150), -1);
    }
    return 0;
}
