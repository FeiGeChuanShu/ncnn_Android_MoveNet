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

const int max_persons = 6;
const int num_joints = 17;
template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

static void reduction_max(const ncnn::Mat& bottom,  ncnn::Mat& top)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Reduction");

    ncnn::Mat axes = ncnn::Mat(1);
    int* axes_ptr = (int*)axes.data;
    axes_ptr[0] = (int)2;
    //printf("%d\n", axes_ptr[0]);
    // set param
    ncnn::ParamDict pd;
    pd.set(0, 4);//
    pd.set(1, 0);//
    pd.set(3, axes);
    pd.set(5, 1);
    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(bottom, top, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void binaryOp_mul(ncnn::Mat& A, ncnn::Mat& B, ncnn::Mat& c)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("BinaryOp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);//

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = A;
    bottoms[1] = B;
    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);
    c = std::move(tops[0]);


    op->destroy_pipeline(opt);

    delete op;
}
static void concat(const std::vector<ncnn::Mat>& inputs, int axis, ncnn::Mat& c)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Concat");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, axis);// axis

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    std::vector<ncnn::Mat> bottoms(inputs.size());
    for(int i = 0; i < inputs.size(); i++)
        bottoms[i] = inputs[i];

    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);

    c = tops[0];

    op->destroy_pipeline(opt);

    delete op;
}
static void transpose(const ncnn::Mat& in, ncnn::Mat& out, const int& order_type)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = true;

    ncnn::Layer* op = ncnn::create_layer("Permute");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, order_type);// order_type

    op->load_param(pd);

    op->create_pipeline(opt);

    ncnn::Mat in_packed = in;
    {
        // resolve dst_elempack
        int dims = in.dims;
        int elemcount = 0;
        if (dims == 1) elemcount = in.elempack * in.w;
        if (dims == 2) elemcount = in.elempack * in.h;
        if (dims == 3) elemcount = in.elempack * in.c;

        int dst_elempack = 1;
        if (op->support_packing)
        {
            if (elemcount % 8 == 0)
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
        }

        if (in.elempack != dst_elempack)
        {
            convert_packing(in, in_packed, dst_elempack, opt);
        }
    }

    // forward
    op->forward(in_packed, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void decodeKeypoints(const ncnn::Mat& detect_result, std::vector<Person>& persons, const PadInfo& pad_info, float box_threshold)
{
    float* detect_result_data = (float*)detect_result.data;
    for (int i = 0; i < detect_result.h; i++)
    {
        if (detect_result_data[i * detect_result.w + 55] < box_threshold)
            continue;
        Person person;
        std::vector<Keypoint> keypoints;
        for (int j = 0; j < 17; j++)
        {
            Keypoint keypoint;
            float y = (detect_result_data[i * detect_result.w + j * 3] * pad_info.target_h - (pad_info.hpad / 2)) / pad_info.scale;
            float x = (detect_result_data[i * detect_result.w + j * 3 + 1] * pad_info.target_w - (pad_info.wpad / 2)) / pad_info.scale;
            float prob = detect_result_data[i * detect_result.w + j * 3 + 2];
            keypoint.x = x;
            keypoint.y = y;
            keypoint.prob = prob;
            keypoints.push_back(std::move(keypoint));
        }
        person.points = std::move(keypoints);
        int y_min = static_cast<int>((detect_result_data[i * detect_result.w + 51] * pad_info.target_h - (pad_info.hpad / 2)) / pad_info.scale);
        int x_min = static_cast<int>((detect_result_data[i * detect_result.w + 52] * pad_info.target_w - (pad_info.wpad / 2)) / pad_info.scale);
        int y_max = static_cast<int>((detect_result_data[i * detect_result.w + 53] * pad_info.target_h - (pad_info.hpad / 2)) / pad_info.scale);
        int x_max = static_cast<int>((detect_result_data[i * detect_result.w + 54] * pad_info.target_w - (pad_info.wpad / 2)) / pad_info.scale);
        person.x = x_min;
        person.y = y_min;
        person.width = x_max - x_min;
        person.height = y_max - y_min;
        person.score = detect_result_data[i * detect_result.w + 55];
        persons.push_back(std::move(person));
    }
}

void NanoDet::calcTopkKptScores(const ncnn::Mat& kpt_regress,const ncnn::Mat& kpt_heatmap, const std::vector<std::pair<int, int>>& topk,
        const std::vector<std::vector<float>>& dist_x, const std::vector<std::vector<float>>& dist_y,int feat_w,int feat_h, ncnn::Mat& scores)
{
    float* kpt_heatmap_data = (float*)kpt_heatmap.data;
    float* kpt_regress_data = (float*)kpt_regress.data;
    std::vector<std::vector<cv::Point2f>> kpt_regress_coords;
    ncnn::Mat kpt_regress_coords_ = ncnn::Mat(2, 17, 6, sizeof(float));
    float* kpt_regress_coords_data = (float*)kpt_regress_coords_.data;
    for (int i = 0; i < topk.size(); i++)
    {
        std::vector<cv::Point2f> kpt_regress_coord;
        for (int j = 0; j < kpt_regress.w / 2; j++)
        {
            float x = kpt_regress_data[topk[i].first * kpt_regress.h * kpt_regress.w + topk[i].second * kpt_regress.w + j * 2];
            float y = kpt_regress_data[topk[i].first * kpt_regress.h * kpt_regress.w + topk[i].second * kpt_regress.w + j * 2 + 1];
            x += topk[i].first;
            y += topk[i].second;
            kpt_regress_coord.push_back(cv::Point2f(x, y));
            kpt_regress_coords_data[i * kpt_regress.w + 2 * j] = x;
            kpt_regress_coords_data[i * kpt_regress.w + 2 * j + 1] = y;
        }
        kpt_regress_coords.push_back(kpt_regress_coord);
    }


    for (int i = 0; i < feat_h; i++)
    {
        float* channel_ptr = scores.channel(i);
        for (int j = 0; j < feat_w; j++)
        {
            float* depth_ptr = channel_ptr + j * max_persons * num_joints;
            for (int n = 0; n < max_persons; n++)
            {
                float* ptr = depth_ptr + n * num_joints;
                for (int c = 0; c < num_joints; c++)
                {
                    float y = (dist_y[i][j] - kpt_regress_coords[n][c].x) * (dist_y[i][j] - kpt_regress_coords[n][c].x);
                    float x = (dist_x[i][j] - kpt_regress_coords[n][c].y) * (dist_x[i][j] - kpt_regress_coords[n][c].y);
                    float dist_weight = (y + x) * scale;
                    ptr[c] = std::exp(dist_weight) * kpt_heatmap_data[i * feat_w * num_joints + j * num_joints + c];
                }
            }
        }
    }
}

void NanoDet::decodeTopkBoxCoord(const ncnn::Mat& box_scale, const ncnn::Mat& box_offset, const std::vector<std::pair<int, int>>& topk,
        ncnn::Mat& box_coords_norm_, std::vector<std::tuple<float, float, float, float>>& box_coords,int feat_w)
{
    const float* box_scale_data = (float*)box_scale.data;
    const float* box_offset_data = (float*)box_offset.data;
    std::vector<std::pair<float, float>> box_scales, box_offsets;
    for (int i = 0; i < topk.size(); i++)
    {
        float scale_x = box_scale_data[topk[i].first * box_scale.h * box_scale.w + topk[i].second * box_scale.w];
        float scale_y = box_scale_data[topk[i].first * box_scale.h * box_scale.w + topk[i].second * box_scale.w + 1];
        box_scales.push_back(std::pair<float, float>(scale_x > 0 ? scale_x : 0, scale_y > 0 ? scale_y : 0));
        float offset_x = box_offset_data[topk[i].first * box_offset.h * box_offset.w + topk[i].second * box_offset.w];
        float offset_y = box_offset_data[topk[i].first * box_offset.h * box_offset.w + topk[i].second * box_offset.w + 1];
        box_offsets.push_back(std::pair<float, float>(offset_x, offset_y));
    }

    //std::vector<std::tuple<float, float, float, float>> box_coords;
    for (int i = 0; i < topk.size(); i++)
    {
        float x1 = (box_offsets[i].first + topk[i].first) - box_scales[i].first * 0.5 > feat_w ? feat_w : (box_offsets[i].first + topk[i].first) - box_scales[i].first * 0.5;
        float x3 = (box_offsets[i].first + topk[i].first) + box_scales[i].first * 0.5 > feat_w ? feat_w : (box_offsets[i].first + topk[i].first) + box_scales[i].first * 0.5;
        float x2 = (box_offsets[i].second + topk[i].second) - box_scales[i].second * 0.5 > feat_w ? feat_w : (box_offsets[i].second + topk[i].second) - box_scales[i].second * 0.5;
        float x4 = (box_offsets[i].second + topk[i].second) + box_scales[i].second * 0.5 > feat_w ? feat_w : (box_offsets[i].second + topk[i].second) + box_scales[i].second * 0.5;
        x1 = x1 < 0 ? 0 : x1;
        x2 = x2 < 0 ? 0 : x2;
        x3 = x3 < 0 ? 0 : x3;
        x4 = x4 < 0 ? 0 : x4;
        box_coords.push_back(std::make_tuple(x1, x2, x3, x4));
    }

    //std::vector<std::tuple<float, float, float, float>> box_coords_norm;
    //ncnn::Mat box_coords_norm_ = ncnn::Mat(4, 6, 1);
    float* box_coords_norm_data = (float*)box_coords_norm_.data;
    for (int i = 0; i < box_coords.size(); i++)
    {
        float x1 = std::min(std::max(std::get<0>(box_coords[i]) * multipose_scale[0], 0.f), 1.f);
        float x2 = std::min(std::max(std::get<1>(box_coords[i]) * multipose_scale[1], 0.f), 1.f);
        float x3 = std::min(std::max(std::get<2>(box_coords[i]) * multipose_scale[2], 0.f), 1.f);
        float x4 = std::min(std::max(std::get<3>(box_coords[i]) * multipose_scale[3], 0.f), 1.f);
        //box_coords_norm.push_back(std::make_tuple(x1, x2, x3, x4));
        box_coords_norm_data[i * 4] = x1;
        box_coords_norm_data[i * 4 + 1] = x2;
        box_coords_norm_data[i * 4 + 2] = x3;
        box_coords_norm_data[i * 4 + 3] = x4;
    }
}
static void calcTopkCenter(ncnn::Mat& center,ncnn::Mat& center_idx, std::vector<std::pair<int, float>>& center_conf,
        std::vector<std::pair<int, int>>& topk, int feat_w)
{
    float* center_data = (float*)center.data;
    float* center_idx_data = (float*)center_idx.data;
    for (int i = 0; i < center_idx.total(); i++)
        center_idx_data[i] = std::fabs(center_idx_data[i]) < 0.000001 ? 1 : 0;

    //std::vector<std::pair<int, float>> center_conf;
    for (int i = 0; i < center_idx.total(); i++)
    {
        float val = center_data[i] * center_idx_data[i];
        center_conf.push_back(std::pair<int, float>(i, val));
    }

    std::sort(center_conf.begin(), center_conf.end(), [](std::pair<int, float> c1, std::pair<int, float> c2) {return c1.second > c2.second; });
    for (int i = 0; i < 6; i++)
    {
        int x = center_conf[i].first - std::floor(center_conf[i].first / feat_w) * feat_w;
        int y = std::floor(center_conf[i].first / feat_w);
        topk.push_back(std::pair<int, int>(y, x));
    }
}
static void calcCenterLogit(const std::vector<std::tuple<float, float, float, float>>& box_coords,int feat_w,int feat_h,
        const std::vector<std::vector<float>>& dist_x, const std::vector<std::vector<float>>& dist_y, ncnn::Mat& logit)
{

    float* logit_data = (float*)logit.data;
    {
        std::vector<float> box_coord1, box_coord2, box_coord3, box_coord4;
        for (int j = 0; j < box_coords.size(); j++)
        {
            float coord1 = std::get<0>(box_coords[j]);
            float coord2 = std::get<1>(box_coords[j]);
            float coord3 = std::get<2>(box_coords[j]);
            float coord4 = std::get<3>(box_coords[j]);
            box_coord1.push_back(coord1);
            box_coord2.push_back(coord2);
            box_coord3.push_back(coord3);
            box_coord4.push_back(coord4);
        }


        ncnn::Mat d0 = ncnn::Mat(max_persons, feat_w, feat_h);
        ncnn::Mat d2 = ncnn::Mat(max_persons, feat_w, feat_h);
        float* d0_data = (float*)d0.data;
        float* d2_data = (float*)d2.data;
        ncnn::Mat d1 = ncnn::Mat(max_persons, feat_w, feat_h);
        ncnn::Mat d3 = ncnn::Mat(max_persons, feat_w, feat_h);
        float* d1_data = (float*)d1.data;
        float* d3_data = (float*)d3.data;
        for (int i = 0; i < feat_h; i++)
        {
            for (int j = 0; j < feat_w; j++)
            {
                for (int n = 0; n < max_persons; n++)
                {
                    float val = dist_y[i][j];
                    if (val < box_coord1[n])
                    {
                        d0_data[i * feat_w * max_persons + j * max_persons + n] = 0;
                    }
                    else
                        d0_data[i * feat_w * max_persons + j * max_persons + n] = 1;
                    if (val < box_coord3[n])
                    {
                        d2_data[i * feat_w * max_persons + j * max_persons + n] = 1;
                    }
                    else
                        d2_data[i * feat_w * max_persons + j * max_persons + n] = 0;
                    if (d2_data[i * feat_w * max_persons + j * max_persons + n] > 0 && d0_data[i * feat_w * max_persons + j * max_persons + n] > 0)
                        d0_data[i * feat_w * max_persons + j * max_persons + n] = d2_data[i * feat_w * max_persons + j * max_persons + n] == d0_data[i * feat_w * max_persons + j * max_persons + n] ? 1 : 0;
                    else
                        d0_data[i * feat_w * max_persons + j * max_persons + n] = 0;
                    float val1 = dist_x[i][j];
                    if (val1 < box_coord2[n])
                    {
                        d1_data[i * feat_w * max_persons + j * max_persons + n] = 0;
                    }
                    else
                        d1_data[i * feat_w * max_persons + j * max_persons + n] = 1;
                    if (val1 < box_coord4[n])
                    {
                        d3_data[i * feat_w * max_persons + j * max_persons + n] = 1;
                    }
                    else
                        d3_data[i * feat_w * max_persons + j * max_persons + n] = 0;
                    if (d3_data[i * feat_w * max_persons + j * max_persons + n] > 0 && d1_data[i * feat_w * max_persons + j * max_persons + n] > 0)
                        d1_data[i * feat_w * max_persons + j * max_persons + n] = d3_data[i * feat_w * max_persons + j * max_persons + n] == d1_data[i * feat_w * max_persons + j * max_persons + n] ? 1 : 0;
                    else
                        d1_data[i * feat_w * max_persons + j * max_persons + n] = 0;

                    if (d0_data[i * feat_w * max_persons + j * max_persons + n] > 0 && d1_data[i * feat_w * max_persons + j * max_persons + n] > 0)
                        logit_data[i * feat_w * max_persons + j * max_persons + n] = d0_data[i * feat_w * max_persons + j * max_persons + n] == d1_data[i * feat_w * max_persons + j * max_persons + n] ? 1 : 0;
                    else
                        logit_data[i * feat_w * max_persons + j * max_persons + n] = 0;
                }
            }
        }

    }
}

void NanoDet::decodeRegressWithOffset(const ncnn::Mat& kpt_offset, const ncnn::Mat& logit_transpose, ncnn::Mat& scores_transpose,int feat_w,int feat_h,
        std::vector<float>& kpt_regress_scores, std::vector<std::vector<std::tuple<float, float, float>>>& kpt_offsets_all, ncnn::Mat& keypoints)
{
    ncnn::Mat logit_scores_matmul;
    ncnn::Option opt;
    logit_scores_matmul.create(feat_w, feat_h, num_joints, max_persons, scores_transpose.elemsize, opt.blob_allocator);

    for (int c = 0; c < max_persons; c++)
    {
        ncnn::Mat scores_channel = scores_transpose.channel(c);
        ncnn::Mat logit_channel = logit_transpose.channel(c);
        binaryOp_mul(scores_channel, logit_channel,scores_channel);
    }

    float* scores_transpose_data = (float*)scores_transpose.data;
    float* kpt_offset_data = (float*)kpt_offset.data;

    std::vector<int> heatmap_idx;
    std::vector<int> heatmap_x, heatmap_sub;
    std::vector<std::tuple<int, int, int>> heatmap_gather;
    int index = 0;
    for (int i = 0; i < max_persons; i++)
    {
        for (int j = 0; j < num_joints; j++)
        {
            int max_idx = int(argmax(scores_transpose_data + feat_h * feat_w * index,
                                     scores_transpose_data + (index + 1) * feat_h * feat_w));
            heatmap_idx.push_back(max_idx);
            heatmap_x.push_back(std::floor(max_idx / feat_w));
            heatmap_sub.push_back(max_idx - std::floor(max_idx / feat_w) * feat_w);
            heatmap_gather.push_back(std::make_tuple(std::floor(max_idx / feat_w), max_idx - std::floor(max_idx / feat_w) * feat_w, j));
            index++;
        }
    }

    std::vector<std::pair<float, float>> kpt_offsets;
    for (int i = 0; i < heatmap_gather.size(); i++)
    {
        int idx1 = std::get<0>(heatmap_gather[i]);
        int idx2 = std::get<1>(heatmap_gather[i]);
        int idx3 = std::get<2>(heatmap_gather[i]);
        float x = kpt_offset_data[idx1 * feat_w * num_joints * 2 + idx2 * num_joints * 2 + idx3 * 2];
        float y = kpt_offset_data[idx1 * feat_w * num_joints * 2 + idx2 * num_joints * 2 + idx3 * 2 + 1];
        kpt_offsets.push_back(std::make_pair(x, y));
    }

    ncnn::Mat scores_transpose1;
    transpose(scores_transpose, scores_transpose1, 16);

    ncnn::Mat reduce_max_scores;
    reduction_max(scores_transpose1, reduce_max_scores);
    const float* reduce_max_scores_data = (float*)reduce_max_scores.data;


    for (int i = 0; i < heatmap_gather.size(); i++)
    {
        int idx1 = std::get<0>(heatmap_gather[i]);
        int idx2 = std::get<1>(heatmap_gather[i]);
        int idx3 = std::get<2>(heatmap_gather[i]);
        float score = reduce_max_scores_data[idx1 * feat_w * num_joints + idx2 * num_joints + idx3];
        kpt_regress_scores.push_back(score);
    }


    for (int i = 0; i < max_persons; i++)
    {
        std::vector<std::tuple<float, float, float>> kpt_offset_;
        for (int j = 0; j < num_joints; j++)
        {
            kpt_offsets[i * num_joints + j].first += heatmap_x[i * num_joints + j];
            kpt_offsets[i * num_joints + j].second += heatmap_sub[i * num_joints + j];
            kpt_offsets[i * num_joints + j].first *= multipose_scale[0];
            kpt_offsets[i * num_joints + j].second *= multipose_scale[1];
            kpt_offset_.push_back(std::make_tuple(kpt_offsets[i * num_joints + j].first, kpt_offsets[i * num_joints + j].second, kpt_regress_scores[i * num_joints + j]));//
        }
        kpt_offsets_all.push_back(kpt_offset_);
    }


    float* keypoints_data = (float*)keypoints.data;
    for (int c = 0; c < keypoints.c; c++)
    {
        for (int h = 0; h < keypoints.h; h++)
        {
            for (int w = 0; w < num_joints; w++)
            {
                float x = std::get<0>(kpt_offsets_all[h][w]);
                float y = std::get<1>(kpt_offsets_all[h][w]);
                float s = std::get<2>(kpt_offsets_all[h][w]);
                keypoints_data[c * keypoints.h * keypoints.w + h * keypoints.w + w * 3] = x;
                keypoints_data[c * keypoints.h * keypoints.w + h * keypoints.w + w * 3 + 1] = y;
                keypoints_data[c * keypoints.h * keypoints.w + h * keypoints.w + w * 3 + 2] = s;
            }
        }
    }
}

static void calcTopkScores(const std::vector<std::pair<int, float>>& center_conf,const std::vector<float>& kpt_regress_scores, ncnn::Mat& topk_scores)
{
    ncnn::Mat landmarks_scores = ncnn::Mat(num_joints, max_persons, 1);
    float* landmarks_scores_data = (float*)landmarks_scores.data;
    for (int i = 0; i < max_persons; i++)
    {
        for (int j = 0; j < num_joints; j++)
        {
            float s = kpt_regress_scores[i * num_joints + j] > 0.12700000405311584 ? 1 : 0;
            float s_not = s == 1 ? 0 : 1;
            float a = kpt_regress_scores[i * num_joints + j] * s;
            float b = kpt_regress_scores[i * num_joints + j] * s_not;
            landmarks_scores_data[i * num_joints + j] = a + b;
        }
    }

    std::vector<int> topk_inds;
    ncnn::Mat topk_inds_all = ncnn::Mat(num_joints, max_persons, 1);
    ncnn::Mat topk_inds_mask = ncnn::Mat(num_joints, max_persons, 1);
    float* topk_inds_all_data = (float*)topk_inds_all.data;
    float* topk_inds_mask_data = (float*)topk_inds_mask.data;
    for (int i = 0; i < max_persons; i++)
    {
        int a = center_conf[i].first;
        int b = std::floor(a / 1);
        topk_inds.push_back(a - b);
        int flag = a - b == 0 ? 1 : 0;
        for (int j = 0; j < num_joints; j++)
        {
            topk_inds_all_data[i * num_joints + j] = center_conf[i].second * flag;
            topk_inds_mask_data[i * num_joints + j] = flag;
        }
    }

    ncnn::Mat landmarks_scores_all = ncnn::Mat(1, max_persons, 1);
    float* landmarks_scores_all_data = (float*)landmarks_scores_all.data;
    for (int i = 0; i < max_persons; i++)
    {
        float sum = 0;
        for (int j = 0; j < num_joints; j++)
        {
            landmarks_scores_data[i * num_joints + j] *= topk_inds_all_data[i * num_joints + j];
            sum += landmarks_scores_data[i * num_joints + j];
        }
        landmarks_scores_all_data[i] = sum;
    }

    float* topk_scores_data = (float*)topk_scores.data;
    for (int i = 0; i < max_persons; i++)
    {
        float sum = 0;
        for (int j = 0; j < num_joints; j++)
        {
            float s = kpt_regress_scores[i * num_joints + j] > 0.12700000405311584 ? 1 : 0;
            sum += topk_inds_mask_data[i * num_joints + j] * s;
        }
        sum = sum > 1.f ? sum : 1.f;
        sum = landmarks_scores_all_data[i] / sum;

        float a = topk_inds_mask_data[i * num_joints] * sum;
        float b = (topk_inds_mask_data[i * num_joints] == 1 ? 0 : 1) * sum;
        topk_scores_data[i] = a + b;
    }
}

void NanoDet::postProcess(ncnn::Mat& kpt_regress, ncnn::Mat& center, ncnn::Mat& kpt_heatmap,  ncnn::Mat& kpt_offset,
        ncnn::Mat& center_idx,  ncnn::Mat& box_scale,  ncnn::Mat& box_offset,const std::vector<std::vector<float>>& dist_x,
        const std::vector<std::vector<float>>& dist_y,int feat_w,int feat_h,ncnn::Mat& detect_result)
{
    std::vector<std::pair<int, float>> center_conf;
    std::vector<std::pair<int, int>> topk;
    calcTopkCenter(center, center_idx, center_conf, topk, feat_w);

    ncnn::Mat box_coords_norm = ncnn::Mat(4, 6, 1,sizeof(float));
    std::vector<std::tuple<float, float, float, float>> box_coords;

    decodeTopkBoxCoord(box_scale, box_offset, topk, box_coords_norm, box_coords, feat_w);

    ncnn::Mat scores = ncnn::Mat(num_joints, max_persons, feat_w, feat_h, sizeof(float));
    calcTopkKptScores(kpt_regress, kpt_heatmap, topk, dist_x, dist_y, feat_w, feat_h, scores);

    ncnn::Mat logit = ncnn::Mat(max_persons, 1, feat_w, feat_h);
    calcCenterLogit(box_coords, feat_w, feat_h, dist_x, dist_y,logit);

    ncnn::Mat scores_transpose;
    transpose(scores, scores_transpose, 16);
    ncnn::Mat logit_transpose;
    transpose(logit, logit_transpose, 22);//22,4

    std::vector<float> kpt_regress_scores;
    std::vector<std::vector<std::tuple<float, float, float>>> kpt_offsets_all;
    ncnn::Mat keypoints = ncnn::Mat(num_joints * 3, max_persons, 1);
    decodeRegressWithOffset(kpt_offset, logit_transpose, scores_transpose, feat_w, feat_h, kpt_regress_scores, kpt_offsets_all, keypoints);

    ncnn::Mat topk_scores = ncnn::Mat(1, max_persons, 1);
    calcTopkScores(center_conf, kpt_regress_scores, topk_scores);

    detect_result.create(num_joints*3+4+1, max_persons, 1);
    std::vector<ncnn::Mat> inputs_concat;
    inputs_concat.push_back(keypoints);
    inputs_concat.push_back(box_coords_norm);
    inputs_concat.push_back(topk_scores);
    concat(inputs_concat, 2, detect_result);
}


NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

}


int NanoDet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals,const float* _multipose_scale, bool use_gpu)
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


    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    multipose_scale[0] = _multipose_scale[0];
    multipose_scale[1] = _multipose_scale[1];
    multipose_scale[2] = _multipose_scale[2];
    multipose_scale[3] = _multipose_scale[3];
    scale = _multipose_scale[4];
    
    if(_target_size == 192)
    {
        input_w = 256;
        input_h = 192;
    }
    else
    {
        input_w = 320;
        input_h = 256;
    }
    int feature_size_h = input_h/4;
    int feature_size_w = input_w/4;
    for (int i = 0; i < feature_size_h; i++)
    {
        std::vector<float> x, y;
        for (int j = 0; j < feature_size_w; j++)
        {
            x.push_back(j);
            y.push_back(i);
        }
        dist_y.push_back(y);
        dist_x.push_back(x);
    }
    return 0;
}

int NanoDet::detect(const cv::Mat &rgb, std::vector<Person> &objects)
{
    int w = rgb.cols;
    int h = rgb.rows;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)input_w / w;
        w = input_w;
        h = h * scale;
    }
    else
    {
        scale = (float)input_h / h;
        h = input_h;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows, w, h);
    int wpad = input_w - w;
    int hpad = input_h - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    ncnn::Extractor ex = poseNet.create_extractor();
    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    
    ex.input("input", in_pad);

    ncnn::Mat kpt_regress, center, kpt_heatmap, kpt_offset;
    ncnn::Mat center_idx, box_scale, box_offset;
    ex.extract("regress", kpt_regress);
    ex.extract("offset", kpt_offset);
    ex.extract("heatmap", kpt_heatmap);
    ex.extract("center", center);
    ex.extract("center_idx", center_idx);
    ex.extract("x", box_scale);
    ex.extract("y", box_offset);

    int feat_w = input_w / 4;
    int feat_h = input_h / 4;
    ncnn::Mat detect_result;
    postProcess(kpt_regress, center, kpt_heatmap, kpt_offset, center_idx, box_scale, box_offset, dist_x, dist_y, feat_w, feat_h, detect_result);

    PadInfo pad_info;
    pad_info.target_w = input_w;
    pad_info.target_h = input_h;
    pad_info.wpad = wpad;
    pad_info.hpad = hpad;
    pad_info.scale = scale;
    decodeKeypoints(detect_result, objects, pad_info, 0.3);

    return 0;
}

int NanoDet::draw(cv::Mat& rgb,std::vector<Person> &objects)
{
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

    for (int i = 0; i < objects.size(); i++)
    {
        Person person = objects[i];
        for (int j = 0; j < 17; j++)
        {
            if (person.points[j].prob > 0.2)
                cv::circle(rgb, cv::Point2f(person.points[j].x, person.points[j].y), 3, cv::Scalar(100, 255, 150), -1);

        }
        for (int j = 0; j < 18; j++)
        {
            if (person.points[skele_index[j][0]].prob > 0.2 && person.points[skele_index[j][1]].prob > 0.2)
                cv::line(rgb, cv::Point(person.points[skele_index[j][0]].x, person.points[skele_index[j][0]].y),
                         cv::Point(person.points[skele_index[j][1]].x, person.points[skele_index[j][1]].y),
                         cv::Scalar(color_index[j][0], color_index[j][1], color_index[j][2]), 2);

        }

        cv::rectangle(rgb, cv::Rect(person.x,person.y,person.width,person.height), cv::Scalar(0, 255, 255), 2, 8, 0);

    }

    return 0;
}
