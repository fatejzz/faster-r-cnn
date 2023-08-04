from train import train
from utils.config import  opt
import ipdb
import matplotlib
from tqdm import tqdm
import torch
import time
from sys import path
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
opt.voc_data_dir = "/home/hkyunqi/jzz/faster-rcnn/VOCdevkit/VOC2007"
opt.num_workers=1
dataset = Dataset(opt)
dataloader = data_.DataLoader(dataset, \
                              batch_size=1, \
                              shuffle=False, \
                              # pin_memory=True,
                              num_workers=opt.num_workers)
for i,data in enumerate(dataloader):
    if i==0:
        data
        break

##train
im,bboxes,labels,im_scale=data
im=im.cuda()
bboxes=bboxes.cuda()
labels=labels.cuda()
im_scale=im_scale.numpy()
_,_,H,W=im.shape
img_size = (H, W)

faster_rcnn = FasterRCNNVGG16()
faster_rcnn.cuda()
features=faster_rcnn.extractor(im)

# start=time.time()
# faster_rcnn.predict(im.cuda(),visualize=True)
# torch.cuda.synchronize()
# end=time.time()
# print(end-start)
'''rpn'''
rpn_locs, rpn_scores, rois, roi_indices, anchor =faster_rcnn.rpn(features, img_size, im_scale)
# from model.region_proposal_network import RegionProposalNetwork
# anchor_scales=[8,16,32]
# ratios=[0.5,1,2]
# rpn = RegionProposalNetwork(
#     512, 512,
#     ratios=[0.5,1,2],
#     anchor_scales=[8,16,32],
#     feat_stride=16,
# )
# '''rpn_init'''
# ######generate_anchor_base
# from model.utils.bbox_tools import generate_anchor_base
# ##生成矩形的边长为(base_size*anchor_scales) 特征图左上角中anchor_box(lt,rb)坐标
# anchor_base = generate_anchor_base(
#             anchor_scales=anchor_scales, ratios=ratios)
# feat_stride=16
# from model.utils.creator_tool import ProposalCreator
# proposal_layer = ProposalCreator(faster_rcnn)
# n_anchor=9
# in_channels=512
# mid_channels=512
# conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
# score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
# loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
# '''rpn_forward'''
# x=feature
# n, _, hh, ww = x.shape
# from model.region_proposal_network import _enumerate_shifted_anchor
# height=hh;width=ww;
# anchor = _enumerate_shifted_anchor(
#     np.array(anchor_base),
#     feat_stride, hh, ww)
# n_anchor = anchor.shape[0] // (hh * ww)
# h = F.relu(conv1(x))
# rpn_locs = loc(h)
# rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
# rpn_scores = score(h)
# rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
# rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
# rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
# rpn_fg_scores = rpn_fg_scores.view(n, -1)
# rpn_scores = rpn_scores.view(n, -1, 2)
# rois = list()
# roi_indices = list()
# for i in range(n):
#     roi = proposal_layer(
#         rpn_locs[i].cpu().data.numpy(),
#         rpn_fg_scores[i].cpu().data.numpy(),
#         anchor, img_size,
#         scale=im_scale)
#     batch_index = i * np.ones((len(roi),), dtype=np.int32)
#     rois.append(roi)
#     roi_indices.append(batch_index)
#
# rois = np.concatenate(rois, axis=0)
# roi_indices = np.concatenate(roi_indices, axis=0)
# ###ProposalCreator
# from model.utils.creator_tool import  *
# from model.utils.bbox_tools import *
# nms_thresh = 0.7
# n_train_pre_nms = 12000
# n_train_post_nms = 2000
# n_test_pre_nms = 6000
# n_test_post_nms = 300
# min_size = 16
# loc=rpn_locs[i].cpu().data.numpy()
# scale=im_scale
# ##rcnn中的解码过程 模型预测的是偏移和比例 得到proposal
# roi = loc2bbox(anchor, loc)
# roi[:, slice(0, 4, 2)] = np.clip(
#     roi[:, slice(0, 4, 2)], 0, img_size[0])
# roi[:, slice(1, 4, 2)] = np.clip(
#     roi[:, slice(1, 4, 2)], 0, img_size[1])
# min_size = min_size * scale
# hs = roi[:, 2] - roi[:, 0]
# ws = roi[:, 3] - roi[:, 1]
# keep = np.where((hs >= min_size.numpy()) & (ws >= min_size.numpy()))[0]
# roi = roi[keep, :]
# score = rpn_fg_scores[0].cpu().data.numpy()
# score=score[keep]
# order = score.ravel().argsort()[::-1]
# if n_train_pre_nms  > 0:
#     order = order[:n_train_pre_nms ]
# roi = roi[order, :]
# score = score[order]
# keep = nms(
#     torch.from_numpy(roi).cuda(),
#     torch.from_numpy(score).cuda(),
#     0.7)
# if n_train_post_nms>0:
#     keep=keep[:n_train_post_nms]
# roi[keep.cpu().numpy()]

'''forward'''
bbox=at.tonumpy(bboxes)[0]
label = labels[0]
rpn_score = rpn_scores[0]
rpn_loc = rpn_locs[0]
roi = rois
loc_normalize_mean = faster_rcnn.loc_normalize_mean
loc_normalize_std = faster_rcnn.loc_normalize_std
from model.utils.creator_tool import *
proposal_target_creator = ProposalTargetCreator()
sample_roi, gt_roi_loc, gt_roi_label = proposal_target_creator(
roi,at.tonumpy(bbox),at.tonumpy(label),
loc_normalize_mean,
loc_normalize_std)

'''ProposalTargetCreator'''
# n_sample = 128
# pos_ratio = 0.25
# pos_iou_thresh = 0.5
# neg_iou_thresh_hi = 0.5
# neg_iou_thresh_lo = 0.0
# bbox=at.tonumpy(bbox)
# n_bbox, _ = bbox.shape
# roi = np.concatenate((roi, bbox), axis=0)
# pos_roi_per_image = np.round(n_sample * pos_ratio) ##每幅图片中的正样本数
# iou = bbox_iou(roi, bbox) ##涉及numpy broadcast机制 [num_detection,gt_box]之间的iou
# gt_assignment = iou.argmax(axis=1) ##经iou计算属于哪个gt_box
# max_iou = iou.max(axis=1)
# # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
# # The label with value 0 is the background.
# gt_roi_label = label[gt_assignment] + 1
#
# # Select foreground RoIs as those with >= pos_iou_thresh IoU. 选取正样本
# pos_index = np.where(max_iou >= pos_iou_thresh)[0]
# pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
# if pos_index.size > 0:
#     pos_index = np.random.choice(
#         pos_index, size=pos_roi_per_this_image, replace=False)
#
# # Select background RoIs as those within
# # [neg_iou_thresh_lo, neg_iou_thresh_hi).
# neg_index = np.where((max_iou < neg_iou_thresh_hi) &
#                      (max_iou >= neg_iou_thresh_lo))[0]
# neg_roi_per_this_image = n_sample - pos_roi_per_this_image
# neg_roi_per_this_image = int(min(neg_roi_per_this_image,
#                                  neg_index.size))
# if neg_index.size > 0:
#     neg_index = np.random.choice(
#         neg_index, size=neg_roi_per_this_image, replace=False)
#
# # The indices that we're selecting (both positive and negative).
# keep_index = np.append(pos_index, neg_index)
# gt_roi_label = gt_roi_label[keep_index]
# gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0 真实标签
# sample_roi = roi[keep_index]
#
# # Compute offsets and scales to match sampled RoIs to the GTs.
# gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]]) ##反向解码生成对应anchor的缩放和平移
# gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
#                ) / np.array(loc_normalize_std, np.float32))

'''forward'''
sample_roi_index = torch.zeros(len(sample_roi)) ##训练时已经提取了正负样本
roi_cls_loc, roi_score = faster_rcnn.head(
    features.cuda(),
    sample_roi,  ##特征尺度在原始图像上
    sample_roi_index)
'''ROIhead'''
# from model.faster_rcnn_vgg16 import *
# roi_size=7
# feat_stride=16
# spatial_scale=(1. / feat_stride)
# extractor, classifier = decom_vgg16()
# roi = RoIPool( (roi_size, roi_size),spatial_scale)
# cls_loc = nn.Linear(4096,20 * 4)
# score = nn.Linear(4096, 20)
#
# x=features
# rois=sample_roi
# roi_indices=sample_roi_index
# roi_indices = at.totensor(roi_indices).float()
# rois = at.totensor(rois).float()
#
# indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
# xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
# indices_and_rois = xy_indices_and_rois.contiguous()
#
# pool = roi(x.cuda(), indices_and_rois) ##roi_pool
# pool = pool.view(pool.size(0), -1)
# fc7 = classifier(pool.cpu())
# roi_cls_locs = cls_loc(fc7)
# roi_scores = score(fc7)

'''forward'''
anchor_target_creator = AnchorTargetCreator()
gt_rpn_loc, gt_rpn_label = anchor_target_creator(
    at.tonumpy(bbox),
    anchor,
    img_size)
# '''anchor_target_creator'''
# from model.utils.creator_tool import _get_inside_index,_unmap
# n_sample = 256
# pos_iou_thresh = 0.7
# neg_iou_thresh = 0.3
# pos_ratio = 0.5
#
# img_H, img_W = img_size
#
# n_anchor = len(anchor)
# inside_index = _get_inside_index(anchor, img_H, img_W) ##锚框完全在原图中的
# anchor = anchor[inside_index]
#
#
# def _create_label(inside_index, anchor, bbox):
#     # label: 1 is positive, 0 is negative, -1 is dont care
#     label = np.empty((len(inside_index),), dtype=np.int32)
#     label.fill(-1)
#
#     argmax_ious, max_ious, gt_argmax_ious = \
#         _calc_ious(anchor, bbox, inside_index)
#
#     # assign negative labels first so that positive labels can clobber them 正负样本打标签 +1 正样本 0 负样本 -1 忽略
#     label[max_ious < neg_iou_thresh] = 0
#
#     # positive label: for each gt, anchor with highest iou
#     label[gt_argmax_ious] = 1
#
#     # positive label: above threshold IOU
#     label[max_ious >= pos_iou_thresh] = 1
#
#     # subsample positive labels if we have too many
#     n_pos = int(pos_ratio * n_sample)
#     pos_index = np.where(label == 1)[0]
#     if len(pos_index) > n_pos:
#         disable_index = np.random.choice(
#             pos_index, size=(len(pos_index) - n_pos), replace=False)
#         label[disable_index] = -1
#
#     # subsample negative labels if we have too many
#     n_neg = n_sample - np.sum(label == 1)
#     neg_index = np.where(label == 0)[0]
#     if len(neg_index) > n_neg:
#         disable_index = np.random.choice(
#             neg_index, size=(len(neg_index) - n_neg), replace=False)
#         label[disable_index] = -1
#
#     return argmax_ious, label
#
#
# def _calc_ious(anchor, bbox, inside_index):
#     # ious between the anchors and the gt boxes
#     ious = bbox_iou(anchor, bbox)
#     argmax_ious = ious.argmax(axis=1)
#     max_ious = ious[np.arange(len(inside_index)), argmax_ious]
#     gt_argmax_ious = ious.argmax(axis=0)
#     gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
#     gt_argmax_ious = np.where(ious == gt_max_ious)[0]
#
#     return argmax_ious, max_ious, gt_argmax_ious
#
# argmax_ious, label = _create_label(
#     inside_index, anchor, bbox)
#
# # compute bounding box regression targets
# loc = bbox2loc(anchor, bbox[argmax_ious])
#
# # map up to original set of anchors 将正负样本映射回到原始anchor中
# label = _unmap(label, n_anchor, inside_index, fill=-1)
# loc = _unmap(loc, n_anchor, inside_index, fill=0)

'''forward '''
gt_rpn_label = at.totensor(gt_rpn_label).long()
gt_rpn_loc = at.totensor(gt_rpn_loc)
rpn_sigma = opt.rpn_sigma
from trainer import  _fast_rcnn_loc_loss
rpn_loc_loss = _fast_rcnn_loc_loss( ##rpn定位损失
    rpn_loc,
    gt_rpn_loc,
    gt_rpn_label.data,
    rpn_sigma)

'''RPN LOSS LOC'''
# def _smooth_l1_loss(x, t, in_weight, sigma):
#     sigma2 = sigma ** 2
#     diff = in_weight * (x - t)
#     abs_diff = diff.abs()
#     flag = (abs_diff.data < (1. / sigma2)).float()
#     y = (flag * (sigma2 / 2.) * (diff ** 2) +
#          (1 - flag) * (abs_diff - 0.5 / sigma2))
#     return y.sum()
#
# def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
#     in_weight = torch.zeros(gt_loc.shape).cuda()
#     # Localization loss is calculated only for positive rois.
#     # NOTE:  unlike origin implementation,
#     # we don't need inside_weight and outside_weight, they can calculate by gt_label
#     in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
#     loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
#     # Normalize by total number of negtive and positive rois.
#     loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
#     return loc_loss

'''forward'''
rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1) ##rpn分类损失
_gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
_rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]

'''ROI head loss'''
n_sample = roi_cls_loc.shape[0]
roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                      at.totensor(gt_roi_label).long()]
gt_roi_label = at.totensor(gt_roi_label).long()
gt_roi_loc = at.totensor(gt_roi_loc)

roi_loc_loss = _fast_rcnn_loc_loss(
    roi_loc.contiguous(),
    gt_roi_loc,
    gt_roi_label.data,
    opt.roi_sigma)

roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
losses = losses + [sum(losses)]

