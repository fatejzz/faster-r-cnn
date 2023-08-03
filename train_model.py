from train import train
from utils.config import  opt
import ipdb
import matplotlib
from tqdm import tqdm
import torch
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
im_scale=im_scale.numpy()
_,_,H,W=im.shape
img_size = (H, W)

faster_rcnn = FasterRCNNVGG16()
features=faster_rcnn.extractor(im)

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
# ##rcnn中的解码过程 模型预测的是偏移和比例 得到预测的box
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
bbox = bboxes[0]
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
n_sample = 128
pos_ratio = 0.25
pos_iou_thresh = 0.5
neg_iou_thresh_hi = 0.5
neg_iou_thresh_lo = 0.0
bbox=at.tonumpy(bbox)
n_bbox, _ = bbox.shape
roi = np.concatenate((roi, bbox), axis=0)
pos_roi_per_image = np.round(n_sample * pos_ratio)
iou = bbox_iou(roi, bbox)
gt_assignment = iou.argmax(axis=1)
max_iou = iou.max(axis=1)
# Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
# The label with value 0 is the background.
gt_roi_label = label[gt_assignment] + 1