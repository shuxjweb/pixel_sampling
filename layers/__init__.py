# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from .triplet_loss import TripletLoss, TripletLossPart, CrossEntropyLabelSmooth, CrossEntropyLabelSmoothMask, MaskMseLoss, CrossEntropy
from .triplet_loss import CrossEntropyLabelMask, TripletLossMask, FeatLoss
from .cluster_loss import ClusterLoss
from .center_loss import CenterLoss
import math
import torch.nn.functional as F


def make_loss_with_center(cfg, num_classes, feat_dim=2048):    # modified by gu
    triplet = TripletLoss(cfg.margin)  # triplet loss
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target):      # [64, 751],  [64, 512], [64,]
        loss_x = xent(score, target)         # 6.618
        loss_t = triplet(feat, target)[0]    # 3.2403
        loss_c = center_criterion(feat, target)         # 2625.2329
        loss = loss_x + loss_t + cfg.center_loss_weight * loss_c         # 11.1710
        return loss

    return loss_func, center_criterion


def make_loss_with_triplet_entropy(cfg, num_classes):    # modified by gu
    triplet = TripletLoss(cfg.margin)  # triplet loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target):      # [64, 751],  [64, 512], [64,]
        loss_x = xent(score, target)         # 6.618
        loss_t = triplet(feat, target)[0]    # 3.2403
        loss = loss_x + loss_t         # 11.1710
        return loss

    return loss_func



def make_loss_with_pcb(cfg, num_classes):       # modified by gu
    triplet = TripletLoss(cfg.margin)           # triplet loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(scores, feats, targets):      # [64, 751],  [64, 512], [64,]
        loss_x = [xent(scores[:, ii], targets) for ii in range(scores.shape[1])]
        loss_x = sum(loss_x) / len(loss_x)      # 6.618
        loss_t = triplet(feats, targets)[0]     # 5.8445
        loss = loss_x + loss_t                  # 15.7047
        return loss

    return loss_func



def make_loss_with_mgn(cfg, num_classes):    # modified by gu
    triplet = TripletLoss(cfg.margin)  # triplet loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(feats, scores, targets):      # [64, 751],  [64, 512], [64,]
        loss_x = [xent(scores, targets) for scores in scores]
        loss_x = sum(loss_x) / len(loss_x)      # 6.618
        loss_t = [triplet(feat, targets)[0] for feat in feats]
        loss_t = sum(loss_t) / len(loss_t)   # 3.0438
        loss = loss_x + loss_t        # 15.7047
        return loss

    return loss_func



def make_loss_ce_triplet(cfg):    # modified by gu
    triplet = TripletLoss(cfg.margin)  # triplet loss
    xent = CrossEntropy()

    def loss_func(score, feat, target):      # [64, 751],  [64, 512], [64,]
        loss_x = xent(score, target)         # 6.618
        loss_t = triplet(feat, target)[0]  # 3.2403
        loss = loss_x + loss_t  # 11.1710
        return loss

    return loss_func


def make_loss_with_triplet_entropy_mse(cfg, num_classes):    # modified by gu
    triplet = TripletLoss(cfg.margin)  # triplet loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    ft_loss = FeatLoss()

    def loss_func(score, feat, target):      # [64, 751],  [64, 512], [64,]
        B = int(score.shape[0 ] / 2)
        loss_x = xent(score, target)         # 6.618
        loss_t = triplet(feat, target)[0]    # 3.2403
        loss_f = ft_loss(feat[0: B], feat[B:])      # 0.99
        loss = loss_x + loss_t + loss_f        # 11.1710
        return loss

    return loss_func


def make_loss_with_triplet_entropy_hrnet(cfg, num_classes):    # modified by gu
    triplet = TripletLoss(cfg.margin)  # triplet loss
    xent = CrossEntropyLabelMask(num_classes=num_classes)

    def loss_func(score, feat, target, mask_node):      # [64, 14, 751],  [64, 512], [64,], [64, 14, 1]
        N = score.shape[1]
        loss = 0.0
        for ii in range(N):
            loss_x = xent(score[:, ii], target, mask_node[:, ii])         # 5.0110
            loss += loss_x                  # 5.5961

        loss_t = triplet(feat, target)[0]  # 0.5851
        loss = loss + N * 10.0 * loss_t
        return loss

    return loss_func




def make_loss_with_entropy(num_classes):    # modified by gu
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, target):               # [64, 751],  [64,]
        loss = xent(score, target)              # 6.618
        return loss

    return loss_func





def make_loss_with_center_pcb_gcn(cfg, num_classes, feat_dim=2048):    # modified by gu
    triplet = TripletLoss(cfg.margin)  # triplet loss
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target):      # [64, 751],  [64, 12288], [64,], [64, 14]
        loss_x = torch.stack([xent(score[:, ii], target) for ii in range(score.shape[1])]).sum()
        loss_t = triplet(feat, target)[0]    # 3.2403
        loss = loss_x + loss_t

        # loss_c = center_criterion(feat, target)         # 2625.2329
        # loss = loss_x + loss_t + cfg.center_loss_weight * loss_c         # 39.7 + 0.3940 + 6.1263
        return loss

    return loss_func, center_criterion


def make_loss_with_center_pcb(cfg, num_classes, feat_dim=2048):    # modified by gu
    triplet = TripletLoss(cfg.margin)  # triplet loss
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    xent = CrossEntropyLabelMask(num_classes=num_classes)

    def loss_func(score, feat, target, mask_node=None):      # [64, 751],  [64, 12288], [64,], [64, 14]
        loss_x = torch.stack([xent(score[:, ii], target, mask_node[:, ii:ii+1]) for ii in range(score.shape[1])]).sum()
        loss_t = triplet(feat, target)[0]    # 3.2403
        loss = loss_x + loss_t

        # loss_c = center_criterion(feat, target)         # 2625.2329
        # loss = loss_x + loss_t + cfg.center_loss_weight * loss_c         # 39.7 + 0.3940 + 6.1263
        return loss

    return loss_func, center_criterion



def make_loss_triplet(cfg, num_classes):    # modified by gu
    triplet = TripletLoss(cfg.margin)  # triplet loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target):       # [64, 3, 751], [64, 6144], [64, 3]
        # loss_x = xent(score, target)
        loss_x = torch.stack([xent(score[:, ii], target) for ii in range(score.shape[1])]).sum()
        loss_t = triplet(feat, target)[0]    # 3.2403
        loss = loss_x + loss_t        # 11.1710
        return loss

    return loss_func



def make_loss_with_triplet(cfg, num_classes):    # modified by gu
    triplet = TripletLoss(cfg.margin)  # triplet loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target):       # [64, 3, 751], [64, 6144], [64, 3]
        loss_x = xent(score, target)
        loss_t = triplet(feat, target)[0]    # 3.2403
        loss = loss_x + loss_t        # 11.1710
        return loss

    return loss_func



def make_loss_triplet(cfg):    # modified by gu
    triplet = TripletLoss(cfg.margin)  # triplet loss

    def loss_func(feat, target):       # [64, 3, 751], [64, 6144], [64, 3]
        loss_t = triplet(feat, target)[0]    # 3.2403
        return loss_t

    return loss_func


def make_loss_cross():    # modified by gu
    # xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    xent = CrossEntropy()

    def loss_func(score, target):       # [64, 751], [64,]
        loss = xent(score, target)
        return loss

    return loss_func






