from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target_with_mask, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms, multiclass_nms_with_extra)
from ..builder import build_loss
from ..registry import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module
class RdsnetHead(AnchorHead):
    """RDSNet head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 rep_channels=32,
                 feat_channels=256,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        self.rep_channels = rep_channels
        super(RdsnetHead, self).__init__(num_classes, in_channels, feat_channels,
                                         anchor_scales, anchor_ratios, anchor_strides,anchor_base_sizes,
                                         target_means, target_stds,
                                         loss_cls, loss_bbox)

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.conv_rep = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.rep_channels * 2, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_rep, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        obj_rep = self.conv_rep(x)
        return cls_score, bbox_pred, obj_rep

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             obj_reps,
             gt_bboxes,
             gt_masks,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target_with_mask(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_masks,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None, None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets, mask_inds, num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        # return pos entries for mask generator
        bs = cls_scores[0].size(0)
        cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(bs, -1, self.cls_out_channels)
                      for cls_score in cls_scores]
        bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(bs, -1, 4) for bbox_pred in bbox_preds]
        obj_reps = [obj_rep.permute(0, 2, 3, 1).reshape(bs, -1, self.rep_channels*2) for obj_rep in obj_reps]
        cls_scores = torch.cat(cls_scores, 1)
        bbox_preds = torch.cat(bbox_preds, 1)
        obj_reps = torch.cat(obj_reps, 1)
        bbox_targets = torch.cat(bbox_targets_list, 1)
        pos_cls_scores = []
        pos_bbox_preds = []
        pos_obj_reps = []
        pos_bbox_gts = []
        pos_mask_gts = []
        for cls_score, bbox_pred, obj_rep, pos_ind, anchor, bbox_target, mask_target, img_meta in zip(
                cls_scores, bbox_preds, obj_reps, mask_inds, anchor_list, bbox_targets, mask_targets, img_metas):
            pos_cls_scores.append(cls_score[pos_ind])
            pos_bbox_preds.append(delta2bbox(anchor[pos_ind], bbox_pred[pos_ind],
                                             means=self.target_means, stds=self.target_stds,
                                             max_shape=img_meta['img_shape']))
            pos_obj_reps.append(obj_rep[pos_ind])
            pos_bbox_gts.append(delta2bbox(anchor[pos_ind], bbox_target[pos_ind],
                                           means=self.target_means, stds=self.target_stds,
                                           max_shape=img_meta['img_shape']))
            pos_mask_gts.append(mask_target)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox),\
               dict(pos_cls=pos_cls_scores, pos_bbox=pos_bbox_preds, pos_obj=pos_obj_reps,
                    gt_bbox=pos_bbox_gts, gt_mask=pos_mask_gts)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, obj_reps, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(obj_reps)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            obj_reps_list = [
                obj_reps[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, obj_reps_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          obj_reps,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        # TODO:
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors) == len(obj_reps)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_reps = []
        for cls_score, bbox_pred, obj_rep, anchors in zip(cls_scores, bbox_preds, obj_reps,
                                                          mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:] == obj_rep.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            obj_rep = obj_rep.permute(1, 2, 0).reshape(-1, self.rep_channels*2)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                obj_rep = obj_rep[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_reps.append(obj_rep)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_reps = torch.cat(mlvl_reps)
        det_bboxes, det_labels, det_reps = multiclass_nms_with_extra(mlvl_bboxes, mlvl_scores, mlvl_reps,
                                                                     cfg.score_thr, cfg.nms,
                                                                     cfg.max_per_img)
        return det_bboxes, det_labels, det_reps
