import torch

from ..bbox import PseudoSampler, assign_and_sample, bbox2delta, build_assigner
from ..utils import multi_apply
from .anchor_target import unmap, anchor_inside_flags
import numpy as np


def anchor_target_with_mask(anchor_list,
                            valid_flag_list,
                            gt_bboxes_list,
                            gt_masks_list,
                            img_metas,
                            target_means,
                            target_stds,
                            cfg,
                            gt_bboxes_ignore_list=None,
                            gt_labels_list=None,
                            label_channels=1,
                            sampling=True,
                            unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        gt_masks_list (list[Tensor]): Ground truth masks of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, masks_targets_list, masks_inds_list,
     pos_inds_list, neg_inds_list) = multi_apply(
        anchor_target_with_mask_single,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        gt_bboxes_ignore_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        target_means=target_means,
        target_stds=target_stds,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            masks_targets_list, masks_inds_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def anchor_target_with_mask_single(flat_anchors,
                                   valid_flags,
                                   gt_bboxes,
                                   gt_bboxes_ignore,
                                   gt_labels,
                                   gt_masks,
                                   img_meta,
                                   target_means,
                                   target_stds,
                                   cfg,
                                   label_channels=1,
                                   sampling=True,
                                   unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None,) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    mask_inds = torch.arange(bbox_weights.size(0))[bbox_weights[:, 0] > 0]

    # here we need move gt_masks to gpu first, because following DefaultFormatBundle,
    # gt_masks from data_loader are on cpu
    gt_masks = torch.from_numpy(gt_masks).float().to(
        bbox_targets.device)

    # if there are too many pos_inds, we need to sample them, for restricted gpu memory occupation
    mask_targets, mask_inds = mask_sampler(sampling_result.pos_assigned_gt_inds,
                                           mask_inds, gt_masks, cfg.masks_to_train)

    # Be careful!!!
    # If 'unmap_outputs' is True, then pos_inds and neg_inds are not the corresponding indices in
    # labels, label_weights, bbox_targets..., but in inside parts
    return (labels, label_weights, bbox_targets, bbox_weights, mask_targets, mask_inds,
            pos_inds, neg_inds)


def mask_sampler(assigned_gt_inds, inds, masks, keep_num=100, ignore_area=100):
    """
    Here, we just random sample mask indexes. Maybe more complicated methods will contribute to better results,
    such as sampling according to mask size, sampling according to category,
    sampling according to matched instance...
    """
    assert inds.size(0) == assigned_gt_inds.size(0)
    size = torch.sum(masks, dim=(1, 2))
    assigned_size = size[assigned_gt_inds]
    keep_inds = assigned_size > ignore_area
    assigned_gt_inds = assigned_gt_inds[keep_inds]
    inds = inds[keep_inds]

    keep = torch.randperm(inds.size(0))[:keep_num]
    sampled_assigned_gt_inds = assigned_gt_inds[keep]

    return masks[sampled_assigned_gt_inds, :, :], inds[keep]
