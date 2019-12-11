import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, xavier_init, normal_init

from mmdet.core import auto_fp16, force_fp32, sanitize_coordinates
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class RdsMaskHead(nn.Module):
    """Multi-level fused pixel head.

    level_3 -> 3x3 conv -> 2* -> 3x3 conv -> 2* -> 3x3 conv -> 2* -
                                                                   |
    level_2 -> 3x3 conv -> 2* -> 3x3 conv -> 2* ------------------ +
                                                                   |
    level_1 -> 3x3 conv -> 2* ------------------------------------ +
                                                                   |
    level_0 -> 3x3 conv -------------------------------------------+ -> 1x1 conv
    """  # noqa: W605

    def __init__(self,
                 num_ins,
                 end_level=-1,
                 num_convs=0,
                 in_channels=256,
                 conv_out_channels=256,
                 rep_channels=32,
                 loss_weight=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 crop_cfg=None):
        super(RdsMaskHead, self).__init__()
        self.num_ins = num_ins
        self.end_level = end_level

        if end_level == -1:
            self.end_level = self.num_ins - 1
        else:
            self.end_level = end_level
            assert end_level < self.num_ins

        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.rep_channels = rep_channels
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.crop_cfg = crop_cfg
        self.fp16_enabled = False

        self.lateral_convs = nn.ModuleList()
        # for level i==0: feat ==> ConvModule
        lateral_conv = nn.ModuleList()
        lateral_conv.append(ConvModule(
            self.in_channels,
            self.conv_out_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=False
        ))
        self.lateral_convs.append(lateral_conv)
        # for level i==1: feat ==> ConvModule ==> UpSampling(*2)
        # for level i==2: feat ==> ConvModule ==> UpSampling(*2) ==> ConvModule ==> UpSampling(*2)
        # for level i==3: ...
        for i in range(1, self.end_level + 1):
            lateral_convs = nn.ModuleList()
            in_channels = self.in_channels
            for j in range(i):
                lateral_convs.append(ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False
                ))
                lateral_convs.append(nn.UpsamplingBilinear2d(scale_factor=2))
                in_channels = self.conv_out_channels
            self.lateral_convs.append(lateral_convs)

        self.final_convs = nn.ModuleList()
        for i in range(self.num_convs):
            final_conv = nn.ModuleList()
            final_conv.append(ConvModule(
                conv_out_channels,
                conv_out_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=False))
            final_conv.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.final_convs.append(final_conv)
        self.conv_reps = nn.Conv2d(conv_out_channels, self.rep_channels, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.conv_reps, std=0.01)

    @auto_fp16()
    def forward(self, feats, obj_reps, img_metas):
        x = feats[0]
        for m in self.lateral_convs[0]:
            x = m(x)

        for lateral_conv, feat in zip(self.lateral_convs[1:], feats[1:]):
            for m in lateral_conv:
                feat = m(feat)
            x += feat

        for i in range(self.num_convs):
            for m in self.final_convs[i]:
                x = m(x)

        pixel_reps = self.conv_reps(x)

        pred_masks_list = []
        for i, (pixel_rep, obj_rep, img_meta) in enumerate(zip(pixel_reps, obj_reps, img_metas)):
            if obj_rep.size(0) == 0:
                h, w = pixel_rep.size()[:2]
                pred_masks_list.append(pixel_rep.new_zeros(0, h, w, 2))
                continue
            pixel_rep = pixel_rep.permute(1, 2, 0)
            obj_rep = obj_rep.view(-1, self.rep_channels)
            # [mask_h, mask_w, rep_channels] * [rep_channels, N*2].T ==> [mask_h, mask_w, N*2]
            pred_masks = pixel_rep @ obj_rep.t()
            # [mask_h, mask_w, N*2] ==> [N*2, mask_h, mask_w]
            pred_masks = pred_masks.permute(2, 0, 1)
            assert pred_masks.size(1) / img_meta['stack_shape'][0] ==\
                   pred_masks.size(2) / img_meta['stack_shape'][1]
            scale_factor = img_meta['stack_shape'][0] // pred_masks.size(1)
            # remove padding which added during stacking
            dh, dw = img_meta['pad_shape'][0]//scale_factor, img_meta['pad_shape'][1]//scale_factor
            pred_masks = pred_masks[:, :dh, :dw]
            # [N*2, mask_h, mask_w] ==> [N, mask_h, mask_w, 2]
            pred_masks = pred_masks.view(-1, 2, pred_masks.size(1), pred_masks.size(2)).permute(0, 2, 3, 1)
            pred_masks_list.append(pred_masks)
        return pred_masks_list

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, pred_masks, bboxes, gt_masks, img_metas, cfg):
        losses = torch.zeros(1, device=pred_masks[0].device)
        total_pos_num = 0
        final_masks = []
        final_boxes = []
        for pred_mask, bbox, gt_mask, img_meta in zip(pred_masks, bboxes, gt_masks, img_metas):
            # pred_mask with size: [N, mask_h, mask_w, 2]; gt_mask with size: [N, h, w]
            assert pred_mask.size(0) == gt_mask.size(0)
            num_pos = pred_mask.size(0)
            if num_pos == 0:
                h, w = img_meta['pad_shape'][:2] if cfg.upsampling else pred_mask.size()[1:3]
                final_masks.append(pred_mask.new_zeros(0, h, w))
                final_boxes.append(bbox.new_zeros(0, 4))
                continue
            # The bbox and gt_mask are in 'pad_shape', while pred_mask may be not.
            # So we need rescale the bbox and gt_mask before cropping and calculating losses.
            assert pred_mask.size(1) / img_meta['pad_shape'][0] == pred_mask.size(2) / img_meta['pad_shape'][1]
            scale_factor = pred_mask.size(1) / img_meta['pad_shape'][0]
            _bbox = bbox * scale_factor
            if scale_factor != 1:
                gt_mask = F.interpolate(gt_mask.unsqueeze(0), scale_factor=scale_factor, mode='bilinear').squeeze(0)
                gt_mask = (gt_mask > 0.5).float()
            assert gt_mask.size()[1:3] == pred_mask.size()[1:3]
            # crop_mask with size: [N, h, w], bool tensor, 1 for pixels in bbox area, 0 for others
            if cfg.crop is not None:
                crop_mask = self.crop(pred_mask, _bbox, cfg.crop)
                gt_mask[torch.bitwise_not(crop_mask)] = -1
            else:
                crop_mask = gt_mask.new_ones(gt_mask.size())

            # pred mask with size: [N, 2, mask_h, mask_w]
            pred_mask = pred_mask.permute(0, 3, 1, 2)
            mask_loss = self.mask_loss(pred_mask, gt_mask, crop_mask, cfg.mask_loss)
            losses += mask_loss
            total_pos_num += num_pos

            # prepare final_mask and final_bbox
            if cfg.upsampling:
                _bbox = bbox
                pred_mask = F.interpolate(pred_mask, img_meta['pad_shape'][:2], mode='bilinear')
            # final_mask should be cropped according to test_cfg
            test_crop_mask = self.crop(pred_mask.permute(0, 2, 3, 1), _bbox, cfg.test_crop)
            final_mask = F.softmax(pred_mask, dim=1)[:, 1, :, :] * test_crop_mask.float()
            final_masks.append(final_mask)
            final_boxes.append(_bbox)
        # avoid dividing by zero
        losses = losses / (total_pos_num + 1) * cfg.mask_loss['loss_weight']
        return dict(loss_mask=losses), final_masks, final_boxes

    def mask_loss(self, pred_mask, gt_mask, crop_mask, cfg):
        """Apply mask loss.

        Args:
            pred_mask (Tensor): with size [N, h, w, 2].
            gt_mask (Tensor): with size [N, h, w]; values in {-1, 0, 1}, for ignore, background and foreground.
            crop_mask (Tensor): with size [N, h, w]; values in {0, 1}, for outside-bbox-area and inside-bbox-area.
            cfg (dict): config of mask loss, possible keys are: use_mask_ohem, pos_neg_ratio.

        Returns:
            Tensor: Processed loss values.
        """
        num_pos, h, w = gt_mask.size()
        pre_loss = F.cross_entropy(pred_mask, gt_mask.long(), reduction='none', ignore_index=-1)

        pre_loss = pre_loss.view(num_pos, -1)
        gt_mask = gt_mask.view(num_pos, -1)
        crop_mask = crop_mask.view(num_pos, -1)
        if cfg['use_mask_ohem']:
            pre_loss, crop_mask = self.mask_ohem_loss(pre_loss, gt_mask, crop_mask, cfg['pos_neg_ratio'])

        pre_loss = pre_loss.sum(dim=1) / (crop_mask.sum(dim=1).float() + 1)
        return pre_loss.sum()

    def mask_ohem_loss(self, pre_loss, gt_mask, crop_mask, pos_neg_ratio):
        loss_c = pre_loss.clone()
        # filter out non-neg pixels
        non_neg = (gt_mask != 0)
        loss_c[non_neg] = 0
        # set non-topk loss of neg-pixels to 0
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = (gt_mask == 1).sum(1, keepdim=True)
        num_neg = torch.clamp(pos_neg_ratio * num_pos, max=gt_mask.size(1) - 1)
        filtered = idx_rank >= num_neg.expand_as(idx_rank)
        filtered[non_neg] = 0
        pre_loss[filtered] = 0
        crop_mask[filtered] = 0
        return pre_loss, crop_mask

    def crop(self, masks, bboxes, cfg):
        n, h, w = masks.size()[:3]
        x1, x2 = sanitize_coordinates(bboxes[:, 0], bboxes[:, 2], w, cfg['padding'], cast=True)
        y1, y2 = sanitize_coordinates(bboxes[:, 1], bboxes[:, 3], h, cfg['padding'], cast=True)

        rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, 1, -1).expand(n, h, w)
        cols = torch.arange(h, device=masks.device, dtype=y1.dtype).view(1, -1, 1).expand(n, h, w)

        masks_left = rows >= x1.view(-1, 1, 1)
        masks_right = rows < x2.view(-1, 1, 1)
        masks_up = cols >= y1.view(-1, 1, 1)
        masks_down = cols < y2.view(-1, 1, 1)

        crop_mask = masks_left * masks_right * masks_up * masks_down
        return crop_mask

    def get_masks(self, mask_preds, bbox_preds, img_metas, cfg, rescale=False):
        mask_pred_list = []
        for mask_pred, bbox_pred, img_meta in zip(mask_preds, bbox_preds, img_metas):
            if mask_pred.size(0) == 0:
                h, w = img_meta['ori_shape'][:2] if rescale else img_meta['img_shape'][:2]
                mask_pred = mask_pred.new_zeros(0, h, w)
                mask_pred_list.append(mask_pred)
                continue
            mask_pred = mask_pred.permute(0, 3, 1, 2)
            # upsampling
            if mask_pred.size()[2:] != img_meta['pad_shape'][:2]:
                mask_pred = F.interpolate(mask_pred, img_meta['pad_shape'][:2], mode='bilinear')
            # unpadding
            h, w, _ = img_meta['img_shape']
            mask_pred = mask_pred[:, :, :h, :w]
            if rescale:
                mask_pred = F.interpolate(mask_pred, img_meta['ori_shape'][:2], mode='bilinear', align_corners=True)

            mask_pred = F.softmax(mask_pred, dim=1)
            mask_pred = mask_pred[:, 1, :, :]
            if cfg.crop is not None:
                crop_mask = self.crop(mask_pred, bbox_pred, cfg.crop)
                mask_pred = mask_pred * crop_mask.float()
            mask_pred_list.append(mask_pred)
        return mask_pred_list



