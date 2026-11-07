# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']
        self.loss_dict = {}
        self.use_dir = False

        quality_cfg = args.get('iou_quality_head', {})
        self.iou_quality_enabled = quality_cfg.get('enabled', False)
        self.iou_loss_weight = float(quality_cfg.get('loss_weight', 1.0))
        self.box_order = args.get('box_order', 'hwl')
        self.iou_anchors = None
        if self.iou_quality_enabled:
            if self.box_order != 'hwl':
                raise ValueError('IoU quality loss currently requires hwl boxes')
            self.iou_anchors = self._generate_anchors(args['anchor_args'])

    @staticmethod
    def _generate_anchors(anchor_args):
        """Generate the same H-W-anchor grid used by VoxelPostprocessor."""
        feature_stride = anchor_args.get('feature_stride', 2)
        width = anchor_args['W'] // feature_stride
        height = anchor_args['H'] // feature_stride
        x_range = anchor_args['cav_lidar_range'][0::3]
        y_range = anchor_args['cav_lidar_range'][1::3]
        x = np.linspace(
            x_range[0] + anchor_args['vw'],
            x_range[1] - anchor_args['vw'],
            width,
        )
        y = np.linspace(
            y_range[0] + anchor_args['vh'],
            y_range[1] - anchor_args['vh'],
            height,
        )
        cx, cy = np.meshgrid(x, y)
        rotations = [math.radians(value) for value in anchor_args['r']]
        anchors = np.zeros((height, width, len(rotations), 7), dtype=np.float32)
        anchors[..., 0] = cx[..., None]
        anchors[..., 1] = cy[..., None]
        anchors[..., 2] = -1.0
        anchors[..., 3] = anchor_args['h']
        anchors[..., 4] = anchor_args['w']
        anchors[..., 5] = anchor_args['l']
        anchors[..., 6] = np.asarray(rotations, dtype=np.float32)
        return torch.from_numpy(anchors)

    @staticmethod
    def decode_deltas(deltas, anchors):
        """Decode hwl PointPillar deltas with broadcasting over batch."""
        anchors = anchors.reshape(1, -1, 7)
        anchors_d = torch.sqrt(anchors[..., 4] ** 2 + anchors[..., 5] ** 2)
        boxes = deltas.clone()
        boxes[..., 0] = deltas[..., 0] * anchors_d + anchors[..., 0]
        boxes[..., 1] = deltas[..., 1] * anchors_d + anchors[..., 1]
        boxes[..., 2] = deltas[..., 2] * anchors[..., 3] + anchors[..., 2]
        boxes[..., 3:6] = torch.exp(deltas[..., 3:6]) * anchors[..., 3:6]
        boxes[..., 6] = deltas[..., 6] + anchors[..., 6]
        return boxes

    @staticmethod
    def scale_iou_target(iou_target):
        return 2.0 * iou_target - 1.0

    @staticmethod
    def _hwl_to_lwh(boxes):
        return boxes[..., [0, 1, 2, 5, 4, 3, 6]]

    def build_iou_targets(self, box_deltas, reg_targets, positives,
                          iou_calculator=None):
        """Build detached scaled IoU targets for positive anchors only."""
        anchors = self.iou_anchors.to(
            device=box_deltas.device, dtype=box_deltas.dtype
        )
        if anchors.numel() // 7 != box_deltas.shape[1]:
            raise ValueError(
                'IoU anchor count ({}) does not match prediction count ({})'
                .format(anchors.numel() // 7, box_deltas.shape[1])
            )

        pred_boxes = self.decode_deltas(box_deltas.detach(), anchors)
        gt_boxes = self.decode_deltas(
            reg_targets.to(dtype=box_deltas.dtype), anchors
        )
        pred_positive = self._hwl_to_lwh(pred_boxes[positives])
        gt_positive = self._hwl_to_lwh(gt_boxes[positives])

        if iou_calculator is None:
            from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import (
                aligned_boxes_iou3d_gpu,
            )
            iou_calculator = aligned_boxes_iou3d_gpu

        iou_target = iou_calculator(pred_positive, gt_positive)
        iou_target = iou_target.reshape(-1).clamp(0.0, 1.0).detach()
        return self.scale_iou_target(iou_target)

    def forward(self, output_dict, target_dict, prefix=''):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        rm = output_dict['rm{}'.format(prefix)]  # [B, 14, 50, 176]
        psm = output_dict['psm{}'.format(prefix)] # [B, 2, 50, 176]
        targets = target_dict['targets']

        cls_preds = psm.permute(0, 2, 3, 1).contiguous() # N, C, H, W -> N, H, W, C

        box_cls_labels = target_dict['pos_equal_one']  # [B, 50, 176, 2]
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), 2,
            dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm,
                                                                 targets)
        loc_loss_src =\
            self.reg_loss_func(box_preds_sin,
                               reg_targets_sin,
                               weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        total_loss = reg_loss + conf_loss

        if self.iou_quality_enabled:
            iou_pred = output_dict['iou{}'.format(prefix)]
            iou_pred = iou_pred.permute(0, 2, 3, 1).contiguous()
            iou_pred = iou_pred.view(iou_pred.shape[0], -1)
            if iou_pred.shape != positives.shape:
                raise ValueError(
                    'IoU prediction shape {} does not match labels {}'
                    .format(tuple(iou_pred.shape), tuple(positives.shape))
                )

            if positives.any():
                iou_target = self.build_iou_targets(
                    rm.detach(), targets, positives
                )
                iou_loss_src = F.smooth_l1_loss(
                    iou_pred[positives], iou_target, reduction='none'
                )
                iou_loss = (
                    iou_loss_src * reg_weights[positives]
                ).sum() / rm.shape[0]
            else:
                iou_loss = iou_pred.sum() * 0.0
            iou_loss = iou_loss * self.iou_loss_weight
            total_loss = total_loss + iou_loss
        else:
            iou_loss = total_loss.new_zeros(())

        self.loss_dict.update({'total_loss{}'.format(prefix): total_loss,
                               'reg_loss{}'.format(prefix): reg_loss,
                               'conf_loss{}'.format(prefix): conf_loss,
                               'iou_loss{}'.format(prefix): iou_loss})

        return total_loss

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2


    def logging(self, epoch, batch_id, batch_len, writer = None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = [v.item() for k, v in self.loss_dict.items() if 'total_loss' in k]
        if len(total_loss) > 1:
            total_loss = sum(total_loss)
        else:
            total_loss = total_loss[0]
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']

        print_msg = "[epoch {}][{}/{}], || Loss: {:.2f} ||".format(epoch, batch_id + 1, batch_len, total_loss)
        for k, v in self.loss_dict.items():
            print_msg += '{}: {:.2f} | '.format(k.replace('_loss', '').replace('_single', ''), v.item())

        # print_msg = ("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
        #             " || Loc Loss: %.4f" % (
        #                 epoch, batch_id + 1, batch_len,
        #                 total_loss.item(), conf_loss.item(), reg_loss.item()))
        
        if self.use_dir:
            dir_loss = self.loss_dict['dir_loss']
            print_msg += " || Dir Loss: %.4f" % dir_loss.item()

        print(print_msg)

        if not writer is None:
            for k, v in self.loss_dict.items():
                writer.add_scalar(k, v.item(), epoch*batch_len + batch_id)
            # writer.add_scalar('Regression_loss', reg_loss.item(),
            #                 epoch*batch_len + batch_id)
            # writer.add_scalar('Confidence_loss', conf_loss.item(),
            #                 epoch*batch_len + batch_id)
                            
            if self.use_dir:
                writer.add_scalar('dir_loss', dir_loss.item(),
                            epoch*batch_len + batch_id)
