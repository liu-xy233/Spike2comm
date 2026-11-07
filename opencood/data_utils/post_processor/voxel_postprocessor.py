# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
from torch.nn.functional import sigmoid
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.data_utils.post_processor.confidence_aware_late_fusion import (
    ConfidenceAwareLateFusion,
    rectify_iou_aware_scores,
)
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils
from opencood.utils.common_utils import limit_period


class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(VoxelPostprocessor, self).__init__(anchor_params, train)
        self.anchor_num = self.params['anchor_args']['num']
        self._iou_debug_printed = False

    def generate_anchor_box(self):
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w']
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        vh = self.params['anchor_args']['vh'] # voxel_size
        vw = self.params['anchor_args']['vw']

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride']
        else:
            feature_stride = 2


        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)


        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num) # center
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        if self.params['order'] == 'hwl': # pointpillar
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1) # (50, 176, 2, 7)

        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        return anchors

    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        # (max_num, 7)
        gt_box_center = kwargs['gt_box_center']
        # (H, W, anchor_num, 7)
        anchors = kwargs['anchors']
        # (max_num)
        masks = kwargs['mask']

        # (H, W)
        feature_map_shape = anchors.shape[:2]

        # (H*W*anchor_num, 7)
        anchors = anchors.reshape(-1, 7)
        # normalization factor, (H * W * anchor_num)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        # (H, W, 2)
        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        # (H, W, self.anchor_num * 7)
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        # (n, 7)
        gt_box_center_valid = gt_box_center[masks == 1]
        # (n, 8, 3)
        gt_box_corner_valid = \
            box_utils.boxes_to_corners_3d(gt_box_center_valid,
                                          self.params['order'])
        # (H*W*anchor_num, 8, 3)
        anchors_corner = \
            box_utils.boxes_to_corners_3d(anchors,
                                          order=self.params['order'])
        # (H*W*anchor_num, 4)
        anchors_standup_2d = \
            box_utils.corner2d_to_standup_box(anchors_corner)
        # (n, 4)
        gt_standup_2d = \
            box_utils.corner2d_to_standup_box(gt_box_corner_valid)

        # (H*W*anchor_n)
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        # the anchor boxes has the largest iou across
        # shape: (n)
        id_highest = np.argmax(iou.T, axis=1)
        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]


        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = \
            np.where(iou >
                     self.params['target_args']['pos_threshold'])
        #  find anchors iou  params['neg_iou']
        id_neg = np.where(np.sum(iou <
                                 self.params['target_args']['neg_threshold'],
                                 axis=1) == iou.shape[1])[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1

        # calculate the targets
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[
                id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0


        label_dict = {'pos_equal_one': pos_equal_one,
                      'neg_equal_one': neg_equal_one,
                      'targets': targets}

        return label_dict

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = \
            torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = \
            torch.from_numpy(np.array(neg_equal_one))
        targets = \
            torch.from_numpy(np.array(targets))

        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one}

    def post_process(self, data_dict, output_dict,
                     return_nms_candidates=False):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        late_fusion_cfg = self.params.get(
            'confidence_aware_late_fusion', {}
        )
        if late_fusion_cfg.get('enabled', False):
            return self._post_process_confidence_aware(
                data_dict, output_dict, late_fusion_cfg
            )

        quality_cfg = self.params.get('iou_quality_head', {})
        quality_enabled = quality_cfg.get('enabled', False)
        quality_inference_mode = quality_cfg.get('inference_mode', 'full')
        if (quality_enabled
                and quality_inference_mode not in ('full', 'nms_only')):
            raise ValueError(
                "iou_quality_head.inference_mode must be 'full' or "
                "'nms_only', got '{}'".format(quality_inference_mode)
            )
        nms_only = quality_enabled and quality_inference_mode == 'nms_only'

        # Keep the existing FULL/OFF behavior while allowing NMS-only to carry
        # a separate ranking score through the same candidate pipeline.
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        pred_nms_score_list = []
        pred_raw_score_list = []
        pred_quality_list = []
        pred_candidate_index_list = []
        candidate_index_offset = 0

        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix'] # no clean

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # classification probability
            raw_prob = output_dict[cav_id]['psm']
            raw_prob = F.sigmoid(raw_prob.permute(0, 2, 3, 1))
            raw_prob = raw_prob.reshape(1, -1)
            nms_prob = raw_prob
            predicted_quality = torch.ones_like(raw_prob)
            if quality_enabled:
                if 'iou' not in output_dict[cav_id]:
                    raise KeyError(
                        "IoU-aware post-processing requires the 'iou' output"
                    )
                iou_pred = output_dict[cav_id]['iou']
                iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(1, -1)
                predicted_quality = ((iou_pred + 1.0) * 0.5).clamp(0.0, 1.0)
                nms_prob = rectify_iou_aware_scores(
                    raw_prob,
                    iou_pred,
                    quality_power=quality_cfg.get('quality_power', 4.0),
                )

            threshold_prob = raw_prob if nms_only else nms_prob
            output_prob = raw_prob if nms_only else nms_prob

            # regression map
            reg = output_dict[cav_id]['rm']

            # convert regression map back to bounding box
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
            mask = \
                torch.gt(threshold_prob,
                         self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(output_prob[0], mask[0])
            nms_scores = torch.masked_select(nms_prob[0], mask[0])
            raw_scores = torch.masked_select(raw_prob[0], mask[0])
            qualities = torch.masked_select(predicted_quality[0], mask[0])
            candidate_indices = torch.arange(
                raw_prob.shape[1], device=raw_prob.device, dtype=torch.long
            ) + candidate_index_offset
            candidate_indices = candidate_indices[mask[0]]
            candidate_index_offset += raw_prob.shape[1]

            # adding dir classifier
            if 'dm' in output_dict[cav_id].keys() and len(boxes3d) !=0:
                dir_offset = self.params['dir_args']['dir_offset']
                num_bins = self.params['dir_args']['num_bins']


                dm  = output_dict[cav_id]['dm'] # [N, H, W, 4]
                dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
                
                period = (2 * np.pi / num_bins) # pi
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                
                # STEP 2
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)
                pred_nms_score_list.append(nms_scores)
                pred_raw_score_list.append(raw_scores)
                pred_quality_list.append(qualities)
                pred_candidate_index_list.append(candidate_indices)

        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            if return_nms_candidates:
                return None
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        nms_scores = torch.cat(pred_nms_score_list, dim=0)
        raw_scores = torch.cat(pred_raw_score_list, dim=0)
        predicted_quality = torch.cat(pred_quality_list, dim=0)
        candidate_indices = torch.cat(pred_candidate_index_list, dim=0)
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        nms_scores = nms_scores[keep_index]
        raw_scores = raw_scores[keep_index]
        predicted_quality = predicted_quality[keep_index]
        candidate_indices = candidate_indices[keep_index]

        if return_nms_candidates:
            return {
                'boxes': pred_box3d_tensor,
                'raw_scores': raw_scores,
                'quality_scores': nms_scores,
                'predicted_quality': predicted_quality,
                'candidate_indices': candidate_indices,
            }

        # STEP3
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           nms_scores,
                                           self.params['nms_thresh']
                                           )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]
        nms_scores = nms_scores[keep_index]

        # filter out the prediction out of the range.
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor, self.params['gt_range'])
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]
        nms_scores = nms_scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]
        assert nms_scores.shape[0] == pred_box3d_tensor.shape[0]

        # return pred_box3d_tensor, scores, count
        return pred_box3d_tensor, scores

    def _decode_candidates(self, psm, rm, anchor_box,
                           transformation_matrix, score_threshold=None,
                           iou_preds=None, iou_quality_cfg=None,
                           return_raw_scores=False):
        """Decode anchor predictions without NMS.

        ``psm`` and ``rm`` may contain one or more agents.  A single 4x4
        transform is accepted only for one prediction map; otherwise one
        transform per map is required.
        """
        if psm.shape[0] != rm.shape[0]:
            raise ValueError("Classification and regression batch sizes differ")

        raw_probabilities = torch.sigmoid(
            psm.permute(0, 2, 3, 1)
        ).reshape(psm.shape[0], -1)
        quality_cfg = iou_quality_cfg or {}
        quality_enabled = quality_cfg.get('enabled', False)
        if quality_enabled:
            if iou_preds is None:
                raise ValueError(
                    "IoU predictions are required for quality-aware decoding"
                )
            flat_iou = iou_preds.permute(0, 2, 3, 1).reshape(
                iou_preds.shape[0], -1
            )
            probabilities = rectify_iou_aware_scores(
                raw_probabilities,
                flat_iou,
                enabled=True,
                quality_power=quality_cfg.get('quality_power', 4.0),
            )
        else:
            probabilities = raw_probabilities
        batch_boxes = self.delta_to_boxes3d(rm, anchor_box)

        if transformation_matrix.ndim == 2:
            if psm.shape[0] != 1:
                raise ValueError(
                    "One transformation matrix cannot decode multiple agents"
                )
            transformations = transformation_matrix.unsqueeze(0)
        elif transformation_matrix.ndim == 3:
            transformations = transformation_matrix
        else:
            raise ValueError("Transformation matrices must have rank 2 or 3")

        if transformations.shape[0] != psm.shape[0]:
            raise ValueError("Prediction and transformation counts differ")

        decoded = []
        for agent_idx in range(psm.shape[0]):
            scores = probabilities[agent_idx]
            raw_scores = raw_probabilities[agent_idx]
            centers = batch_boxes[agent_idx]
            if score_threshold is not None:
                keep = scores > score_threshold
                scores = scores[keep]
                raw_scores = raw_scores[keep]
                centers = centers[keep]

            if scores.numel() == 0:
                empty_boxes = centers.new_empty((0, 8, 3))
                if return_raw_scores:
                    decoded.append((empty_boxes, raw_scores, scores))
                else:
                    decoded.append((empty_boxes, scores))
                continue

            corners = box_utils.boxes_to_corners_3d(
                centers, order=self.params['order']
            )
            projected = box_utils.project_box3d(
                corners, transformations[agent_idx]
            )
            if return_raw_scores:
                decoded.append((projected, raw_scores, scores))
            else:
                decoded.append((projected, scores))
        return decoded

    def _print_iou_debug_once(self, predictions, agent_count, final_count,
                              quality_cfg):
        if (self._iou_debug_printed
                or not quality_cfg.get('enabled', False)
                or not quality_cfg.get('debug', False)):
            return
        cls_parts = [predictions['psm'].sigmoid().reshape(-1)]
        iou_parts = [predictions['iou'].reshape(-1)]
        if agent_count > 1:
            cls_parts.append(
                predictions['psm_single'][1:agent_count].sigmoid().reshape(-1)
            )
            iou_parts.append(
                predictions['iou_single'][1:agent_count].reshape(-1)
            )
        cls_scores = torch.cat(cls_parts).detach()
        iou_raw = torch.cat(iou_parts).detach()
        iou_quality = ((iou_raw + 1.0) * 0.5).clamp(0.0, 1.0)
        quality_scores = rectify_iou_aware_scores(
            cls_scores,
            iou_raw,
            quality_power=quality_cfg.get('quality_power', 4.0),
        )

        def stats(tensor):
            return "{:.6f}/{:.6f}/{:.6f}".format(
                tensor.mean().item(), tensor.min().item(), tensor.max().item()
            )

        print("[iou-quality]")
        print("cls score mean/min/max: {}".format(stats(cls_scores)))
        print("iou raw mean/min/max: {}".format(stats(iou_raw)))
        print("iou quality mean/min/max: {}".format(stats(iou_quality)))
        print("quality score mean/min/max: {}".format(stats(quality_scores)))
        print("num candidates before: {}".format(cls_scores.numel()))
        print("num candidates after: {}".format(final_count))
        self._iou_debug_printed = True

    def _post_process_confidence_aware(self, data_dict, output_dict,
                                       late_fusion_cfg):
        """NMS-before-free intermediate/local candidate hybridization."""
        if set(data_dict.keys()) != {'ego'} or 'ego' not in output_dict:
            raise ValueError(
                "Confidence-aware late fusion expects intermediate-fusion "
                "data with a single top-level 'ego' entry"
            )

        cav_content = data_dict['ego']
        predictions = output_dict['ego']
        required_output_keys = {'psm', 'rm', 'psm_single', 'rm_single'}
        quality_cfg = self.params.get('iou_quality_head', {})
        if quality_cfg.get('enabled', False):
            required_output_keys.update({'iou', 'iou_single'})
        missing = required_output_keys.difference(predictions.keys())
        if missing:
            raise KeyError(
                "Confidence-aware late fusion is missing model outputs: {}"
                .format(sorted(missing))
            )

        record_len = cav_content['record_len']
        if record_len.numel() != 1:
            raise ValueError(
                "Confidence-aware inference follows the existing evaluation "
                "contract and requires batch size 1"
            )
        agent_count = int(record_len.reshape(-1)[0].item())
        if predictions['psm_single'].shape[0] != agent_count:
            raise ValueError(
                "Packed local prediction count ({}) does not match record_len "
                "({})".format(
                    predictions['psm_single'].shape[0], agent_count
                )
            )

        anchor_box = cav_content['anchor_box']
        base_score_threshold = self.params['target_args']['score_threshold']

        ego_transform = cav_content['transformation_matrix'].to(
            device=predictions['rm'].device,
            dtype=predictions['rm'].dtype,
        )
        ego_boxes, ego_scores = self._decode_candidates(
            predictions['psm'],
            predictions['rm'],
            anchor_box,
            ego_transform,
            score_threshold=base_score_threshold,
            iou_preds=predictions.get('iou'),
            iou_quality_cfg=quality_cfg,
        )[0]

        collaborator_predictions = []
        if agent_count > 1:
            pairwise = cav_content['pairwise_t_matrix']
            # Dataset convention: pairwise[i, j] maps agent i coordinates to
            # agent j coordinates.  Vehicle/agent 0 is the fixed ego.
            collaborator_transforms = pairwise[0, 1:agent_count, 0].to(
                device=predictions['rm_single'].device,
                dtype=predictions['rm_single'].dtype,
            )
            collaborator_predictions = self._decode_candidates(
                predictions['psm_single'][1:agent_count],
                predictions['rm_single'][1:agent_count],
                anchor_box,
                collaborator_transforms,
                score_threshold=None,
                iou_preds=predictions.get('iou_single', None)[1:agent_count]
                if predictions.get('iou_single', None) is not None else None,
                iou_quality_cfg=quality_cfg,
                return_raw_scores=True,
            )

        late_fuser = ConfidenceAwareLateFusion(
            score_threshold=late_fusion_cfg['score_threshold'],
            collaborator_score_weight=(
                late_fusion_cfg['collaborator_score_weight']
            ),
            base_score_threshold=base_score_threshold,
            nms_threshold=self.params['nms_thresh'],
            nms_method=late_fusion_cfg.get('nms_method', 'hard'),
            soft_nms_sigma=late_fusion_cfg.get('soft_nms_sigma', 0.5),
            enable_rotated_box_voting=late_fusion_cfg.get(
                'enable_rotated_box_voting', None
            ),
            rbv_iou_threshold=late_fusion_cfg.get(
                'rbv_iou_threshold', None
            ),
            rbv_score_power=late_fusion_cfg.get(
                'rbv_score_power', None
            ),
            rbv_iou_power=late_fusion_cfg.get(
                'rbv_iou_power', None
            ),
        )
        pred_box3d_tensor, scores = late_fuser(
            ego_boxes, ego_scores, collaborator_predictions
        )
        if pred_box3d_tensor is None:
            self._print_iou_debug_once(
                predictions, agent_count, 0, quality_cfg
            )
            return None, None

        range_keep = box_utils.get_mask_for_boxes_within_range_torch(
            pred_box3d_tensor, self.params['gt_range']
        )
        pred_box3d_tensor = pred_box3d_tensor[range_keep]
        scores = scores[range_keep]
        assert scores.shape[0] == pred_box3d_tensor.shape[0]
        self._print_iou_debug_once(
            predictions, agent_count, scores.numel(), quality_cfg
        )
        return pred_box3d_tensor, scores

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)?? should be (N, 14, H, W)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    show_vis,
                                                    save_path)
