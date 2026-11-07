# -*- coding: utf-8 -*-
"""Confidence-aware detection-level late fusion.

This module deliberately operates on decoded boxes.  It does not depend on a
backbone, feature communication, or a particular intermediate fusion method.
"""

import math

import torch

from opencood.utils import box_utils
from opencood.utils import common_utils


ENABLE_ROTATED_BOX_VOTING = False
RBV_IOU_THRESHOLD = 0.5
RBV_SCORE_POWER = 1.0
RBV_IOU_POWER = 1.0


def rectify_iou_aware_scores(cls_scores, iou_preds, enabled=True,
                             quality_power=4.0):
    """Rectify classification scores with predicted localization quality."""
    if not enabled:
        return cls_scores
    if iou_preds is None:
        raise ValueError("IoU predictions are required when quality is enabled")
    if cls_scores.shape != iou_preds.shape:
        raise ValueError("Classification and IoU predictions must align")
    iou_quality = ((iou_preds + 1.0) * 0.5).clamp(0.0, 1.0)
    return cls_scores * iou_quality.pow(float(quality_power))


def pairwise_rotated_bev_iou(reference_boxes, candidate_boxes):
    """Compute rotated BEV IoU for corner boxes as a [K, N] tensor."""
    if reference_boxes.ndim != 3 or reference_boxes.shape[1:] != (8, 3):
        raise ValueError("Reference boxes must have shape [K, 8, 3]")
    if candidate_boxes.ndim != 3 or candidate_boxes.shape[1:] != (8, 3):
        raise ValueError("Candidate boxes must have shape [N, 8, 3]")
    if reference_boxes.shape[0] == 0 or candidate_boxes.shape[0] == 0:
        return reference_boxes.new_zeros(
            (reference_boxes.shape[0], candidate_boxes.shape[0])
        )

    reference_polygons = common_utils.convert_format(
        reference_boxes.detach().cpu().numpy()
    )
    candidate_polygons = common_utils.convert_format(
        candidate_boxes.detach().cpu().numpy()
    )
    iou_rows = [
        common_utils.compute_iou(reference, candidate_polygons)
        for reference in reference_polygons
    ]
    iou_matrix = torch.stack(
        [reference_boxes.new_tensor(row) for row in iou_rows], dim=0
    )
    return torch.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def rotated_soft_nms(boxes, scores, score_threshold, sigma=0.5):
    """Gaussian Soft-NMS on rotated BEV polygons from ``[N, 8, 3]`` corners.

    Return original input indices (long tensor) and decayed scores, both on
    the input device, in greedy selection order. Scores must be strictly
    greater than ``score_threshold``, matching the detector candidate gate.
    Ties select the lowest original index. Inputs are never modified.
    All remaining boxes receive Gaussian decay; no hard IoU cutoff or top-k
    truncation is applied. Invalid/non-finite inputs raise ValueError.
    """
    if boxes.ndim != 3 or boxes.shape[1:] != (8, 3):
        raise ValueError("Boxes must have shape [N, 8, 3]")
    if scores.ndim != 1 or boxes.shape[0] != scores.shape[0]:
        raise ValueError("Scores must have shape [N] and align with boxes")
    if boxes.device != scores.device:
        raise ValueError("Boxes and scores must be on the same device")
    if not boxes.is_floating_point() or not scores.is_floating_point():
        raise ValueError("Boxes and scores must be floating-point tensors")
    sigma = float(sigma)
    score_threshold = float(score_threshold)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Soft-NMS sigma must be finite and positive")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("Soft-NMS score threshold must be in [0, 1]")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("Boxes and scores must be finite")

    decayed_scores = scores.clone()
    # Keep remaining indices in input order so argmax ties are reproducible.
    remaining = torch.arange(scores.numel(), device=scores.device)
    remaining = remaining[decayed_scores > score_threshold]
    if remaining.numel() == 0:
        return remaining, decayed_scores[remaining]

    # Reuse exactly the polygon conversion and rotated IoU used by Hard NMS.
    # Convert once, and compute only one IoU row per iteration (O(N) memory).
    polygons = common_utils.convert_format(
        boxes.detach().to(device="cpu", dtype=torch.float64).numpy()
    )
    if any(not polygon.is_valid or polygon.area <= 0 for polygon in polygons):
        raise ValueError("Boxes must define valid, positive-area BEV polygons")

    selected = []
    while remaining.numel() > 0:
        best_position = torch.argmax(decayed_scores[remaining])
        best_index = remaining[best_position]
        selected.append(best_index)
        remaining = remaining[remaining != best_index]
        if remaining.numel() == 0:
            break
        iou = common_utils.compute_iou(
            polygons[best_index.item()],
            polygons[remaining.cpu().numpy()],
        )
        # Double precision also avoids zero/zero for very small positive sigma.
        iou = torch.as_tensor(iou, device=scores.device, dtype=torch.float64)
        decay = torch.exp(-iou.square() / sigma).to(dtype=scores.dtype)
        decayed_scores[remaining] *= decay
        remaining = remaining[decayed_scores[remaining] > score_threshold]

    keep = torch.stack(selected)
    return keep, decayed_scores[keep]


def rotated_box_voting(nms_boxes, candidate_boxes, candidate_scores,
                       iou_threshold=RBV_IOU_THRESHOLD,
                       score_power=RBV_SCORE_POWER,
                       iou_power=RBV_IOU_POWER):
    """Refine only NMS box centers using uniquely assigned pre-NMS boxes."""
    if nms_boxes.ndim != 3 or nms_boxes.shape[1:] != (8, 3):
        raise ValueError("NMS boxes must have shape [K, 8, 3]")
    if candidate_boxes.ndim != 3 or candidate_boxes.shape[1:] != (8, 3):
        raise ValueError("Candidate boxes must have shape [N, 8, 3]")
    if candidate_boxes.shape[0] != candidate_scores.shape[0]:
        raise ValueError("Candidate boxes and scores must have equal length")
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("RBV IoU threshold must be in [0, 1]")
    if float(score_power) < 0.0 or float(iou_power) < 0.0:
        raise ValueError("RBV score and IoU powers must be non-negative")
    if nms_boxes.shape[0] == 0 or candidate_boxes.shape[0] == 0:
        return nms_boxes

    iou_matrix = pairwise_rotated_bev_iou(nms_boxes, candidate_boxes)
    best_iou, owner = iou_matrix.max(dim=0)
    eligible = best_iou >= float(iou_threshold)
    candidate_centers = candidate_boxes.mean(dim=1)
    voted_boxes = nms_boxes.clone()

    for nms_index in range(nms_boxes.shape[0]):
        assigned = eligible & (owner == nms_index)
        if not assigned.any():
            continue
        assigned_iou = iou_matrix[nms_index, assigned]
        weights = (
            candidate_scores[assigned].clamp(min=0).pow(float(score_power))
            * assigned_iou.clamp(min=0).pow(float(iou_power))
        )
        weight_sum = weights.sum()
        if weight_sum <= 0:
            continue
        voted_center = (
            candidate_centers[assigned] * weights.unsqueeze(-1)
        ).sum(dim=0) / weight_sum
        current_center = nms_boxes[nms_index].mean(dim=0)
        voted_boxes[nms_index] += voted_center - current_center

    return voted_boxes


class ConfidenceAwareLateFusion(object):
    """Merge ego intermediate detections with weighted local detections."""

    def __init__(self, score_threshold, collaborator_score_weight,
                 base_score_threshold, nms_threshold,
                 enable_rotated_box_voting=None,
                 rbv_iou_threshold=None,
                 rbv_score_power=None,
                 rbv_iou_power=None,
                 nms_method='hard',
                 soft_nms_sigma=0.5):
        self.score_threshold = float(score_threshold)
        self.collaborator_score_weight = float(collaborator_score_weight)
        self.base_score_threshold = float(base_score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.nms_method = nms_method
        self.soft_nms_sigma = float(soft_nms_sigma)
        self.enable_rotated_box_voting = (
            ENABLE_ROTATED_BOX_VOTING
            if enable_rotated_box_voting is None
            else bool(enable_rotated_box_voting)
        )
        self.rbv_iou_threshold = float(
            RBV_IOU_THRESHOLD if rbv_iou_threshold is None else rbv_iou_threshold
        )
        self.rbv_score_power = float(
            RBV_SCORE_POWER if rbv_score_power is None else rbv_score_power
        )
        self.rbv_iou_power = float(
            RBV_IOU_POWER if rbv_iou_power is None else rbv_iou_power
        )
        self._validate()

    def _validate(self):
        if self.nms_method not in ('hard', 'soft'):
            raise ValueError("nms_method must be 'hard' or 'soft'")
        if (not math.isfinite(self.soft_nms_sigma)
                or self.soft_nms_sigma <= 0.0):
            raise ValueError("Soft-NMS sigma must be finite and positive")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(
                "confidence-aware late-fusion score_threshold must be in "
                "[0, 1], got {}".format(self.score_threshold)
            )
        if not 0.0 < self.collaborator_score_weight < 1.0:
            raise ValueError(
                "collaborator_score_weight must be in (0, 1), got {}".format(
                    self.collaborator_score_weight
                )
            )
        if not 0.0 <= self.base_score_threshold <= 1.0:
            raise ValueError(
                "base score threshold must be in [0, 1], got {}".format(
                    self.base_score_threshold
                )
            )
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise ValueError(
                "NMS threshold must be in [0, 1], got {}".format(
                    self.nms_threshold
                )
            )
        if not 0.0 <= self.rbv_iou_threshold <= 1.0:
            raise ValueError(
                "RBV IoU threshold must be in [0, 1], got {}".format(
                    self.rbv_iou_threshold
                )
            )
        if self.rbv_score_power < 0.0 or self.rbv_iou_power < 0.0:
            raise ValueError("RBV score and IoU powers must be non-negative")

    def filter_and_weight_collaborator(self, boxes, raw_scores,
                                       ranking_scores=None):
        """Apply the raw-score gate before quality and collaborator weights."""
        if boxes.shape[0] != raw_scores.shape[0]:
            raise ValueError("Collaborator boxes and scores must have equal length")
        if ranking_scores is None:
            ranking_scores = raw_scores
        if ranking_scores.shape != raw_scores.shape:
            raise ValueError("Raw and ranking scores must have equal shape")
        if raw_scores.numel() == 0:
            return boxes, ranking_scores

        raw_keep = raw_scores >= self.score_threshold
        boxes = boxes[raw_keep]
        weighted_scores = (
            ranking_scores[raw_keep] * self.collaborator_score_weight
        )

        # Match the existing OpenCOOD/CoSDH candidate gate: confidence-aware
        # filtering is based on the raw score, while the ordinary detector
        # threshold is applied to the score that participates in final NMS.
        detector_keep = weighted_scores > self.base_score_threshold
        return boxes[detector_keep], weighted_scores[detector_keep]

    @staticmethod
    def _append_nonempty(boxes_list, scores_list, boxes, scores):
        if boxes is None or scores is None or scores.numel() == 0:
            return
        if boxes.shape[0] != scores.shape[0]:
            raise ValueError("Candidate boxes and scores must have equal length")
        boxes_list.append(boxes)
        scores_list.append(scores)

    def __call__(self, ego_boxes, ego_scores, collaborator_predictions):
        """Concatenate candidates and execute exactly one rotated NMS.

        Parameters
        ----------
        ego_boxes : torch.Tensor or None
            Ego intermediate-fusion boxes in ego coordinates, ``[N, 8, 3]``.
        ego_scores : torch.Tensor or None
            Unmodified ego confidence scores, ``[N]``.
        collaborator_predictions : iterable
            ``(boxes, raw_scores)`` or ``(boxes, raw_scores, ranking_scores)``
            tuples. Boxes must already be in the ego coordinate system.
        """
        boxes_list = []
        scores_list = []
        self._append_nonempty(
            boxes_list, scores_list, ego_boxes, ego_scores
        )

        for prediction in collaborator_predictions:
            if len(prediction) == 2:
                boxes, raw_scores = prediction
                ranking_scores = raw_scores
            elif len(prediction) == 3:
                boxes, raw_scores, ranking_scores = prediction
            else:
                raise ValueError(
                    "Collaborator prediction must contain 2 or 3 tensors"
                )
            boxes, scores = self.filter_and_weight_collaborator(
                boxes, raw_scores, ranking_scores
            )
            self._append_nonempty(boxes_list, scores_list, boxes, scores)

        if not boxes_list:
            return None, None

        candidate_boxes = torch.cat(boxes_list, dim=0)
        candidate_scores = torch.cat(scores_list, dim=0)

        keep_size = box_utils.remove_large_pred_bbx(candidate_boxes)
        keep_z = box_utils.remove_bbx_abnormal_z(candidate_boxes)
        keep = torch.logical_and(keep_size, keep_z)
        candidate_boxes = candidate_boxes[keep]
        candidate_scores = candidate_scores[keep]

        if candidate_scores.numel() == 0:
            return None, None

        # This is the only NMS invocation in the enabled path.
        if self.nms_method == 'hard':
            keep = box_utils.nms_rotated(
                candidate_boxes, candidate_scores, self.nms_threshold
            )
            nms_scores = candidate_scores[keep]
        else:
            keep, nms_scores = rotated_soft_nms(
                candidate_boxes, candidate_scores,
                score_threshold=self.base_score_threshold,
                sigma=self.soft_nms_sigma,
            )
            if keep.numel() == 0:
                return None, None
        nms_boxes = candidate_boxes[keep]
        if self.enable_rotated_box_voting:
            nms_boxes = rotated_box_voting(
                nms_boxes,
                candidate_boxes,
                candidate_scores,
                iou_threshold=self.rbv_iou_threshold,
                score_power=self.rbv_score_power,
                iou_power=self.rbv_iou_power,
            )
        return nms_boxes, nms_scores
