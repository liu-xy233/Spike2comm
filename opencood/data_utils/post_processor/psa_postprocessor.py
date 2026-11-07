"""Inference-only adapter: raw detector maps -> ego corners -> official PSA.

The detector is single foreground class (anchor channels are NOT classes).
No network, parameter, communication mask, quality weight, NMS or RBV is added.
The baseline postprocessor never imports this module.
"""

import time

import numpy as np
import torch

from opencood.utils import box_utils
from opencood.utils.psa_utils import promote_suppress_aggregation


PSA_MODES = ('psa_strict', 'psa_hybrid')


def _validate_transform(transform):
    if transform.shape != (4, 4):
        raise ValueError("PSA requires finite rigid 4x4 transforms in meters")

    identity = torch.eye(3, device=transform.device, dtype=transform.dtype)
    bottom = transform.new_tensor([0, 0, 0, 1])
    rotation = transform[:3, :3]

    # Explicit 3x3 determinant avoids the CUDA cuSOLVER dependency
    # in torch.linalg.det, which is problematic in some older builds.
    det_rotation = (
        rotation[0, 0] * (
            rotation[1, 1] * rotation[2, 2]
            - rotation[1, 2] * rotation[2, 1]
        )
        - rotation[0, 1] * (
            rotation[1, 0] * rotation[2, 2]
            - rotation[1, 2] * rotation[2, 0]
        )
        + rotation[0, 2] * (
            rotation[1, 0] * rotation[2, 1]
            - rotation[1, 1] * rotation[2, 0]
        )
    )

    if (not torch.isfinite(transform).all()
            or not torch.allclose(transform[3], bottom, atol=1e-4, rtol=0)
            or not torch.allclose(rotation.T @ rotation, identity, atol=1e-3, rtol=0)
            or not torch.allclose(det_rotation, transform.new_tensor(1.),
                                  atol=1e-3, rtol=0)):
        raise ValueError("PSA requires finite rigid 4x4 transforms in meters")




def decode_psa_candidates(postprocessor, cav_content, predictions, mode):
    """Return one candidate dict per scene, never mix packed agent dimensions.

    Accepts shared [H,W,A,7] or per-scene [B,H,W,A,7] anchors. The existing
    dataset/evaluation entry still requires B=1; this function also supports
    synthetic B>1 calls. Source IDs: -1=fused, 0=local ego, 1..=local neighbors.
    """
    if mode not in PSA_MODES:
        raise ValueError("Unknown PSA mode: {}".format(mode))
    record_len = cav_content['record_len']
    if (record_len.ndim != 1 or record_len.numel() == 0
            or record_len.dtype not in (torch.int32, torch.int64)
            or torch.any(record_len < 1)):
        raise ValueError("record_len must contain positive integer agent counts")
    counts = record_len.detach().cpu().tolist()
    total_agents, batch_size = sum(counts), len(counts)
    keys = ['psm_single', 'rm_single']
    if mode == 'psa_hybrid':
        keys += ['psm', 'rm']
    for key in keys:
        if key not in predictions:
            raise KeyError("PSA requires raw model output '{}'".format(key))
        value = predictions[key]
        expected = total_agents if key.endswith('_single') else batch_size
        if value.ndim != 4 or value.shape[0] != expected:
            raise ValueError("{} batch dimension disagrees with record_len".format(key))
    for suffix in (['_single', ''] if mode == 'psa_hybrid' else ['_single']):
        psm, rm = predictions['psm' + suffix], predictions['rm' + suffix]
        if rm.shape[1] != 7 * psm.shape[1] or rm.shape[2:] != psm.shape[2:]:
            raise ValueError("Classification/anchor regression shapes disagree")

    pairwise = cav_content['pairwise_t_matrix']
    if (pairwise.ndim != 5 or pairwise.shape[0] != batch_size
            or pairwise.shape[1] != pairwise.shape[2]
            or pairwise.shape[1] < max(counts) or pairwise.shape[-2:] != (4, 4)):
        raise ValueError("pairwise_t_matrix must be [B,L,L,4,4]")
    anchors = cav_content['anchor_box']
    if anchors.ndim not in (4, 5) or (anchors.ndim == 5 and anchors.shape[0] != batch_size):
        raise ValueError("anchor_box must be shared [H,W,A,7] or [B,H,W,A,7]")
    threshold = float(postprocessor.params['target_args']['score_threshold'])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Detector score_threshold must be in [0,1]")
    scenes, offset = [], 0
    for batch_index, count in enumerate(counts):
        anchor = anchors[batch_index] if anchors.ndim == 5 else anchors
        maps = predictions['psm_single']
        if tuple(anchor.shape) != (*maps.shape[2:], maps.shape[1], 7):
            raise ValueError("Anchor layout must match the local prediction maps")
        identity = torch.eye(4, device=maps.device, dtype=maps.dtype)
        boxes_list, scores_list, source_list = [], [], []
        source_counts = {}
        sources = ([-1] if mode == 'psa_hybrid' else []) + list(range(count))
        for source in sources:
            if source == -1:
                psm = predictions['psm'][batch_index:batch_index + 1]
                rm = predictions['rm'][batch_index:batch_index + 1]
                transform = identity  # Fused is ALREADY in ego coordinates.
                if psm.shape[1:] != maps.shape[1:]:
                    raise ValueError("Fused and local anchor grids differ")
            else:
                psm = maps[offset + source:offset + source + 1]
                rm = predictions['rm_single'][offset + source:offset + source + 1]
                # Dataset get_pairwise_transformation builds solve(T_j, T_i).
                transform = pairwise[batch_index, source, 0].to(rm)
            _validate_transform(transform)
            if source == 0 and not torch.allclose(transform, identity, atol=1e-4, rtol=0):
                raise ValueError("The local ego-to-ego transform must be identity")
            boxes, scores = postprocessor._decode_candidates(
                psm.masked_fill(~torch.isfinite(psm), float('nan')),
                rm, anchor, transform, score_threshold=threshold,
                iou_quality_cfg={'enabled': False},
            )[0]
            # All sources use the SAME raw sigmoid score and detector threshold.
            # Never consume the output of confidence-aware Late Fusion here.
            finite = torch.isfinite(boxes).all(dim=2).all(dim=1) & torch.isfinite(scores)
            boxes, scores = boxes[finite], scores[finite]
            source_counts['fused' if source == -1 else 'agent_{}'.format(source)] = len(scores)
            boxes_list.append(boxes)
            scores_list.append(scores)
            source_list.append(torch.full((len(scores),), source,
                                           device=scores.device, dtype=torch.long))
        scenes.append({'boxes': torch.cat(boxes_list),
                       'scores': torch.cat(scores_list),
                       'source_ids': torch.cat(source_list),
                       'source_counts': source_counts})
        offset += count
    return scenes


def aggregate_psa_candidates(postprocessor, candidates):
    """Aggregate one scene and return corners/scores plus inference diagnostics."""
    boxes, scores = candidates['boxes'], candidates['scores']
    keep = (box_utils.remove_large_pred_bbx(boxes)
            & box_utils.remove_bbx_abnormal_z(boxes))
    boxes, scores = boxes[keep], scores[keep]
    if boxes.is_cuda:
        torch.cuda.synchronize(boxes.device)
    start = time.perf_counter()
    selected, details = promote_suppress_aggregation(
        boxes.detach().cpu().numpy(), scores.detach().cpu().numpy(), return_details=True,
    )
    # Engineering adaptation: overlapping official groups can select the SAME
    # input index twice. Emit it once; do NOT merge distinct overlapping boxes.
    unique = list(dict.fromkeys(selected.tolist()))
    indices = torch.as_tensor(unique, dtype=torch.long, device=boxes.device)
    final_boxes, final_scores = boxes[indices], scores[indices]
    if boxes.is_cuda:
        torch.cuda.synchronize(boxes.device)
    elapsed = time.perf_counter() - start
    range_mask = box_utils.get_mask_for_boxes_within_range_torch(
        final_boxes, postprocessor.params['gt_range'])
    final_boxes, final_scores = final_boxes[range_mask], final_scores[range_mask]
    stats = {'mode': 'official_psa', 'score_definition': 'raw_sigmoid',
             'source_counts': candidates['source_counts'],
             'candidate_count': len(candidates['scores']),
             'geometry_filtered_count': len(scores),
             'valid_candidate_count': len(details['valid_indices']),
             'cluster_count': len(details['clusters']),
             'duplicate_selected_indices': len(selected) - len(unique),
             'retained_count': len(final_scores), 'psa_seconds': elapsed,
             'nms': False, 'rbv': False, 'quality_rectification': False}
    if len(final_scores) == 0:
        return None, None, stats
    return final_boxes, final_scores, stats


def post_process_psa(postprocessor, data_dict, output_dict, mode):
    if postprocessor.train:
        raise ValueError("PSA is inference-only; use baseline for training")
    if set(data_dict) != {'ego'} or set(output_dict) != {'ego'}:
        raise ValueError("PSA expects intermediate fusion with one top-level ego entry")
    if data_dict['ego']['record_len'].numel() != 1:
        raise ValueError("The dataset post_process API requires batch size 1; "
                         "use decode_psa_candidates per scene for batched inputs")
    candidates = decode_psa_candidates(postprocessor, data_dict['ego'],
                                       output_dict['ego'], mode)[0]
    boxes, scores, stats = aggregate_psa_candidates(postprocessor, candidates)
    stats['input_mode'] = mode
    postprocessor.psa_last_stats = stats
    return boxes, scores
