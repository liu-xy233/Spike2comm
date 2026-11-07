# -*- coding: utf-8 -*-
"""Diagnose how IoU-aware scores change rotated-NMS decisions."""

import argparse
import csv
import json
import logging
import math
import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils import box_utils, common_utils


GT_BACKGROUND_IOU = 0.1
OUTCOME_EPSILON = 1e-3
CSV_FIELDS = [
    "frame_index",
    "raw_candidate_index",
    "quality_candidate_index",
    "gt_index",
    "competition_bev_iou",
    "raw_cls_score",
    "quality_cls_score",
    "raw_predicted_q",
    "quality_predicted_q",
    "raw_nms_score",
    "quality_nms_score",
    "raw_actual_gt_bev_iou",
    "quality_actual_gt_bev_iou",
    "actual_gt_bev_iou_delta",
    "outcome",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare raw-score and IoU-quality rotated NMS"
    )
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--eval_epoch", default="26", type=str)
    parser.add_argument("--max_samples", default=300, type=int)
    parser.add_argument("--quality_power", default=1.0, type=float)
    parser.add_argument("--output_dir", default=None, type=str)
    return parser.parse_args()


def setup_logger(output_dir):
    logger = logging.getLogger("iou_nms_diagnostic")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "diagnostic.log"), mode="w"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def best_gt_bev_matches(boxes, gt_boxes):
    """Return each candidate's best GT index and GLOBAL-SORT BEV IoU."""
    count = int(boxes.shape[0])
    best_iou = np.zeros(count, dtype=np.float32)
    best_gt = np.full(count, -1, dtype=np.int64)
    if count == 0 or gt_boxes is None or gt_boxes.shape[0] == 0:
        return best_iou, best_gt

    box_np = common_utils.torch_tensor_to_numpy(boxes)
    gt_np = common_utils.torch_tensor_to_numpy(gt_boxes)
    box_polygons = list(common_utils.convert_format(box_np))
    gt_polygons = list(common_utils.convert_format(gt_np))
    for index, polygon in enumerate(box_polygons):
        ious = common_utils.compute_iou(polygon, gt_polygons)
        if ious.size:
            best_gt[index] = int(np.argmax(ious))
            best_iou[index] = float(ious[best_gt[index]])
    return best_iou, best_gt


def changed_box_bev_iou(raw_boxes, quality_boxes):
    """Compute only the raw-only x quality-only competition IoUs."""
    rows = int(raw_boxes.shape[0])
    cols = int(quality_boxes.shape[0])
    matrix = np.zeros((rows, cols), dtype=np.float32)
    if rows == 0 or cols == 0:
        return matrix
    raw_polygons = list(common_utils.convert_format(
        common_utils.torch_tensor_to_numpy(raw_boxes)
    ))
    quality_polygons = list(common_utils.convert_format(
        common_utils.torch_tensor_to_numpy(quality_boxes)
    ))
    for row, polygon in enumerate(raw_polygons):
        matrix[row] = common_utils.compute_iou(polygon, quality_polygons)
    return matrix


def run_nms(candidate_set, score_key, nms_threshold, gt_range):
    """Run NMS and the existing post-NMS range filter, retaining row ids."""
    boxes = candidate_set["boxes"]
    scores = candidate_set[score_key]
    keep = np.asarray(
        box_utils.nms_rotated(boxes, scores, nms_threshold),
        dtype=np.int64,
    )
    if keep.size == 0:
        return keep
    range_keep = box_utils.get_mask_for_boxes_within_range_torch(
        boxes[keep], gt_range
    )
    return keep[range_keep.detach().cpu().numpy().astype(bool)]


def _empty_frame_result():
    return {
        "candidate_count": 0,
        "raw_keep_count": 0,
        "quality_keep_count": 0,
        "changed": False,
        "raw_only_count": 0,
        "quality_only_count": 0,
        "reliable_rows": [],
        "background_exclusive_count": 0,
        "unpaired_exclusive_count": 0,
        "unresolved_nonbackground_exclusive_count": 0,
        "ambiguous_competition_candidates": 0,
        "actual_gt_ious": np.empty(0, dtype=np.float32),
        "predicted_qualities": np.empty(0, dtype=np.float32),
        "no_gt": False,
    }


def analyze_frame(frame_index, candidate_set, gt_boxes, nms_threshold,
                  gt_range, background_iou=GT_BACKGROUND_IOU,
                  outcome_epsilon=OUTCOME_EPSILON):
    """Compare both NMS branches and conservatively pair replacements."""
    if candidate_set is None or candidate_set["boxes"].shape[0] == 0:
        result = _empty_frame_result()
        result["no_gt"] = gt_boxes is None or gt_boxes.shape[0] == 0
        return result

    result = _empty_frame_result()
    boxes = candidate_set["boxes"]
    candidate_count = int(boxes.shape[0])
    aligned_keys = (
        "raw_scores",
        "quality_scores",
        "predicted_quality",
        "candidate_indices",
    )
    if any(candidate_set[key].shape[0] != candidate_count
           for key in aligned_keys):
        raise ValueError("Diagnostic candidate tensors are not index-aligned")
    ids = candidate_set["candidate_indices"].detach().cpu().numpy()
    if np.unique(ids).size != ids.size:
        raise ValueError("Diagnostic candidate indices must be unique")
    raw_keep = run_nms(candidate_set, "raw_scores", nms_threshold, gt_range)
    quality_keep = run_nms(
        candidate_set, "quality_scores", nms_threshold, gt_range
    )
    raw_ids = ids[raw_keep]
    quality_ids = ids[quality_keep]
    raw_set = set(raw_ids.tolist())
    quality_set = set(quality_ids.tolist())
    raw_only_ids = sorted(raw_set.difference(quality_set))
    quality_only_ids = sorted(quality_set.difference(raw_set))

    actual_iou, best_gt = best_gt_bev_matches(boxes, gt_boxes)
    predicted_q = candidate_set["predicted_quality"].detach().cpu().numpy()
    result.update({
        "candidate_count": candidate_count,
        "raw_keep_count": int(raw_keep.size),
        "quality_keep_count": int(quality_keep.size),
        "changed": raw_set != quality_set,
        "raw_only_count": len(raw_only_ids),
        "quality_only_count": len(quality_only_ids),
        "actual_gt_ious": actual_iou,
        "predicted_qualities": predicted_q,
        "no_gt": gt_boxes is None or gt_boxes.shape[0] == 0,
    })
    if not result["changed"]:
        return result

    id_to_row = {int(candidate_id): row for row, candidate_id in enumerate(ids)}
    raw_rows = [id_to_row[index] for index in raw_only_ids]
    quality_rows = [id_to_row[index] for index in quality_only_ids]
    competition_iou = changed_box_bev_iou(
        boxes[raw_rows], boxes[quality_rows]
    )

    reliable_edges = []
    degrees = {}
    for raw_local, raw_row in enumerate(raw_rows):
        for quality_local, quality_row in enumerate(quality_rows):
            overlap = float(competition_iou[raw_local, quality_local])
            if overlap <= nms_threshold:
                continue
            same_gt = (
                best_gt[raw_row] >= 0
                and best_gt[raw_row] == best_gt[quality_row]
            )
            foreground = (
                actual_iou[raw_row] >= background_iou
                and actual_iou[quality_row] >= background_iou
            )
            if same_gt and foreground:
                reliable_edges.append((overlap, raw_row, quality_row))
                degrees[raw_row] = degrees.get(raw_row, 0) + 1
                degrees[quality_row] = degrees.get(quality_row, 0) + 1

    result["ambiguous_competition_candidates"] = sum(
        degree > 1 for degree in degrees.values()
    )
    used_raw = set()
    used_quality = set()
    raw_scores = candidate_set["raw_scores"].detach().cpu().numpy()
    quality_scores = candidate_set["quality_scores"].detach().cpu().numpy()
    for overlap, raw_row, quality_row in sorted(
            reliable_edges, key=lambda edge: edge[0], reverse=True):
        if raw_row in used_raw or quality_row in used_quality:
            continue
        used_raw.add(raw_row)
        used_quality.add(quality_row)
        delta = float(actual_iou[quality_row] - actual_iou[raw_row])
        if delta > outcome_epsilon:
            outcome = "positive"
        elif delta < -outcome_epsilon:
            outcome = "negative"
        else:
            outcome = "unchanged"
        result["reliable_rows"].append({
            "frame_index": int(frame_index),
            "raw_candidate_index": int(ids[raw_row]),
            "quality_candidate_index": int(ids[quality_row]),
            "gt_index": int(best_gt[raw_row]),
            "competition_bev_iou": overlap,
            "raw_cls_score": float(raw_scores[raw_row]),
            "quality_cls_score": float(raw_scores[quality_row]),
            "raw_predicted_q": float(predicted_q[raw_row]),
            "quality_predicted_q": float(predicted_q[quality_row]),
            "raw_nms_score": float(quality_scores[raw_row]),
            "quality_nms_score": float(quality_scores[quality_row]),
            "raw_actual_gt_bev_iou": float(actual_iou[raw_row]),
            "quality_actual_gt_bev_iou": float(actual_iou[quality_row]),
            "actual_gt_bev_iou_delta": delta,
            "outcome": outcome,
        })

    exclusive_rows = set(raw_rows + quality_rows)
    paired_rows = used_raw.union(used_quality)
    unpaired_rows = exclusive_rows.difference(paired_rows)
    result["unpaired_exclusive_count"] = len(unpaired_rows)
    result["background_exclusive_count"] = sum(
        best_gt[row] < 0 or actual_iou[row] < background_iou
        for row in unpaired_rows
    )
    result["unresolved_nonbackground_exclusive_count"] = (
        result["unpaired_exclusive_count"]
        - result["background_exclusive_count"]
    )
    return result


def new_totals():
    return {
        "frames_total": 0,
        "frames_empty_candidates": 0,
        "frames_without_gt": 0,
        "candidates_total": 0,
        "raw_nms_kept_total": 0,
        "quality_nms_kept_total": 0,
        "changed_frames": 0,
        "keep_count_changed_frames": 0,
        "keep_count_delta_total": 0,
        "keep_count_absolute_delta_total": 0,
        "raw_nms_exclusive_candidates": 0,
        "quality_nms_exclusive_candidates": 0,
        "reliable_replacement_events": 0,
        "positive_replacements": 0,
        "negative_replacements": 0,
        "unchanged_replacements": 0,
        "background_exclusive_candidates": 0,
        "unpaired_exclusive_candidates": 0,
        "unresolved_nonbackground_exclusive_candidates": 0,
        "ambiguous_competition_candidates": 0,
        "changed_frames_without_reliable_replacement": 0,
        "positive_deltas": [],
        "negative_deltas": [],
        "all_actual_gt_ious": [],
        "all_predicted_qualities": [],
        "ordering_correct": 0,
        "ordering_incorrect": 0,
        "ordering_ties": 0,
    }


def update_totals(totals, frame_result):
    totals["frames_total"] += 1
    totals["frames_empty_candidates"] += (
        frame_result["candidate_count"] == 0
    )
    totals["frames_without_gt"] += frame_result["no_gt"]
    totals["candidates_total"] += frame_result["candidate_count"]
    totals["raw_nms_kept_total"] += frame_result["raw_keep_count"]
    totals["quality_nms_kept_total"] += frame_result["quality_keep_count"]
    totals["raw_nms_exclusive_candidates"] += frame_result["raw_only_count"]
    totals["quality_nms_exclusive_candidates"] += frame_result[
        "quality_only_count"
    ]
    delta_count = (
        frame_result["quality_keep_count"] - frame_result["raw_keep_count"]
    )
    totals["keep_count_delta_total"] += delta_count
    totals["keep_count_absolute_delta_total"] += abs(delta_count)
    if delta_count != 0:
        totals["keep_count_changed_frames"] += 1
    if frame_result["changed"]:
        totals["changed_frames"] += 1
        if not frame_result["reliable_rows"]:
            totals["changed_frames_without_reliable_replacement"] += 1

    totals["background_exclusive_candidates"] += frame_result[
        "background_exclusive_count"
    ]
    totals["unpaired_exclusive_candidates"] += frame_result[
        "unpaired_exclusive_count"
    ]
    totals["unresolved_nonbackground_exclusive_candidates"] += frame_result[
        "unresolved_nonbackground_exclusive_count"
    ]
    totals["ambiguous_competition_candidates"] += frame_result[
        "ambiguous_competition_candidates"
    ]
    totals["all_actual_gt_ious"].extend(
        frame_result["actual_gt_ious"].tolist()
    )
    totals["all_predicted_qualities"].extend(
        frame_result["predicted_qualities"].tolist()
    )

    for row in frame_result["reliable_rows"]:
        totals["reliable_replacement_events"] += 1
        outcome = row["outcome"]
        totals[outcome + "_replacements"] += 1
        if outcome == "positive":
            totals["positive_deltas"].append(
                row["actual_gt_bev_iou_delta"]
            )
        elif outcome == "negative":
            totals["negative_deltas"].append(
                row["actual_gt_bev_iou_delta"]
            )

        actual_delta = row["actual_gt_bev_iou_delta"]
        quality_delta = row["quality_predicted_q"] - row["raw_predicted_q"]
        if (abs(actual_delta) <= OUTCOME_EPSILON
                or abs(quality_delta) <= OUTCOME_EPSILON):
            totals["ordering_ties"] += 1
        elif actual_delta * quality_delta > 0:
            totals["ordering_correct"] += 1
        else:
            totals["ordering_incorrect"] += 1


def _average_rank(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) * 0.5 + 1.0
        start = end
    return ranks


def spearman_correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(first) & np.isfinite(second)
    first = first[finite]
    second = second[finite]
    if first.size < 2:
        return None
    first_rank = _average_rank(first)
    second_rank = _average_rank(second)
    first_centered = first_rank - float(np.mean(first_rank))
    second_centered = second_rank - float(np.mean(second_rank))
    first_square_sum = float(np.sum(first_centered * first_centered))
    second_square_sum = float(np.sum(second_centered * second_centered))
    denominator = math.sqrt(first_square_sum * second_square_sum)
    if denominator == 0.0:
        return None
    numerator = float(np.sum(first_centered * second_centered))
    return numerator / denominator


def finalize_summary(totals, metadata):
    frames = totals["frames_total"]
    ordering_comparable = (
        totals["ordering_correct"] + totals["ordering_incorrect"]
    )
    all_actual_iou = np.asarray(
        totals["all_actual_gt_ious"], dtype=np.float64
    )
    all_predicted_q = np.asarray(
        totals["all_predicted_qualities"], dtype=np.float64
    )
    foreground = all_actual_iou >= GT_BACKGROUND_IOU
    summary = {
        "metadata": metadata,
        "iou_definition": {
            "primary": "rotated_bev_iou",
            "matches_global_sort_ap": True,
            "three_dimensional_iou_reported": False,
            "background_gt_iou_threshold": GT_BACKGROUND_IOU,
            "replacement_outcome_epsilon": OUTCOME_EPSILON,
        },
    }
    for key, value in totals.items():
        if not isinstance(value, list):
            summary[key] = int(value) if isinstance(
                value, (bool, np.bool_, np.integer)
            ) else value
    summary["changed_frame_ratio"] = (
        totals["changed_frames"] / float(frames) if frames else 0.0
    )
    summary["mean_positive_actual_gt_bev_iou_delta"] = (
        float(np.mean(totals["positive_deltas"]))
        if totals["positive_deltas"] else None
    )
    summary["mean_negative_actual_gt_bev_iou_delta"] = (
        float(np.mean(totals["negative_deltas"]))
        if totals["negative_deltas"] else None
    )
    summary["spearman_candidate_count_all"] = int(all_actual_iou.size)
    summary["spearman_candidate_count_foreground"] = int(
        np.count_nonzero(foreground)
    )
    summary["predicted_quality_actual_gt_bev_iou_spearman_all"] = (
        spearman_correlation(
            all_predicted_q,
            all_actual_iou,
        )
    )
    summary["predicted_quality_actual_gt_bev_iou_spearman_foreground"] = (
        spearman_correlation(
            all_predicted_q[foreground],
            all_actual_iou[foreground],
        )
    )
    summary["quality_ordering_comparable_pairs"] = ordering_comparable
    summary["quality_ordering_accuracy"] = (
        totals["ordering_correct"] / float(ordering_comparable)
        if ordering_comparable else None
    )
    return summary


def write_outputs(output_dir, summary, rows):
    with open(os.path.join(output_dir, "summary.json"), "w") as file_handle:
        json.dump(summary, file_handle, indent=2, sort_keys=True)
    with open(
        os.path.join(output_dir, "changed_cases.csv"),
        "w",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    opt = parse_args()
    if opt.quality_power < 0:
        raise ValueError("quality_power must be non-negative")
    if opt.max_samples == 0 or opt.max_samples < -1:
        raise ValueError("max_samples must be positive or -1 for all samples")

    output_dir = opt.output_dir or os.path.join(
        opt.model_dir,
        "nms_diagnostic_epoch{}_p{}".format(
            opt.eval_epoch, opt.quality_power
        ),
    )
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir)

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils

    hypes = yaml_utils.load_yaml(None, opt)
    hypes["validate_dir"] = hypes["test_dir"]
    quality_cfg = hypes.setdefault("iou_quality_head", {})
    if not quality_cfg.get("enabled", False):
        raise ValueError(
            "The model config does not mark iou_quality_head as trained; "
            "refusing to diagnose an unverified/random Quality Head"
        )
    quality_cfg.update({
        "enabled": True,
        "inference_mode": "nms_only",
        "quality_power": opt.quality_power,
    })
    hypes["postprocess"]["iou_quality_head"] = quality_cfg.copy()
    hypes["model"]["args"]["iou_quality_head"] = quality_cfg.copy()
    hypes["postprocess"].setdefault(
        "confidence_aware_late_fusion", {}
    )["enabled"] = False

    logger.info("Building DAIR-V2X validation dataset")
    dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    logger.info("Creating model and loading epoch %s", opt.eval_epoch)
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epoch_id, model = train_utils.load_saved_model(
        opt.model_dir, model, opt.eval_epoch
    )
    model.eval()

    sample_limit = len(dataset)
    if opt.max_samples > 0:
        sample_limit = min(sample_limit, opt.max_samples)
    nms_threshold = float(hypes["postprocess"]["nms_thresh"])
    gt_range = hypes["postprocess"]["gt_range"]
    logger.info(
        "epoch=%s device=%s samples=%d quality_power=%s nms_threshold=%s",
        epoch_id, device, sample_limit, opt.quality_power, nms_threshold,
    )
    logger.info("Confidence-aware Late Fusion disabled; RBV path unreachable")

    totals = new_totals()
    changed_rows = []
    processed = 0
    iterator = tqdm(enumerate(data_loader), total=sample_limit)
    for frame_index, batch_data in iterator:
        if frame_index >= sample_limit:
            break
        if batch_data is None:
            logger.warning("frame=%d collate returned None", frame_index)
            continue
        batch_data = train_utils.to_device(batch_data, device)
        with torch.no_grad():
            output_dict = OrderedDict()
            output_dict["ego"] = model(batch_data["ego"])
            candidate_set = dataset.post_processor.post_process(
                batch_data,
                output_dict,
                return_nms_candidates=True,
            )
            gt_boxes = dataset.post_processor.generate_gt_bbx(batch_data)

        frame_result = analyze_frame(
            frame_index,
            candidate_set,
            gt_boxes,
            nms_threshold,
            gt_range,
        )
        if opt.quality_power == 0.0 and frame_result["changed"]:
            raise AssertionError(
                "quality_power=0 must produce identical NMS candidate indices"
            )
        update_totals(totals, frame_result)
        changed_rows.extend(frame_result["reliable_rows"])
        processed += 1
        if processed % 50 == 0:
            logger.info(
                "processed=%d candidates=%d changed_frames=%d reliable=%d",
                processed,
                totals["candidates_total"],
                totals["changed_frames"],
                totals["reliable_replacement_events"],
            )

    metadata = {
        "model_dir": os.path.abspath(opt.model_dir),
        "eval_epoch": str(epoch_id),
        "requested_max_samples": opt.max_samples,
        "processed_samples": processed,
        "dataset_size": len(dataset),
        "quality_power": opt.quality_power,
        "nms_threshold": nms_threshold,
        "score_threshold": hypes["postprocess"]["target_args"][
            "score_threshold"
        ],
        "late_fusion_enabled": False,
        "rbv_executed": False,
        "model_forwards_per_frame": 1,
        "ap_computed": False,
    }
    summary = finalize_summary(totals, metadata)
    write_outputs(output_dir, summary, changed_rows)
    logger.info(
        "done frames=%d changed=%d (%.4f) reliable=%d positive=%d "
        "negative=%d unchanged=%d",
        summary["frames_total"],
        summary["changed_frames"],
        summary["changed_frame_ratio"],
        summary["reliable_replacement_events"],
        summary["positive_replacements"],
        summary["negative_replacements"],
        summary["unchanged_replacements"],
    )
    logger.info("wrote %s", output_dir)


if __name__ == "__main__":
    main()
