# -*- coding: utf-8 -*-
"""Compare baseline, Independent Oracle Top-K, and confidence-aware late fusion.

This is an analysis/visualization script.  It deliberately reuses the model,
Oracle, and post-processing implementations used by the existing evaluators;
it does not compute or report dataset AP.
"""

import argparse
import copy
import json
import os
import random
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch

ORACLE_PATCH_SIZE = (2, 3)
RECOVERY_IOU = 0.5
HIGH_QUALITY_IOU = 0.7
LOCALIZATION_GAIN = 0.1

METHOD_STYLES = {
    "baseline": {"color": "#d62728", "linestyle": ":"},
    "oracle": {"color": "#1f77b4", "linestyle": "--"},
    "late": {"color": "#ff7f0e", "linestyle": "-."},
}
GT_STYLE = {"color": "#2ca02c", "linestyle": "-"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Visualize fixed DAIR-V2X validation frames for E3DSNN + "
            "Where2comm baseline, Independent Oracle Top-K, and "
            "confidence-aware late fusion."
        )
    )
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--eval_epoch", default=None, type=str)
    parser.add_argument("--num_samples", default=8, type=int)
    parser.add_argument("--seed", default=20260919, type=int)
    parser.add_argument("--oracle_topk", default=80, type=int)
    parser.add_argument("--late_threshold", default=0.35, type=float)
    parser.add_argument("--late_beta", default=0.60, type=float)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
        "--frame_indices",
        nargs="+",
        type=int,
        default=None,
        help="Validation dataset indices. When supplied, random sampling is disabled.",
    )
    return parser.parse_args(argv)


def validate_options(opt):
    if opt.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if opt.oracle_topk <= 0:
        raise ValueError("--oracle_topk must be positive")
    if not 0.0 <= opt.late_threshold <= 1.0:
        raise ValueError("--late_threshold must be in [0, 1]")
    if not 0.0 < opt.late_beta < 1.0:
        raise ValueError("--late_beta must be in (0, 1)")
    if opt.frame_indices is not None:
        if len(opt.frame_indices) != len(set(opt.frame_indices)):
            raise ValueError("--frame_indices must not contain duplicates")


def select_frame_indices(dataset_size, num_samples, seed, frame_indices=None):
    if dataset_size <= 0:
        raise ValueError("The validation dataset is empty")
    if frame_indices is not None:
        invalid = [idx for idx in frame_indices if idx < 0 or idx >= dataset_size]
        if invalid:
            raise IndexError(
                "Validation frame indices out of range [0, {}): {}".format(
                    dataset_size, invalid
                )
            )
        return list(frame_indices), "explicit"
    if num_samples > dataset_size:
        raise ValueError(
            "--num_samples ({}) exceeds validation dataset size ({})".format(
                num_samples, dataset_size
            )
        )
    return random.Random(seed).sample(range(dataset_size), num_samples), "random"


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)


def tensor_to_numpy(value, empty_shape):
    if value is None:
        return np.empty(empty_shape, dtype=np.float32)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def normalize_detections(boxes, scores):
    boxes_np = tensor_to_numpy(boxes, (0, 8, 3))
    scores_np = tensor_to_numpy(scores, (0,)).reshape(-1)
    if boxes_np.ndim != 3 or boxes_np.shape[1:] != (8, 3):
        raise ValueError("Expected boxes with shape [N, 8, 3], got {}".format(boxes_np.shape))
    if boxes_np.shape[0] != scores_np.shape[0]:
        raise ValueError("Box and score counts differ")
    return boxes_np, scores_np


def normalize_gt(boxes):
    boxes_np = tensor_to_numpy(boxes, (0, 8, 3))
    if boxes_np.ndim != 3 or boxes_np.shape[1:] != (8, 3):
        raise ValueError("Expected GT boxes with shape [N, 8, 3], got {}".format(boxes_np.shape))
    return boxes_np


def best_rotated_iou_per_gt(gt_boxes, pred_boxes):
    """Return each GT's best rotated BEV IoU using project polygon helpers."""
    from opencood.utils import common_utils

    gt_boxes = normalize_gt(gt_boxes)
    pred_boxes = tensor_to_numpy(pred_boxes, (0, 8, 3))
    best_ious = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    best_indices = np.full(gt_boxes.shape[0], -1, dtype=np.int64)
    if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return best_ious, best_indices

    gt_polygons = list(common_utils.convert_format(gt_boxes))
    pred_polygons = list(common_utils.convert_format(pred_boxes))
    for gt_idx, gt_polygon in enumerate(gt_polygons):
        ious = common_utils.compute_iou(gt_polygon, pred_polygons)
        if ious.size:
            best_idx = int(np.argmax(ious))
            best_ious[gt_idx] = float(ious[best_idx])
            best_indices[gt_idx] = best_idx
    return best_ious, best_indices


def analyze_changes(frame_index, gt_boxes, baseline_boxes, oracle_boxes, late_boxes):
    baseline_iou, baseline_idx = best_rotated_iou_per_gt(gt_boxes, baseline_boxes)
    oracle_iou, oracle_idx = best_rotated_iou_per_gt(gt_boxes, oracle_boxes)
    late_iou, late_idx = best_rotated_iou_per_gt(gt_boxes, late_boxes)

    objects = []
    counts = {
        "baseline_to_oracle_recovered": 0,
        "baseline_to_late_recovered": 0,
        "oracle_localization_improved": 0,
        "late_localization_improved": 0,
        "both_improved": 0,
        "oracle_only_improved": 0,
        "late_only_improved": 0,
    }
    for gt_idx in range(len(baseline_iou)):
        base = float(baseline_iou[gt_idx])
        oracle = float(oracle_iou[gt_idx])
        late = float(late_iou[gt_idx])
        oracle_recovered = base < RECOVERY_IOU <= oracle
        late_recovered = base < RECOVERY_IOU <= late
        oracle_improved = oracle - base >= LOCALIZATION_GAIN
        late_improved = late - base >= LOCALIZATION_GAIN

        labels = []
        if oracle_recovered:
            labels.append("oracle_recovered")
            counts["baseline_to_oracle_recovered"] += 1
        if late_recovered:
            labels.append("late_recovered")
            counts["baseline_to_late_recovered"] += 1
        if oracle_recovered and late_recovered:
            labels.append("both_recovered")
        if oracle_improved:
            labels.append("oracle_improved_localization")
            counts["oracle_localization_improved"] += 1
        if late_improved:
            labels.append("late_improved_localization")
            counts["late_localization_improved"] += 1
        if oracle_improved and late_improved:
            counts["both_improved"] += 1
        elif oracle_improved:
            counts["oracle_only_improved"] += 1
        elif late_improved:
            counts["late_only_improved"] += 1
        if not labels:
            labels.append("no_change")

        objects.append(
            {
                "gt_index": gt_idx,
                "baseline_best_iou": base,
                "baseline_best_pred_index": int(baseline_idx[gt_idx]),
                "oracle_best_iou": oracle,
                "oracle_best_pred_index": int(oracle_idx[gt_idx]),
                "late_best_iou": late,
                "late_best_pred_index": int(late_idx[gt_idx]),
                "baseline_matched_iou_0_5": base >= RECOVERY_IOU,
                "oracle_matched_iou_0_5": oracle >= RECOVERY_IOU,
                "late_matched_iou_0_5": late >= RECOVERY_IOU,
                "baseline_matched_iou_0_7": base >= HIGH_QUALITY_IOU,
                "oracle_matched_iou_0_7": oracle >= HIGH_QUALITY_IOU,
                "late_matched_iou_0_7": late >= HIGH_QUALITY_IOU,
                "categories": labels,
            }
        )
    return {
        "frame_index": int(frame_index),
        "thresholds": {
            "recovery_iou": RECOVERY_IOU,
            "high_quality_iou": HIGH_QUALITY_IOU,
            "localization_gain": LOCALIZATION_GAIN,
        },
        "objects": objects,
        "counts": counts,
    }


def _plot_point_cloud(axis, point_cloud):
    if point_cloud.size:
        axis.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            s=0.15,
            c="#808080",
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )


def _plot_boxes(axis, boxes, style, label=None, scores=None, annotate_scores=False):
    for box_idx, box in enumerate(boxes):
        corners = box[:4, :2]
        closed = np.vstack((corners, corners[0]))
        axis.plot(
            closed[:, 0],
            closed[:, 1],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.25,
            alpha=0.9,
            label=label if box_idx == 0 else None,
        )
        if annotate_scores and scores is not None:
            axis.text(
                corners[0, 0],
                corners[0, 1],
                "{:.2f}".format(float(scores[box_idx])),
                color=style["color"],
                fontsize=5,
                alpha=0.9,
            )


def _setup_bev_axis(axis, point_cloud, pc_range, title):
    _plot_point_cloud(axis, point_cloud)
    axis.set_xlim(float(pc_range[0]), float(pc_range[3]))
    axis.set_ylim(float(pc_range[1]), float(pc_range[4]))
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("x (m)", fontsize=7)
    axis.set_ylabel("y (m)", fontsize=7)
    axis.tick_params(labelsize=6)
    axis.grid(False)


def save_four_panel(
    path,
    frame_index,
    point_cloud,
    pc_range,
    gt_boxes,
    baseline_boxes,
    baseline_scores,
    oracle_boxes,
    oracle_scores,
    late_boxes,
    late_scores,
    oracle_topk,
    late_threshold,
    late_beta,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], "Frame {} | GT only".format(frame_index), None, None, None),
        (axes[0, 1], "Baseline + GT", baseline_boxes, baseline_scores, "baseline"),
        (
            axes[1, 0],
            "Oracle K{} + GT".format(oracle_topk),
            oracle_boxes,
            oracle_scores,
            "oracle",
        ),
        (
            axes[1, 1],
            "Late T={:.2f} Beta={:.2f} + GT".format(late_threshold, late_beta),
            late_boxes,
            late_scores,
            "late",
        ),
    ]
    for axis, title, pred_boxes, pred_scores, method_name in panels:
        _setup_bev_axis(axis, point_cloud, pc_range, title)
        _plot_boxes(axis, gt_boxes, GT_STYLE, label="GT")
        if method_name is not None:
            _plot_boxes(
                axis,
                pred_boxes,
                METHOD_STYLES[method_name],
                label=method_name.capitalize(),
                scores=pred_scores,
                annotate_scores=True,
            )
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, loc="upper right", fontsize=6)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_overlay(
    path,
    frame_index,
    point_cloud,
    pc_range,
    gt_boxes,
    baseline_boxes,
    oracle_boxes,
    late_boxes,
):
    fig, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    _setup_bev_axis(axis, point_cloud, pc_range, "Frame {} | Detection overlay".format(frame_index))
    _plot_boxes(axis, gt_boxes, GT_STYLE, label="GT")
    _plot_boxes(axis, baseline_boxes, METHOD_STYLES["baseline"], label="Baseline")
    _plot_boxes(axis, oracle_boxes, METHOD_STYLES["oracle"], label="Oracle")
    _plot_boxes(axis, late_boxes, METHOD_STYLES["late"], label="Late")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="upper right", fontsize=8)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


@contextmanager
def confidence_aware_postprocess(postprocessor, enabled, threshold=None, beta=None):
    """Temporarily change only the existing confidence-aware postprocessor config."""
    had_key = "confidence_aware_late_fusion" in postprocessor.params
    original = copy.deepcopy(postprocessor.params.get("confidence_aware_late_fusion"))
    config = copy.deepcopy(original) if original is not None else {}
    config["enabled"] = bool(enabled)
    if threshold is not None:
        config["score_threshold"] = float(threshold)
    if beta is not None:
        config["collaborator_score_weight"] = float(beta)
    postprocessor.params["confidence_aware_late_fusion"] = config
    try:
        yield
    finally:
        if had_key:
            postprocessor.params["confidence_aware_late_fusion"] = original
        else:
            postprocessor.params.pop("confidence_aware_late_fusion", None)


def normal_postprocess(dataset, batch_data, output):
    with confidence_aware_postprocess(dataset.post_processor, enabled=False):
        return dataset.post_process(batch_data, {"ego": output})


def late_postprocess(dataset, batch_data, output, threshold, beta):
    with confidence_aware_postprocess(
        dataset.post_processor,
        enabled=True,
        threshold=threshold,
        beta=beta,
    ):
        return dataset.post_process(batch_data, {"ego": output})


def run_independent_oracle(
    model, cached, base_mask, criterion, label_dict, oracle_topk
):
    """Exact Independent Oracle utility/selection path from eval_patch_oracle.py."""
    from opencood.models.comm_modules.patch_utils import (
        apply_patch_state,
        patch_confidence_stats,
        topk_patch_ids,
    )
    from opencood.tools.eval_patch_oracle import (
        build_confidence_map,
        clone_with_selected_patches,
        scalar_loss,
    )

    target_size = base_mask.shape[-2:]
    confidence = build_confidence_map(cached["psm_single"], target_size)
    if confidence.shape[0] <= 1:
        raise RuntimeError("Independent Oracle requires at least one collaborator")
    stats = patch_confidence_stats(
        confidence[1], patch_size=ORACLE_PATCH_SIZE
    )
    utilities = []
    for patch_id in range(stats["mean"].numel()):
        mask_off = apply_patch_state(
            base_mask,
            patch_id,
            0.0,
            patch_size=ORACLE_PATCH_SIZE,
            agent_index=1,
        )
        mask_on = apply_patch_state(
            base_mask,
            patch_id,
            1.0,
            patch_size=ORACLE_PATCH_SIZE,
            agent_index=1,
        )
        output_off = model._forward_from_cached_context(
            cached, override_communication_mask=mask_off
        )
        output_on = model._forward_from_cached_context(
            cached, override_communication_mask=mask_on
        )
        utilities.append(scalar_loss(criterion, output_off, label_dict) - scalar_loss(criterion, output_on, label_dict))

    utilities = torch.stack(utilities)
    oracle_ids = topk_patch_ids(
        utilities, oracle_topk, threshold=0.0, largest=True
    )
    oracle_mask = clone_with_selected_patches(
        base_mask, oracle_ids, ORACLE_PATCH_SIZE
    )
    oracle_output = model._forward_from_cached_context(
        cached, override_communication_mask=oracle_mask
    )
    return oracle_output, oracle_ids, utilities


def add_counts(total_counts, frame_counts):
    for key in total_counts:
        total_counts[key] += int(frame_counts[key])


def process_frame(
    frame_index,
    dataset,
    model,
    criterion,
    device,
    pc_range,
    output_dir,
    oracle_topk,
    late_threshold,
    late_beta,
):
    from opencood.tools import train_utils

    raw_sample = dataset[frame_index]
    if raw_sample is None:
        raise RuntimeError("Dataset returned None for frame {}".format(frame_index))
    batch_data = dataset.collate_batch_test([raw_sample])
    if batch_data is None:
        raise RuntimeError("Dataset collate returned None for frame {}".format(frame_index))
    batch_data = train_utils.to_device(batch_data, device)
    ego_data = batch_data["ego"]

    cached = model._extract_cached_context(ego_data)
    if int(cached["record_len"].reshape(-1)[0].item()) <= 1:
        raise RuntimeError("Frame {} has no collaborator".format(frame_index))

    baseline_output = model._forward_from_cached_context(
        cached, return_communication_debug=True
    )
    base_mask = baseline_output.get("communication_masks")
    if base_mask is None:
        raise RuntimeError("Baseline output does not contain communication_masks")

    baseline_boxes_t, baseline_scores_t, gt_boxes_t = normal_postprocess(
        dataset, batch_data, baseline_output
    )
    oracle_output, oracle_ids, utilities = run_independent_oracle(
        model,
        cached,
        base_mask,
        criterion,
        ego_data["label_dict"],
        oracle_topk,
    )
    oracle_boxes_t, oracle_scores_t, oracle_gt_t = normal_postprocess(
        dataset, batch_data, oracle_output
    )
    late_boxes_t, late_scores_t, late_gt_t = late_postprocess(
        dataset,
        batch_data,
        baseline_output,
        late_threshold,
        late_beta,
    )

    gt_boxes = normalize_gt(gt_boxes_t)
    if not np.array_equal(gt_boxes, normalize_gt(oracle_gt_t)) or not np.array_equal(
        gt_boxes, normalize_gt(late_gt_t)
    ):
        raise RuntimeError("Post-processing branches returned inconsistent GT boxes")
    baseline_boxes, baseline_scores = normalize_detections(
        baseline_boxes_t, baseline_scores_t
    )
    oracle_boxes, oracle_scores = normalize_detections(oracle_boxes_t, oracle_scores_t)
    late_boxes, late_scores = normalize_detections(late_boxes_t, late_scores_t)
    point_cloud = tensor_to_numpy(ego_data["origin_lidar"][0], (0, 4))

    frame_dir = os.path.join(output_dir, "frame_{:06d}".format(frame_index))
    os.makedirs(frame_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(frame_dir, "boxes.npz"),
        gt_boxes=gt_boxes,
        baseline_boxes=baseline_boxes,
        baseline_scores=baseline_scores,
        oracle_boxes=oracle_boxes,
        oracle_scores=oracle_scores,
        late_boxes=late_boxes,
        late_scores=late_scores,
    )

    changes = analyze_changes(
        frame_index,
        gt_boxes,
        baseline_boxes,
        oracle_boxes,
        late_boxes,
    )
    changes.update(
        {
            "oracle_topk": int(oracle_topk),
            "oracle_selected_patch_count": len(oracle_ids),
            "oracle_patch_ids": [int(value) for value in oracle_ids],
            "oracle_positive_utilities": int((utilities > 0).sum().item()),
            "late_threshold": float(late_threshold),
            "late_beta": float(late_beta),
        }
    )
    write_json(os.path.join(frame_dir, "changes.json"), changes)
    save_four_panel(
        os.path.join(frame_dir, "compare_4panel.png"),
        frame_index,
        point_cloud,
        pc_range,
        gt_boxes,
        baseline_boxes,
        baseline_scores,
        oracle_boxes,
        oracle_scores,
        late_boxes,
        late_scores,
        oracle_topk,
        late_threshold,
        late_beta,
    )
    save_overlay(
        os.path.join(frame_dir, "overlay.png"),
        frame_index,
        point_cloud,
        pc_range,
        gt_boxes,
        baseline_boxes,
        oracle_boxes,
        late_boxes,
    )
    return changes


def main(argv=None):
    opt = parse_args(argv)
    validate_options(opt)

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils

    os.makedirs(opt.output_dir, exist_ok=True)

    hypes = yaml_utils.load_yaml(None, opt)
    hypes["validate_dir"] = hypes["test_dir"]
    hypes.setdefault("postprocess", {}).setdefault(
        "confidence_aware_late_fusion", {}
    )["enabled"] = False

    print("Building validation dataset")
    dataset = build_dataset(hypes, visualize=True, train=False)
    selected_indices, selection_mode = select_frame_indices(
        len(dataset), opt.num_samples, opt.seed, opt.frame_indices
    )
    write_json(
        os.path.join(opt.output_dir, "selected_frames.json"),
        {
            "seed": int(opt.seed),
            "indices": selected_indices,
            "selection_mode": selection_mode,
            "dataset_size": len(dataset),
        },
    )

    print("Creating model and loading checkpoint")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epoch_id, model = train_utils.load_saved_model(
        opt.model_dir, model, opt.eval_epoch
    )
    model.eval()
    criterion = train_utils.create_loss(hypes)
    if not hasattr(model, "_extract_cached_context") or not hasattr(
        model, "_forward_from_cached_context"
    ):
        raise RuntimeError("This script requires a Where2comm model with cache APIs")

    total_counts = {
        "baseline_to_oracle_recovered": 0,
        "baseline_to_late_recovered": 0,
        "oracle_localization_improved": 0,
        "late_localization_improved": 0,
        "both_improved": 0,
        "oracle_only_improved": 0,
        "late_only_improved": 0,
    }
    completed_indices = []
    frame_summaries = []
    pc_range = hypes["postprocess"]["gt_range"]
    with torch.no_grad():
        for position, frame_index in enumerate(selected_indices, start=1):
            print(
                "[{}/{}] Processing validation frame {}".format(
                    position, len(selected_indices), frame_index
                )
            )
            changes = process_frame(
                frame_index,
                dataset,
                model,
                criterion,
                device,
                pc_range,
                opt.output_dir,
                opt.oracle_topk,
                opt.late_threshold,
                opt.late_beta,
            )
            completed_indices.append(frame_index)
            add_counts(total_counts, changes["counts"])
            frame_summaries.append(
                {
                    "frame_index": frame_index,
                    "ground_truth_objects": len(changes["objects"]),
                    "counts": changes["counts"],
                }
            )

    summary = {
        "analysis_only": True,
        "not_an_ap_evaluation": True,
        "epoch": str(epoch_id),
        "seed": int(opt.seed),
        "selected_indices": selected_indices,
        "completed_indices": completed_indices,
        "num_frames": len(completed_indices),
        "ground_truth_objects": sum(
            item["ground_truth_objects"] for item in frame_summaries
        ),
        "oracle_topk": int(opt.oracle_topk),
        "oracle_patch_size": list(ORACLE_PATCH_SIZE),
        "late_threshold": float(opt.late_threshold),
        "late_beta": float(opt.late_beta),
        "thresholds": {
            "recovery_iou": RECOVERY_IOU,
            "high_quality_iou": HIGH_QUALITY_IOU,
            "localization_gain": LOCALIZATION_GAIN,
        },
        "counts": total_counts,
        "frames": frame_summaries,
    }
    write_json(os.path.join(opt.output_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
