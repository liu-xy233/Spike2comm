# -*- coding: utf-8 -*-

"""
Four-way qualitative comparison:

1. Baseline
2. Late Fusion
3. Late Fusion + Post-NMS Rotated Box Voting
4. Oracle K80 + Late Fusion + Post-NMS Rotated Box Voting

This wrapper reuses the already validated:
    visualize_baseline_oracle_late.py

No model weights or training logic are modified.
"""

import inspect
import json
import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from opencood.tools import visualize_baseline_oracle_late as base


# ----------------------------------------------------------------------
# Fixed configurations for this qualitative comparison.
# ----------------------------------------------------------------------

LATE_ONLY_THRESHOLD = 0.35
LATE_ONLY_BETA = 0.60

RBV_IOU_THRESHOLD = 0.50
RBV_SCORE_POWER = 1.0
RBV_IOU_POWER = 1.0

ORACLE_LATE_THRESHOLD = 0.30
ORACLE_LATE_BETA = 0.55


# Save original process_frame before monkey-patching it.
_legacy_process_frame = base.process_frame


@contextmanager
def temporary_env(updates):
    """Temporarily modify environment variables."""
    old = {}

    for key, value in updates.items():
        old[key] = os.environ.get(key)

        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)

    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def postprocess_late_rbv(
    dataset,
    batch_data,
    output_dict,
    threshold,
    beta,
    rbv_enabled,
    rbv_iou=0.50,
):
    """Run existing Confidence-Aware Late Fusion with optional RBV."""

    if not hasattr(dataset, "post_processor"):
        raise AttributeError("dataset has no post_processor")

    params = dataset.post_processor.params
    key = "confidence_aware_late_fusion"

    old_exists = key in params
    old_cfg = params.get(key, None)

    cfg = dict(old_cfg) if isinstance(old_cfg, dict) else {}

    cfg["enabled"] = True
    cfg["score_threshold"] = float(threshold)
    cfg["collaborator_score_weight"] = float(beta)

    params[key] = cfg

    env = {
        "ENABLE_ROTATED_BOX_VOTING":
            "1" if rbv_enabled else "0",
        "RBV_IOU_THRESHOLD": str(float(rbv_iou)),
        "RBV_SCORE_POWER": str(RBV_SCORE_POWER),
        "RBV_IOU_POWER": str(RBV_IOU_POWER),
    }

    try:
        with temporary_env(env):
            pred_boxes, pred_scores, gt_boxes = dataset.post_process(
                batch_data,
                {"ego": output_dict},
            )
    finally:
        if old_exists:
            params[key] = old_cfg
        else:
            params.pop(key, None)

    return pred_boxes, pred_scores, gt_boxes


def tensor_to_numpy(x):
    if x is None:
        return None

    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def extract_point_cloud(batch_data):
    """Best-effort extraction of ego raw lidar for BEV background."""

    ego = batch_data["ego"]

    point_cloud = None

    for key in [
        "origin_lidar",
        "origin_lidar_np",
        "lidar_np",
    ]:
        if key in ego:
            point_cloud = ego[key]
            break

    if point_cloud is None:
        return None

    if isinstance(point_cloud, (list, tuple)):
        if len(point_cloud) == 0:
            return None
        point_cloud = point_cloud[0]

    point_cloud = tensor_to_numpy(point_cloud)

    if point_cloud is None:
        return None

    # Remove batch dimension if present.
    while point_cloud.ndim > 2 and point_cloud.shape[0] == 1:
        point_cloud = point_cloud[0]

    if point_cloud.ndim != 2 or point_cloud.shape[1] < 2:
        return None

    return point_cloud


def draw_boxes(ax, boxes, color, linewidth=1.4, linestyle="-"):
    """Draw rotated BEV boxes represented by [N, 8, 3] corners."""

    boxes = tensor_to_numpy(boxes)

    if boxes is None:
        return

    if boxes.size == 0:
        return

    if boxes.ndim != 3 or boxes.shape[1] < 4:
        return

    for box in boxes:
        xy = box[:4, :2]

        closed = np.concatenate(
            [xy, xy[:1]],
            axis=0,
        )

        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )


def collect_xy(point_cloud, gt_boxes, method_boxes):
    """
    Determine a shared visualization crop from GT + predictions only.

    IMPORTANT:
    The full LiDAR point cloud is intentionally NOT used to determine
    x/y limits, otherwise a few distant points make vehicle boxes tiny.
    """

    arrays = []

    # Ground-truth boxes.
    gt = tensor_to_numpy(gt_boxes)

    if gt is not None and gt.size > 0:
        arrays.append(
            gt[:, :4, :2].reshape(-1, 2)
        )

    # Predictions from all four methods.
    for boxes in method_boxes:
        arr = tensor_to_numpy(boxes)

        if arr is not None and arr.size > 0:
            arrays.append(
                arr[:, :4, :2].reshape(-1, 2)
            )

    # Fallback if no boxes exist.
    if not arrays:
        return (-60, 60), (-60, 60)

    xy = np.concatenate(arrays, axis=0)

    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]

    if xy.shape[0] == 0:
        return (-60, 60), (-60, 60)

    xmin = float(np.min(xy[:, 0]))
    xmax = float(np.max(xy[:, 0]))
    ymin = float(np.min(xy[:, 1]))
    ymax = float(np.max(xy[:, 1]))

    # Center of all GT/prediction boxes.
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    width = xmax - xmin
    height = ymax - ymin

    # Add ~15% margin, but never zoom in too aggressively.
    span = max(
        width * 1.30,
        height * 1.30,
        100.0,
    )

    half = 0.5 * span

    return (
        (cx - half, cx + half),
        (cy - half, cy + half),
    )


def plot_four_way(
    output_path,
    frame_index,
    point_cloud,
    gt_boxes,
    baseline_boxes,
    late_boxes,
    late_rbv_boxes,
    oracle_late_rbv_boxes,
):
    methods = [
        (
            "(a) Baseline",
            baseline_boxes,
        ),
        (
            "(b) Late Fusion\nT=0.35, beta=0.60",
            late_boxes,
        ),
        (
            "(c) Late Fusion + RBV\n"
            "T=0.35, beta=0.60, RBV=0.50",
            late_rbv_boxes,
        ),
        (
            "(d) Oracle K80 + Late Fusion + RBV\n"
            "T=0.30, beta=0.55, RBV=0.50",
            oracle_late_rbv_boxes,
        ),
    ]

    xlim, ylim = collect_xy(
        point_cloud,
        gt_boxes,
        [m[1] for m in methods],
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    for ax, (title, pred_boxes) in zip(
        axes.reshape(-1),
        methods,
    ):
        if point_cloud is not None:
            # Thin point cloud to keep the PNG readable.
            pc = point_cloud

            if pc.shape[0] > 80000:
                step = max(1, pc.shape[0] // 80000)
                pc = pc[::step]

            ax.scatter(
                pc[:, 0],
                pc[:, 1],
                s=0.22,
                c="0.82",
                alpha=0.30,
                rasterized=True,
            )

        # Ground truth.
        draw_boxes(
            ax,
            gt_boxes,
            color="green",
            linewidth=2.8,
            linestyle="--",
        )

        # Prediction.
        draw_boxes(
            ax,
            pred_boxes,
            color="red",
            linewidth=2.2,
            linestyle="-",
        )

        pred_np = tensor_to_numpy(pred_boxes)

        pred_count = (
            0 if pred_np is None
            else int(pred_np.shape[0])
        )

        ax.set_title(
            "{}\nPred={}".format(
                title,
                pred_count,
            ),
            fontsize=10,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(False)

    fig.suptitle(
        "Validation frame {}".format(frame_index),
        fontsize=13,
    )

    # Simple common legend.
    axes[0, 0].plot(
        [],
        [],
        color="green",
        linestyle="--",
        label="Ground Truth",
    )

    axes[0, 0].plot(
        [],
        [],
        color="red",
        linestyle="-",
        label="Prediction",
    )

    axes[0, 0].legend(
        loc="upper right",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    fig.savefig(
        str(output_path),
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def locate_frame_dir(output_dir, frame_index):
    """Find the directory produced by the legacy visualizer."""

    output_dir = Path(output_dir)

    tokens = {
        str(frame_index),
        "{:04d}".format(frame_index),
        "{:05d}".format(frame_index),
        "{:06d}".format(frame_index),
    }

    candidates = list(
        output_dir.rglob("boxes.npz")
    )

    matched = []

    for path in candidates:
        parent = path.parent.name

        if any(token in parent for token in tokens):
            matched.append(path.parent)

    if matched:
        return sorted(
            matched,
            key=lambda x: x.stat().st_mtime,
        )[-1]

    if candidates:
        return sorted(
            [p.parent for p in candidates],
            key=lambda x: x.stat().st_mtime,
        )[-1]

    raise RuntimeError(
        "Could not find boxes.npz under {}".format(
            output_dir
        )
    )


def load_legacy_boxes(frame_dir):
    path = Path(frame_dir) / "boxes.npz"

    data = np.load(
        str(path),
        allow_pickle=True,
    )

    keys = set(data.files)

    required = {
        "gt_boxes",
        "baseline_boxes",
        "baseline_scores",
        "late_boxes",
        "late_scores",
    }

    missing = required.difference(keys)

    if missing:
        raise KeyError(
            "Legacy boxes.npz is missing keys {}. "
            "Available keys: {}".format(
                sorted(missing),
                sorted(keys),
            )
        )

    return {
        key: data[key]
        for key in data.files
    }


def call_oracle(
    model,
    cached,
    criterion,
    label_dict,
    base_mask,
    oracle_topk,
):
    """Call existing run_independent_oracle using its real signature."""

    fn = base.run_independent_oracle
    sig = inspect.signature(fn)

    pool = {
        "model": model,
        "cached": cached,
        "criterion": criterion,
        "label_dict": label_dict,
        "base_mask": base_mask,
        "oracle_topk": oracle_topk,
    }

    kwargs = {}

    missing = []

    for name, param in sig.parameters.items():
        if name in pool:
            kwargs[name] = pool[name]
        elif (
            param.default is inspect.Parameter.empty
            and param.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ):
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Cannot automatically call run_independent_oracle. "
            "Unresolved arguments: {}. Signature={}".format(
                missing,
                sig,
            )
        )

    result = fn(**kwargs)

    if not isinstance(result, tuple):
        raise RuntimeError(
            "run_independent_oracle must return a tuple"
        )

    if len(result) < 1:
        raise RuntimeError(
            "run_independent_oracle returned empty tuple"
        )

    return result[0]


def pick_argument(arguments, names, required=True):
    for name in names:
        if name in arguments:
            return arguments[name]

    if required:
        raise KeyError(
            "Could not resolve process_frame argument. "
            "Tried: {}. Available: {}".format(
                names,
                sorted(arguments.keys()),
            )
        )

    return None


def process_frame_4way(*args, **kwargs):
    """
    Wrapper around the validated legacy process_frame.

    Legacy run provides:
        GT
        Baseline
        Late Fusion

    This wrapper additionally computes:
        Late Fusion + RBV
        Oracle + Late Fusion + RBV
    """

    sig = inspect.signature(
        _legacy_process_frame
    )

    bound = sig.bind_partial(
        *args,
        **kwargs
    )

    bound.apply_defaults()

    arguments = bound.arguments

    model = pick_argument(
        arguments,
        ["model"],
    )

    criterion = pick_argument(
        arguments,
        ["criterion"],
    )

    dataset = pick_argument(
        arguments,
        ["dataset"],
    )

    device = pick_argument(
        arguments,
        ["device"],
    )

    frame_index = pick_argument(
        arguments,
        [
            "frame_index",
            "sample_idx",
            "sample_id",
            "dataset_index",
        ],
    )

    # The validated legacy process_frame() does not receive batch_data.
    # It receives dataset + frame_index and loads/collates the frame
    # internally. Reconstruct exactly the same evaluation batch here.
    from opencood.tools import train_utils

    raw_sample = dataset[int(frame_index)]

    if raw_sample is None:
        raise RuntimeError(
            "Dataset returned None for validation frame {}".format(
                frame_index
            )
        )

    batch_data = dataset.collate_batch_test(
        [raw_sample]
    )

    if batch_data is None:
        raise RuntimeError(
            "collate_batch_test returned None for frame {}".format(
                frame_index
            )
        )

    batch_data = train_utils.to_device(
        batch_data,
        device,
    )

    output_dir = pick_argument(
        arguments,
        [
            "output_dir",
            "save_dir",
        ],
    )

    oracle_topk = pick_argument(
        arguments,
        [
            "oracle_topk",
        ],
    )

    # --------------------------------------------------------------
    # First execute the already validated legacy visualizer.
    #
    # Explicitly disable RBV so its saved baseline/late results remain
    # the original controlled comparison.
    # --------------------------------------------------------------
    with temporary_env(
        {
            "ENABLE_ROTATED_BOX_VOTING": "0",
        }
    ):
        legacy_result = _legacy_process_frame(
            *args,
            **kwargs
        )

    frame_dir = locate_frame_dir(
        output_dir,
        int(frame_index),
    )

    legacy = load_legacy_boxes(
        frame_dir
    )

    baseline_boxes = legacy[
        "baseline_boxes"
    ]

    baseline_scores = legacy[
        "baseline_scores"
    ]

    late_boxes = legacy[
        "late_boxes"
    ]

    late_scores = legacy[
        "late_scores"
    ]

    gt_boxes = legacy[
        "gt_boxes"
    ]

    # --------------------------------------------------------------
    # Reuse the same batch for the two new branches.
    # --------------------------------------------------------------
    ego_data = batch_data["ego"]

    with torch.no_grad():
        cached = model._extract_cached_context(
            ego_data
        )

        normal_output = (
            model._forward_from_cached_context(
                cached,
                return_communication_debug=True,
            )
        )

        if "communication_masks" not in normal_output:
            raise KeyError(
                "normal_output has no communication_masks"
            )

        base_mask = normal_output[
            "communication_masks"
        ]

        if base_mask is None:
            raise RuntimeError(
                "communication_masks is None"
            )

        # ----------------------------------------------------------
        # Branch 3:
        # Normal Where2comm + Late Fusion + RBV.
        # ----------------------------------------------------------
        late_rbv_boxes, late_rbv_scores, _ = \
            postprocess_late_rbv(
                dataset=dataset,
                batch_data=batch_data,
                output_dict=normal_output,
                threshold=LATE_ONLY_THRESHOLD,
                beta=LATE_ONLY_BETA,
                rbv_enabled=True,
                rbv_iou=RBV_IOU_THRESHOLD,
            )

        # ----------------------------------------------------------
        # Branch 4:
        # Oracle K80 -> Late Fusion -> NMS -> Post-NMS RBV.
        # ----------------------------------------------------------
        label_dict = ego_data["label_dict"]

        oracle_output = call_oracle(
            model=model,
            cached=cached,
            criterion=criterion,
            label_dict=label_dict,
            base_mask=base_mask,
            oracle_topk=int(oracle_topk),
        )

        oracle_late_rbv_boxes, \
        oracle_late_rbv_scores, _ = \
            postprocess_late_rbv(
                dataset=dataset,
                batch_data=batch_data,
                output_dict=oracle_output,
                threshold=ORACLE_LATE_THRESHOLD,
                beta=ORACLE_LATE_BETA,
                rbv_enabled=True,
                rbv_iou=RBV_IOU_THRESHOLD,
            )

    point_cloud = extract_point_cloud(
        batch_data
    )

    compare_path = (
        Path(frame_dir)
        / "compare_4way.png"
    )

    plot_four_way(
        output_path=compare_path,
        frame_index=int(frame_index),
        point_cloud=point_cloud,
        gt_boxes=gt_boxes,
        baseline_boxes=baseline_boxes,
        late_boxes=late_boxes,
        late_rbv_boxes=late_rbv_boxes,
        oracle_late_rbv_boxes=oracle_late_rbv_boxes,
    )

    # --------------------------------------------------------------
    # Save all four methods for later re-plotting without inference.
    # --------------------------------------------------------------
    np.savez_compressed(
        str(
            Path(frame_dir)
            / "boxes_4way.npz"
        ),

        gt_boxes=tensor_to_numpy(
            gt_boxes
        ),

        baseline_boxes=tensor_to_numpy(
            baseline_boxes
        ),
        baseline_scores=tensor_to_numpy(
            baseline_scores
        ),

        late_boxes=tensor_to_numpy(
            late_boxes
        ),
        late_scores=tensor_to_numpy(
            late_scores
        ),

        late_rbv_boxes=tensor_to_numpy(
            late_rbv_boxes
        ),
        late_rbv_scores=tensor_to_numpy(
            late_rbv_scores
        ),

        oracle_late_rbv_boxes=tensor_to_numpy(
            oracle_late_rbv_boxes
        ),
        oracle_late_rbv_scores=tensor_to_numpy(
            oracle_late_rbv_scores
        ),
    )

    frame_summary = {
        "frame_index": int(frame_index),

        "baseline_predictions":
            0 if baseline_boxes is None
            else int(len(baseline_boxes)),

        "late_predictions":
            0 if late_boxes is None
            else int(len(late_boxes)),

        "late_rbv_predictions":
            0 if late_rbv_boxes is None
            else int(len(late_rbv_boxes)),

        "oracle_late_rbv_predictions":
            0 if oracle_late_rbv_boxes is None
            else int(len(oracle_late_rbv_boxes)),

        "configs": {
            "late": {
                "threshold":
                    LATE_ONLY_THRESHOLD,
                "beta":
                    LATE_ONLY_BETA,
            },

            "late_rbv": {
                "threshold":
                    LATE_ONLY_THRESHOLD,
                "beta":
                    LATE_ONLY_BETA,
                "rbv_iou":
                    RBV_IOU_THRESHOLD,
            },

            "oracle_late_rbv": {
                "oracle_topk":
                    int(oracle_topk),
                "oracle_patch_size":
                    [2, 3],
                "threshold":
                    ORACLE_LATE_THRESHOLD,
                "beta":
                    ORACLE_LATE_BETA,
                "rbv_iou":
                    RBV_IOU_THRESHOLD,
            },
        },
    }

    with open(
        str(
            Path(frame_dir)
            / "summary_4way.json"
        ),
        "w",
    ) as f:
        json.dump(
            frame_summary,
            f,
            indent=2,
        )

    print(
        "[4way] frame={} -> {}".format(
            frame_index,
            compare_path,
        )
    )

    return legacy_result


# Replace only the per-frame worker.
base.process_frame = process_frame_4way


if __name__ == "__main__":
    base.main()
