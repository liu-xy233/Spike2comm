# -*- coding: utf-8 -*-
"""Dense patch-level Oracle Utility evaluation."""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.comm_modules.patch_utils import (
    apply_patch_state,
    communication_rate_from_mask,
    patch_confidence_stats,
    topk_patch_ids,
)
from opencood.tools import train_utils
from opencood.utils import eval_utils_global_safe as eval_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dense Oracle patch utility.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--eval_epoch", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--patch_h", type=int, default=3)
    parser.add_argument("--patch_w", type=int, default=2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--utility_cache",
        type=str,
        required=True,
        help="dense_patch_utility.pt from a completed Oracle run.",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[10, 20, 40],
        help="Oracle Top-K values to replay.",
    )
    parser.add_argument(
        "--late_threshold",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--late_beta",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--original_only",
        action="store_true",
        help="Evaluate only the original Where2comm mask without counterfactual utility or Top-K baselines.",
    )
    return parser.parse_args()


def build_confidence_map(psm_single, target_size):
    confidence = psm_single.sigmoid().max(dim=1, keepdim=True)[0]
    if confidence.shape[-2:] != target_size:
        confidence = F.interpolate(confidence, size=target_size, mode="nearest")
    return confidence


def clone_with_selected_patches(base_mask, selected_patch_ids, patch_size):
    mask = base_mask.clone()
    if mask.dim() != 4:
        raise ValueError("Expected flattened mask [N,1,H,W], got {}.".format(mask.shape))
    mask[1:] = 0
    mask[0] = 1
    for patch_id in selected_patch_ids:
        mask = apply_patch_state(mask, patch_id, 1.0, patch_size=patch_size, agent_index=1)
    return mask



# Oracle K80 + Confidence-Aware Late Fusion sweep.
# Oracle forward is shared by all combinations.
LATE_SWEEP = [
    (0.30, 0.45),
    (0.30, 0.50),
    (0.30, 0.55),
    (0.30, 0.60),

    (0.35, 0.45),
    (0.35, 0.50),
    (0.35, 0.55),
    (0.35, 0.60),

    (0.40, 0.45),
    (0.40, 0.50),
    (0.40, 0.55),
    (0.40, 0.60),
]


def late_method_name(threshold, beta):
    return "oracle_late_t{:03d}_b{:03d}".format(
        int(round(float(threshold) * 100)),
        int(round(float(beta) * 100)),
    )


def new_result_stat():
    return {
        0.3: {"tp": [], "fp": [], "score": [], "gt": 0},
        0.5: {"tp": [], "fp": [], "score": [], "gt": 0},
        0.7: {"tp": [], "fp": [], "score": [], "gt": 0},
    }


def _post_process_with_late_cfg(
    dataset,
    batch_data,
    output_dict,
    enabled,
    threshold=None,
    beta=None,
):
    """Temporarily override confidence-aware late-fusion postprocess config."""
    if not hasattr(dataset, "post_processor"):
        raise AttributeError("Dataset has no post_processor attribute")

    params = dataset.post_processor.params
    cfg_key = "confidence_aware_late_fusion"

    old_exists = cfg_key in params
    old_cfg = params.get(cfg_key, None)

    cfg = dict(old_cfg) if isinstance(old_cfg, dict) else {}
    cfg["enabled"] = bool(enabled)

    if enabled:
        if threshold is None or beta is None:
            raise ValueError(
                "Late fusion requires both threshold and beta"
            )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                "Late threshold must be in [0,1], got {}".format(threshold)
            )
        if not 0.0 < float(beta) < 1.0:
            raise ValueError(
                "Late beta must be in (0,1), got {}".format(beta)
            )

        cfg["score_threshold"] = float(threshold)
        cfg["collaborator_score_weight"] = float(beta)

    params[cfg_key] = cfg

    try:
        return dataset.post_process(
            batch_data, {"ego": output_dict}
        )
    finally:
        if old_exists:
            params[cfg_key] = old_cfg
        else:
            params.pop(cfg_key, None)


def _accumulate_detection_stats(
    pred_box_tensor,
    pred_score,
    gt_box_tensor,
    result_stat,
):
    if pred_box_tensor is None:
        return

    for iou in [0.3, 0.5, 0.7]:
        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat,
            iou,
        )


def update_detection_stats(
    dataset,
    batch_data,
    output_dict,
    result_stat,
):
    """Normal postprocess with Late Fusion explicitly disabled."""
    pred_box_tensor, pred_score, gt_box_tensor =         _post_process_with_late_cfg(
            dataset,
            batch_data,
            output_dict,
            enabled=False,
        )

    _accumulate_detection_stats(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat,
    )


def update_detection_stats_late(
    dataset,
    batch_data,
    output_dict,
    result_stat,
    threshold,
    beta,
):
    """Postprocess the SAME Oracle output with Late Fusion enabled."""
    pred_box_tensor, pred_score, gt_box_tensor =         _post_process_with_late_cfg(
            dataset,
            batch_data,
            output_dict,
            enabled=True,
            threshold=threshold,
            beta=beta,
        )

    _accumulate_detection_stats(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat,
    )


def scalar_loss(criterion, output_dict, label_dict):
    return criterion(output_dict, label_dict).detach()


def should_stop(processed, max_samples):
    return max_samples is not None and max_samples > 0 and processed >= max_samples


def evaluate_original_only(opt, data_loader, dataset, model, device, epoch_id):
    result_stat = new_result_stat()
    comm_rates = []
    processed = 0

    for _, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if should_stop(processed, opt.max_samples):
            break
        if batch_data is None:
            continue

        batch_data = train_utils.to_device(batch_data, device)
        ego_data = batch_data["ego"]

        with torch.no_grad():
            cached = model._extract_cached_context(ego_data)
            original_output = model._forward_from_cached_context(
                cached, return_communication_debug=True
            )
            base_mask = original_output["communication_masks"]
            if base_mask is None:
                raise RuntimeError("Original-only evaluation requires communication masks.")

            update_detection_stats(dataset, batch_data, original_output, result_stat)
            comm_rates.append(
                float(communication_rate_from_mask(base_mask, cached["record_len"]).item())
            )

        processed += 1

    ap_30 = eval_utils.calculate_ap(result_stat, 0.3)[0]
    ap_50 = eval_utils.calculate_ap(result_stat, 0.5)[0]
    ap_70 = eval_utils.calculate_ap(result_stat, 0.7)[0]
    summary = {
        "epoch": epoch_id,
        "samples": processed,
        "original": {
            "ap_30": ap_30,
            "ap_50": ap_50,
            "ap_70": ap_70,
            "comm_rate": sum(comm_rates) / len(comm_rates) if comm_rates else 0.0,
        },
    }

    with open(os.path.join(opt.output_dir, "original_only_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))



# Formal single-Late configuration.
# Avoid repeated 12-way RBV/Shapely post-processing.
LATE_SWEEP = [(0.30, 0.55)]

def main():
    opt = parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)
    random_gen = random.Random(opt.seed)
    patch_size = (opt.patch_h, opt.patch_w)

    hypes = yaml_utils.load_yaml(None, opt)
    hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epoch_id, model = train_utils.load_saved_model(opt.model_dir, model, opt.eval_epoch)
    model.eval()
    criterion = train_utils.create_loss(hypes)

    if not hasattr(model, "_extract_cached_context"):
        raise RuntimeError("This script requires a Where2comm model with cache APIs.")

    if opt.original_only:
        evaluate_original_only(opt, data_loader, dataset, model, device, epoch_id)
        return

    # --------------------------------------------------
    # Replay previously computed dense Oracle utilities.
    # No patch-wise OFF/ON counterfactual forward is needed.
    # --------------------------------------------------
    cache_records = torch.load(
        opt.utility_cache,
        map_location="cpu",
    )

    if not isinstance(cache_records, list):
        raise TypeError(
            "Expected utility cache to contain a list, got {}"
            .format(type(cache_records))
        )

    cache_by_sample = {}

    for rec in cache_records:
        sample_id = int(rec["sample_id"])

        if sample_id in cache_by_sample:
            raise RuntimeError(
                "Duplicate sample_id {} in utility cache"
                .format(sample_id)
            )

        cache_by_sample[sample_id] = rec

    ks = sorted(set(int(k) for k in opt.ks))

    if not ks:
        raise ValueError("--ks must contain at least one K")

    if any(k <= 0 for k in ks):
        raise ValueError(
            "All K values must be positive, got {}"
            .format(ks)
        )

    print(
        "[ReplayOracle] loaded {} cached frames from {}"
        .format(len(cache_records), opt.utility_cache)
    )
    print(
        "[ReplayOracle] K values = {}"
        .format(ks)
    )
    print(
        "[ReplayOracle] Late Fusion: threshold={} beta={}"
        .format(opt.late_threshold, opt.late_beta)
    )

    result_stats = {}
    comm_rates = {}

    for k in ks:
        oracle_name = "oracle_k{}".format(k)
        late_name = (
            "oracle_k{}_late_t{:03d}_b{:03d}".format(
                k,
                int(round(opt.late_threshold * 100)),
                int(round(opt.late_beta * 100)),
            )
        )

        result_stats[oracle_name] = new_result_stat()
        result_stats[late_name] = new_result_stat()

        comm_rates[oracle_name] = []
        comm_rates[late_name] = []

    processed = 0

    for batch_idx, batch_data in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
    ):
        if should_stop(processed, opt.max_samples):
            break

        if batch_data is None:
            continue

        batch_data = train_utils.to_device(
            batch_data,
            device,
        )

        ego_data = batch_data["ego"]

        sample_id = int(
            ego_data.get("sample_idx", batch_idx)
        )

        with torch.no_grad():

            # Run backbone/cache ONCE for this frame.
            cached = model._extract_cached_context(
                ego_data
            )

            if int(cached["record_len"][0]) <= 1:
                continue

            # Obtain the normal Where2comm mask.
            # This is needed as the template for constructing
            # replayed Top-K masks.
            original_output = (
                model._forward_from_cached_context(
                    cached,
                    return_communication_debug=True,
                )
            )

            base_mask = original_output[
                "communication_masks"
            ]

            if base_mask is None:
                raise RuntimeError(
                    "Replay evaluation requires "
                    "communication masks."
                )

            if sample_id not in cache_by_sample:
                raise KeyError(
                    "sample_id {} is missing from "
                    "utility cache"
                    .format(sample_id)
                )

            rec = cache_by_sample[sample_id]

            cached_patch_size = tuple(
                int(x) for x in rec["patch_size"]
            )

            if cached_patch_size != tuple(patch_size):
                raise RuntimeError(
                    "Patch-size mismatch at sample {}: "
                    "cache={} current={}"
                    .format(
                        sample_id,
                        cached_patch_size,
                        tuple(patch_size),
                    )
                )

            current_feature_size = tuple(
                int(x)
                for x in base_mask.shape[-2:]
            )

            cached_feature_size = tuple(
                int(x)
                for x in rec["feature_size"]
            )

            if (
                current_feature_size
                != cached_feature_size
            ):
                raise RuntimeError(
                    "Feature-size mismatch at sample {}: "
                    "cache={} current={}"
                    .format(
                        sample_id,
                        cached_feature_size,
                        current_feature_size,
                    )
                )

            utilities = torch.tensor(
                rec["utility"],
                dtype=torch.float32,
                device=device,
            )

            # Expected number of patches for the current
            # feature map and patch dimensions.
            h, w = current_feature_size
            ph, pw = patch_size

            expected_num_patches = (
                ((h + ph - 1) // ph)
                * ((w + pw - 1) // pw)
            )

            if utilities.numel() != expected_num_patches:
                raise RuntimeError(
                    "Utility-count mismatch at sample {}: "
                    "cache={} expected={}"
                    .format(
                        sample_id,
                        utilities.numel(),
                        expected_num_patches,
                    )
                )

            # ------------------------------------------
            # Reuse the SAME utility vector for all K.
            # Only final communication/fusion forward
            # changes between K values.
            # ------------------------------------------
            for k in ks:

                oracle_ids = topk_patch_ids(
                    utilities,
                    k,
                    threshold=0.0,
                    largest=True,
                )

                mask = clone_with_selected_patches(
                    base_mask,
                    oracle_ids,
                    patch_size,
                )

                output = (
                    model._forward_from_cached_context(
                        cached,
                        override_communication_mask=mask,
                    )
                )

                oracle_name = "oracle_k{}".format(k)

                late_name = (
                    "oracle_k{}_late_t{:03d}_b{:03d}"
                    .format(
                        k,
                        int(
                            round(
                                opt.late_threshold * 100
                            )
                        ),
                        int(
                            round(
                                opt.late_beta * 100
                            )
                        ),
                    )
                )

                # Oracle-only
                update_detection_stats(
                    dataset,
                    batch_data,
                    output,
                    result_stats[oracle_name],
                )

                # Same Oracle output reused by
                # Late Fusion + optional RBV.
                update_detection_stats_late(
                    dataset,
                    batch_data,
                    output,
                    result_stats[late_name],
                    threshold=opt.late_threshold,
                    beta=opt.late_beta,
                )

                comm_rate = float(
                    communication_rate_from_mask(
                        mask,
                        cached["record_len"],
                    ).item()
                )

                comm_rates[oracle_name].append(
                    comm_rate
                )

                comm_rates[late_name].append(
                    comm_rate
                )

        processed += 1

    summary = {
        "epoch": epoch_id,
        "samples": processed,
        "utility_cache": opt.utility_cache,
        "ks": ks,
        "late_threshold": opt.late_threshold,
        "late_beta": opt.late_beta,
        "methods": {},
    }

    for method_name, stat in result_stats.items():

        ap_30 = eval_utils.calculate_ap(
            stat, 0.3
        )[0]

        ap_50 = eval_utils.calculate_ap(
            stat, 0.5
        )[0]

        ap_70 = eval_utils.calculate_ap(
            stat, 0.7
        )[0]

        rates = comm_rates[method_name]

        summary["methods"][method_name] = {
            "ap_30": ap_30,
            "ap_50": ap_50,
            "ap_70": ap_70,
            "comm_rate": (
                sum(rates) / len(rates)
                if rates else 0.0
            ),
        }

    os.makedirs(
        opt.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        opt.output_dir,
        "oracle_replay_multik_summary.json",
    )

    with open(output_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(json.dumps(summary, indent=2))



if __name__ == "__main__":
    main()
