# -*- coding: utf-8 -*-
"""Evaluate ValueNet Top-K communication against Oracle on DAIR-V2X."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.data_utils.value_predictor_dataset import ValuePredictorDataset
from opencood.models.comm_modules.value_predictor_factory import (
    build_value_predictor,
    validate_value_predictor_checkpoint,
)
from opencood.models.comm_modules.patch_utils import (
    communication_rate_from_mask,
    dense_mask_from_patch_values,
    patchify_map,
    topk_patch_ids,
)
from opencood.tools import train_utils
from opencood.tools.align_value_predictor_oracle_labels import build_theta
from opencood.tools.value_predictor_geometry import geometry_valid_mask
from opencood.tools.eval_patch_oracle_replay_multik_safe import (
    clone_with_selected_patches,
    new_result_stat,
    update_detection_stats,
    update_detection_stats_late,
)
from opencood.utils import eval_utils_global_safe as eval_utils


def inverse_map_topk(
    prediction, valid_mask, metadata, height, width, patch_size, k, device
):
    """Map Ego-grid predictions back to the neighbor source grid."""
    ph, pw = patch_size
    expected = (height // ph) * (width // pw)

    prediction = prediction.reshape(-1).float()
    valid_mask = valid_mask.reshape(-1).bool()

    if prediction.numel() != expected:
        raise ValueError(
            "Prediction count mismatch: {} vs {}".format(
                prediction.numel(), expected
            )
        )
    if valid_mask.numel() != expected:
        raise ValueError("valid_mask length mismatch")

    # The training labels are valid only at supervised Ego-grid patches.
    # Exclude invalid patches when aggregating scores after inverse warping.
    pred_dense = dense_mask_from_patch_values(
        prediction * valid_mask.float(),
        height,
        width,
        patch_size,
    ).unsqueeze(0)

    valid_dense = dense_mask_from_patch_values(
        valid_mask.float(),
        height,
        width,
        patch_size,
    ).unsqueeze(0)

    theta = build_theta(
        metadata,
        height,
        width,
        downsample_rate=8,
        discrete_ratio=0.2,
    ).to(device)

    # grid_sample's theta maps output coordinates to input coordinates.
    # The forward alignment uses Ego output -> neighbor input.
    # Invert the homogeneous affine matrix for neighbor-grid output.
    theta_h = torch.eye(3, dtype=theta.dtype, device=device)
    theta_h[:2, :] = theta
    # Invert the tiny affine matrix on CPU to avoid CUDA cuSolver failures.
    theta_inv = torch.inverse(
        theta_h.detach().cpu()
    )[:2, :].unsqueeze(0).to(device=device, dtype=theta.dtype)

    grid = F.affine_grid(
        theta_inv,
        pred_dense.shape,
        align_corners=False,
    )

    source_scores = F.grid_sample(
        pred_dense,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]

    source_valid = F.grid_sample(
        valid_dense,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]

    score_patches = patchify_map(source_scores, patch_size)
    valid_patches = patchify_map(source_valid, patch_size)

    valid_count = valid_patches.sum(dim=(-1, -2))
    valid_fraction = valid_count / float(ph * pw)

    # Average only over cells whose aligned predictions are valid.
    patch_scores = (
        score_patches.sum(dim=(-1, -2))
        / valid_count.clamp_min(1.0)
    )

    eligible = (valid_fraction >= 0.5) & (valid_count > 0)
    eligible_ids = torch.nonzero(
        eligible, as_tuple=False
    ).flatten()

    if eligible_ids.numel() < k:
        raise RuntimeError(
            "Only {} valid source patches; cannot select Top-{}".format(
                eligible_ids.numel(), k
            )
        )

    selected = eligible_ids[
        torch.topk(patch_scores[eligible_ids], k=k).indices
    ]

    return selected.cpu().tolist(), int(eligible_ids.numel()), eligible


def calculate_ap(result_stat, iou):
    return float(eval_utils.calculate_ap(result_stat, iou)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--eval_epoch", type=int, default=26)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--patch_h", type=int, default=2)
    parser.add_argument("--patch_w", type=int, default=3)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--late_threshold", type=float, default=0.30)
    parser.add_argument("--late_beta", type=float, default=0.55)
    parser.add_argument("--predictor_config", default=(
        "outputs/value_predictor_formal_ep26/"
        "value_predictor_aligned_masked_v1.yaml"
    ))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_dir", default=(
        "outputs/value_predictor_formal_ep26/features_val_full"
    ))
    parser.add_argument("--label_path", default=(
        "outputs/value_predictor_formal_ep26/"
        "oracle_val_ego_aligned_full.pt"
    ))
    parser.add_argument("--oracle_cache", default=(
        "outputs/value_predictor_formal_ep26/"
        "oracle_val_full/dense_patch_utility.pt"
    ))
    opt = parser.parse_args()

    patch_size = (opt.patch_h, opt.patch_w)
    os.makedirs(opt.output_dir, exist_ok=True)

    for path in (
        opt.predictor_config,
        opt.checkpoint,
        opt.feature_dir,
        opt.label_path,
        opt.oracle_cache,
        os.path.join(opt.model_dir, "config.yaml"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError("Required path missing: " + path)

    print("[INFO] RBV environment:",
          os.environ.get("ENABLE_ROTATED_BOX_VOTING", "<unset>"))

    with open(opt.predictor_config) as f:
        predictor_cfg = yaml.safe_load(f)

    device_name = predictor_cfg["training"].get("device", "cuda")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    feature_dataset = ValuePredictorDataset(
        opt.feature_dir,
        opt.label_path,
        in_channels=int(
            predictor_cfg["model"].get("in_channels", 128)
        ),
        num_patch=int(
            predictor_cfg["model"].get("num_patch", 1050)
        ),
    )

    predictor = build_value_predictor(
        predictor_cfg["model"]
    ).to(device)

    checkpoint = torch.load(
        opt.checkpoint,
        map_location=device,
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unexpected ValueNet checkpoint format")

    validate_value_predictor_checkpoint(predictor, checkpoint)

    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict"),
    )
    if state is None:
        raise RuntimeError(
            "Checkpoint has no model_state_dict/state_dict"
        )

    predictor.load_state_dict(state, strict=True)
    predictor.eval()

    print(
        "[PASS] ValueNet architecture and checkpoint verified."
    )

    # Build the detector validation loader exactly as the Oracle replay does.
    hypes = yaml_utils.load_yaml(None, opt)
    hypes["validate_dir"] = hypes["test_dir"]

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    if len(feature_dataset) != len(loader):
        raise RuntimeError(
            "Feature/detector validation count mismatch: {} vs {}".format(
                len(feature_dataset), len(loader)
            )
        )

    cache_records = torch.load(opt.oracle_cache, map_location="cpu")
    cache_by_sample = {}
    for rec in cache_records:
        sid = int(rec["sample_id"])
        if sid in cache_by_sample:
            raise RuntimeError("Duplicate Oracle sample_id: {}".format(sid))
        cache_by_sample[sid] = rec

    if len(cache_by_sample) != len(feature_dataset):
        raise RuntimeError(
            "Oracle/feature count mismatch: {} vs {}".format(
                len(cache_by_sample), len(feature_dataset)
            )
        )

    detector = train_utils.create_model(hypes).to(device)
    epoch_id, detector = train_utils.load_saved_model(
        opt.model_dir, detector, opt.eval_epoch
    )
    detector.eval()

    if not hasattr(detector, "_extract_cached_context"):
        raise RuntimeError("Detector cache API is unavailable")

    names = (
        "original",
        "original_late",
        "oracle_k10",
        "oracle_k10_late",
        "valuenet_k10",
        "valuenet_k10_late",
    )
    stats = {name: new_result_stat() for name in names}
    comm_rates = {name: [] for name in names}

    evaluated = 0
    skipped_single_agent = 0

    for batch_idx, batch_data in tqdm(
        enumerate(loader), total=len(loader)
    ):
        if opt.max_samples > 0 and evaluated >= opt.max_samples:
            break
        if batch_data is None:
            continue

        batch_data = train_utils.to_device(batch_data, device)
        ego_data = batch_data["ego"]
        sample_id = int(ego_data.get("sample_idx", batch_idx))

        # The saved features and Oracle records are ordered by sample ID.
        if sample_id != batch_idx:
            raise RuntimeError(
                "Validation order mismatch: batch={} sample_id={}".format(
                    batch_idx, sample_id
                )
            )

        if sample_id not in cache_by_sample:
            raise KeyError("Oracle cache missing sample {}".format(sample_id))

        sample = feature_dataset[batch_idx]
        ego_feature = sample["ego_feature"].float()
        while ego_feature.dim() > 3 and ego_feature.shape[0] == 1:
            ego_feature = ego_feature.squeeze(0)
        if ego_feature.dim() != 3:
            raise RuntimeError(
                "Unexpected ego_feature shape: {}".format(
                    tuple(ego_feature.shape)
                )
            )

        ego_feature = ego_feature.unsqueeze(0).to(device)
        metadata_tensor = sample["neighbor_metadata"].float().reshape(-1)
        if metadata_tensor.numel() != 4:
            raise RuntimeError("Expected 4 neighbor metadata values")
        metadata = metadata_tensor.tolist()
        metadata_tensor = metadata_tensor.unsqueeze(0).to(device)

        rec = cache_by_sample[sample_id]
        if tuple(rec["patch_size"]) != patch_size:
            raise RuntimeError("Oracle patch-size mismatch")
        expected_patch_count = (
            (ego_feature.shape[-2] // patch_size[0])
            * (ego_feature.shape[-1] // patch_size[1])
        )
        if len(rec["utility"]) != expected_patch_count:
            raise RuntimeError(
                "Oracle/ValueNet patch count mismatch: {} vs {}".format(
                    len(rec["utility"]), expected_patch_count
                )
            )

        with torch.no_grad():
            cached = detector._extract_cached_context(ego_data)
            record_len = int(cached["record_len"][0])

            if record_len <= 1:
                skipped_single_agent += 1
                continue
            if record_len != 2:
                raise RuntimeError(
                    "This experiment expects exactly one neighbor; "
                    "sample {} has {} agents".format(sample_id, record_len)
                )

            original_output = detector._forward_from_cached_context(
                cached, return_communication_debug=True
            )
            base_mask = original_output["communication_masks"]
            if base_mask is None:
                raise RuntimeError("Communication mask unavailable")

            height, width = base_mask.shape[-2:]
            if tuple(ego_feature.shape[-2:]) != (height, width):
                raise RuntimeError(
                    "Feature/mask spatial mismatch: {} vs {}".format(
                        tuple(ego_feature.shape[-2:]), (height, width)
                    )
                )
            if tuple(rec["feature_size"]) != (height, width):
                raise RuntimeError("Oracle feature-size mismatch")

            # Inference-time geometry mask: no Oracle label is used
            # to decide which patches are eligible for communication.
            valid_mask = geometry_valid_mask(
                metadata,
                feature_size=(height, width),
                patch_size=patch_size,
                min_valid_fraction=0.5,
            ).to(device=device, dtype=torch.bool)

            if valid_mask.numel() != expected_patch_count:
                raise RuntimeError(
                    "Geometry mask patch count mismatch: {} vs {}".format(
                        valid_mask.numel(), expected_patch_count
                    )
                )

            update_detection_stats(
                dataset, batch_data, original_output, stats["original"]
            )
            update_detection_stats_late(
                dataset, batch_data, original_output,
                stats["original_late"],
                threshold=opt.late_threshold,
                beta=opt.late_beta,
            )
            base_rate = float(
                communication_rate_from_mask(
                    base_mask, cached["record_len"]
                ).item()
            )
            comm_rates["original"].append(base_rate)
            comm_rates["original_late"].append(base_rate)

            # Oracle utility is already in the neighbor/source grid.
            oracle_utility = torch.as_tensor(
                rec["utility"], dtype=torch.float32, device=device
            )
            if oracle_utility.numel() != expected_patch_count:
                raise RuntimeError(
                    "Oracle utility length mismatch: {} vs {}".format(
                        oracle_utility.numel(), expected_patch_count
                    )
                )

            # ValueNet predicts in Ego coordinates: inverse-map before Top-K.
            prediction = predictor(ego_feature, metadata_tensor)
            if isinstance(prediction, dict):
                prediction = prediction["value_map"]
            prediction = prediction.reshape(-1)

            predicted_ids, valid_source_count, eligible_source_mask = inverse_map_topk(
                prediction,
                valid_mask,
                metadata,
                height,
                width,
                patch_size,
                opt.topk,
                device,
            )

            # Fixed-budget Oracle: select exactly K patches from the
            # same geometry-valid source-coordinate candidate set.
            eligible_source_ids = torch.nonzero(
                eligible_source_mask, as_tuple=False
            ).flatten()

            if eligible_source_ids.numel() < opt.topk:
                raise RuntimeError(
                    "Only {} eligible source patches; cannot select Top-{}".format(
                        eligible_source_ids.numel(), opt.topk
                    )
                )

            oracle_ids = eligible_source_ids[
                torch.topk(
                    oracle_utility[eligible_source_ids],
                    k=opt.topk,
                    largest=True,
                ).indices
            ].detach().cpu().tolist()

            print(
                "[SAMPLE {}] Oracle Top-K={}, ValueNet Top-K={}, "
                "eligible source patches={}".format(
                    sample_id, len(oracle_ids), len(predicted_ids),
                    valid_source_count
                )
            )

            selected_methods = (
                ("oracle_k10", oracle_ids),
                ("valuenet_k10", predicted_ids),
            )

            for method_name, patch_ids in selected_methods:
                mask = clone_with_selected_patches(
                    base_mask, patch_ids, patch_size
                )
                output = detector._forward_from_cached_context(
                    cached, override_communication_mask=mask
                )

                update_detection_stats(
                    dataset, batch_data, output, stats[method_name]
                )
                late_name = method_name + "_late"
                update_detection_stats_late(
                    dataset, batch_data, output, stats[late_name],
                    threshold=opt.late_threshold,
                    beta=opt.late_beta,
                )

                rate = float(
                    communication_rate_from_mask(
                        mask, cached["record_len"]
                    ).item()
                )
                comm_rates[method_name].append(rate)
                comm_rates[late_name].append(rate)

        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No cooperative validation samples evaluated")

    summary = {
        "checkpoint": opt.checkpoint,
        "detector_epoch": int(epoch_id),
        "samples": evaluated,
        "skipped_single_agent": skipped_single_agent,
        "topk": opt.topk,
        "patch_size": list(patch_size),
        "late_threshold": opt.late_threshold,
        "late_beta": opt.late_beta,
        "rbv_environment": os.environ.get(
            "ENABLE_ROTATED_BOX_VOTING", "<unset>"
        ),
        "methods": {},
    }

    for name in names:
        rates = comm_rates[name]
        summary["methods"][name] = {
            "ap_30": calculate_ap(stats[name], 0.3),
            "ap_50": calculate_ap(stats[name], 0.5),
            "ap_70": calculate_ap(stats[name], 0.7),
            "comm_rate": (
                sum(rates) / len(rates) if rates else 0.0
            ),
        }

    output_path = Path(opt.output_dir) / "summary.json"
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== COMMUNICATION REPLAY SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print("[OK] Saved:", output_path)


if __name__ == "__main__":
    main()
