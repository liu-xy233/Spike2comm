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
from opencood.utils import eval_utils_global as eval_utils


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


def new_result_stat():
    return {
        0.3: {"tp": [], "fp": [], "score": [], "gt": 0},
        0.5: {"tp": [], "fp": [], "score": [], "gt": 0},
        0.7: {"tp": [], "fp": [], "score": [], "gt": 0},
    }


def update_detection_stats(dataset, batch_data, output_dict, result_stat):
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
        batch_data, {"ego": output_dict}
    )
    if pred_box_tensor is None:
        return
    for iou in [0.3, 0.5, 0.7]:
        eval_utils.caluclate_tp_fp(
            pred_box_tensor, pred_score, gt_box_tensor, result_stat, iou
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


def main():
    opt = parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)
    random_gen = random.Random(opt.seed)
    patch_size = (opt.patch_h, opt.patch_w)

    hypes = yaml_utils.load_yaml(None, opt)
    hypes["validate_dir"] = hypes["train_dir"]
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

    result_stats = {
        "oracle_topk": new_result_stat(),
        "confidence_topk": new_result_stat(),
        "random_topk": new_result_stat(),
        "original": new_result_stat(),
    }
    comm_rates = {name: [] for name in result_stats}
    dense_records = []
    processed = 0

    for batch_idx, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if should_stop(processed, opt.max_samples):
            break
        if batch_data is None:
            continue
        batch_data = train_utils.to_device(batch_data, device)
        ego_data = batch_data["ego"]
        label_dict = ego_data["label_dict"]
        sample_id = int(ego_data.get("sample_idx", batch_idx))

        with torch.no_grad():
            cached = model._extract_cached_context(ego_data)
            if int(cached["record_len"][0]) <= 1:
                continue

            original_output = model._forward_from_cached_context(
                cached, return_communication_debug=True
            )
            base_mask = original_output["communication_masks"]
            if base_mask is None:
                raise RuntimeError("Oracle evaluation requires communication masks.")
            target_size = base_mask.shape[-2:]
            confidence = build_confidence_map(cached["psm_single"], target_size)
            neighbor_confidence = confidence[1]
            stats = patch_confidence_stats(neighbor_confidence, patch_size=patch_size)
            num_patches = stats["mean"].numel()

            utilities = []
            for patch_id in range(num_patches):
                mask_off = apply_patch_state(
                    base_mask, patch_id, 0.0, patch_size=patch_size, agent_index=1
                )
                mask_on = apply_patch_state(
                    base_mask, patch_id, 1.0, patch_size=patch_size, agent_index=1
                )
                output_off = model._forward_from_cached_context(
                    cached, override_communication_mask=mask_off
                )
                output_on = model._forward_from_cached_context(
                    cached, override_communication_mask=mask_on
                )
                loss_off = scalar_loss(criterion, output_off, label_dict)
                loss_on = scalar_loss(criterion, output_on, label_dict)
                utilities.append(loss_off - loss_on)

            utilities = torch.stack(utilities)
            oracle_ids = topk_patch_ids(
                utilities,
                opt.topk,
                threshold=0.0,
                largest=True,
            )
            confidence_ids = topk_patch_ids(stats["mean"], opt.topk, largest=True)
            random_ids = random_gen.sample(range(num_patches), min(opt.topk, num_patches))

            method_patch_ids = {
                "oracle_topk": oracle_ids,
                "confidence_topk": confidence_ids,
                "random_topk": random_ids,
            }

            update_detection_stats(dataset, batch_data, original_output, result_stats["original"])
            comm_rates["original"].append(
                float(
                    communication_rate_from_mask(
                        base_mask, cached["record_len"]
                    ).item()
                )
            )

            for method_name, patch_ids in method_patch_ids.items():
                mask = clone_with_selected_patches(base_mask, patch_ids, patch_size)
                output = model._forward_from_cached_context(
                    cached, override_communication_mask=mask
                )
                update_detection_stats(dataset, batch_data, output, result_stats[method_name])
                comm_rates[method_name].append(
                    float(communication_rate_from_mask(mask, cached["record_len"]).item())
                )

            dense_records.append(
                {
                    "sample_id": sample_id,
                    "batch_idx": batch_idx,
                    "patch_size": list(patch_size),
                    "feature_size": list(target_size),
                    "topk": opt.topk,
                    "oracle_patch_ids": oracle_ids,
                    "confidence_patch_ids": confidence_ids,
                    "random_patch_ids": random_ids,
                    "utility": [float(x.item()) for x in utilities],
                    "patch_mean_confidence": [float(x.item()) for x in stats["mean"]],
                }
            )

        processed += 1

    summary = {"epoch": epoch_id, "samples": processed, "methods": {}}
    for method_name, stat in result_stats.items():
        ap_30 = eval_utils.calculate_ap(stat, 0.3)[0]
        ap_50 = eval_utils.calculate_ap(stat, 0.5)[0]
        ap_70 = eval_utils.calculate_ap(stat, 0.7)[0]
        rates = comm_rates[method_name]
        summary["methods"][method_name] = {
            "ap_30": ap_30,
            "ap_50": ap_50,
            "ap_70": ap_70,
            "comm_rate": sum(rates) / len(rates) if rates else 0.0,
        }

    with open(os.path.join(opt.output_dir, "oracle_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(dense_records, os.path.join(opt.output_dir, "dense_patch_utility.pt"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
