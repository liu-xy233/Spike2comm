# -*- coding: utf-8 -*-
"""Replay saved greedy Oracle diagnostics for multi-K AP evaluation."""

import argparse
import json
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.comm_modules.patch_utils import communication_rate_from_mask
from opencood.tools import train_utils
from opencood.tools.eval_patch_greedy_oracle import (
    calculate_ap_values,
    make_empty_neighbor_mask,
    mask_with_selected_patches,
    new_result_stat,
    patch_slices,
    should_stop,
    update_detection_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple K values from saved greedy Oracle diagnostics."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--eval_epoch", type=str, default=None)
    parser.add_argument(
        "--diagnostics",
        type=str,
        default="outputs/greedy_oracle_100_h2w3_k15/greedy_oracle_diagnostics.pt",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ks", type=str, required=True, help="Comma-separated K values, e.g. 8,10,11,12,15.")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def parse_ks(ks_text: str) -> List[int]:
    ks = [int(item.strip()) for item in ks_text.split(",") if item.strip()]
    if not ks:
        raise ValueError("--ks must contain at least one K.")
    if any(k <= 0 for k in ks):
        raise ValueError("All K values must be positive.")
    return ks


def load_diagnostics(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def assert_record_matches_sample(record, batch_idx, sample_id):
    if int(record["batch_idx"]) != int(batch_idx):
        raise RuntimeError(
            "Diagnostics batch_idx mismatch: record={}, dataloader={}.".format(
                record["batch_idx"], batch_idx
            )
        )
    if "sample_id" in record and sample_id is not None:
        if int(record["sample_id"]) != int(sample_id):
            raise RuntimeError(
                "Diagnostics sample_id mismatch: record={}, dataloader={}.".format(
                    record["sample_id"], sample_id
                )
            )


def assert_one_shot_prefix_sorted(record):
    utilities = record["one_shot_utilities"]
    patch_ids = record["one_shot_patch_ids"]
    for idx in range(len(patch_ids) - 1):
        cur_utility = float(utilities[int(patch_ids[idx])])
        next_utility = float(utilities[int(patch_ids[idx + 1])])
        if cur_utility + 1e-12 < next_utility:
            raise RuntimeError(
                "one_shot_patch_ids are not sorted by non-increasing utility "
                "for batch_idx {} at rank {}.".format(record.get("batch_idx"), idx)
            )


def ensure_record_supports_ks(record, ks):
    max_k = max(ks)
    if len(record["one_shot_patch_ids"]) < max_k:
        raise RuntimeError(
            "Diagnostics one_shot_patch_ids length {} is smaller than requested K {}.".format(
                len(record["one_shot_patch_ids"]), max_k
            )
        )
    if len(record["greedy_round_selected_patch_ids"]) < max_k:
        raise RuntimeError(
            "Diagnostics greedy_round_selected_patch_ids length {} is smaller than requested K {}.".format(
                len(record["greedy_round_selected_patch_ids"]), max_k
            )
        )


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def build_multik_stats(ks):
    return {
        k: {
            "standalone_one_shot": new_result_stat(),
            "greedy": new_result_stat(),
        }
        for k in ks
    }


def main():
    opt = parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)
    ks = parse_ks(opt.ks)
    diagnostics = load_diagnostics(opt.diagnostics)

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

    if not hasattr(model, "_extract_cached_context"):
        raise RuntimeError("This script requires a Where2comm model with cache APIs.")

    original_stat = new_result_stat()
    original_comm_rates = []
    multik_stats = build_multik_stats(ks)
    multik_comm_rates = {
        k: {"standalone_one_shot": [], "greedy": []}
        for k in ks
    }
    multik_overlaps = {k: [] for k in ks}
    processed = 0

    for batch_idx, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if should_stop(processed, opt.max_samples):
            break
        if batch_data is None:
            continue
        if processed >= len(diagnostics):
            raise RuntimeError(
                "Diagnostics ended before dataloader samples: processed={}.".format(processed)
            )

        record = diagnostics[processed]
        batch_data = train_utils.to_device(batch_data, device)
        ego_data = batch_data["ego"]
        sample_id = int(ego_data["sample_idx"]) if "sample_idx" in ego_data else None
        assert_record_matches_sample(record, batch_idx, sample_id)
        assert_one_shot_prefix_sorted(record)
        ensure_record_supports_ks(record, ks)

        with torch.no_grad():
            cached = model._extract_cached_context(ego_data)
            if int(cached["record_len"][0]) != 2:
                raise RuntimeError("This experiment expects DAIR-V2X two-agent batch=1 data.")

            original_output = model._forward_from_cached_context(
                cached, return_communication_debug=True
            )
            base_mask = original_output["communication_masks"]
            if base_mask is None:
                raise RuntimeError("Multi-K evaluation requires original communication masks.")

            height, width = base_mask.shape[-2:]
            if [int(height), int(width)] != [int(x) for x in record["feature_size"]]:
                raise RuntimeError(
                    "Feature size mismatch for batch_idx {}: record={}, current={}.".format(
                        batch_idx, record["feature_size"], [int(height), int(width)]
                    )
                )
            patch_size = tuple(int(x) for x in record["patch_size"])
            slices = patch_slices(height, width, patch_size)
            if len(slices) != int(record["num_patches"]):
                raise RuntimeError(
                    "num_patches mismatch for batch_idx {}: record={}, current={}.".format(
                        batch_idx, record["num_patches"], len(slices)
                    )
                )

            empty_neighbor_mask = make_empty_neighbor_mask(base_mask)

            update_detection_stats(dataset, batch_data, original_output, original_stat)
            original_comm_rates.append(
                float(communication_rate_from_mask(base_mask, cached["record_len"]).item())
            )

            for k in ks:
                one_shot_ids_k = [int(x) for x in record["one_shot_patch_ids"][:k]]
                greedy_ids_k = [
                    int(x) for x in record["greedy_round_selected_patch_ids"][:k]
                ]

                one_shot_mask = mask_with_selected_patches(
                    empty_neighbor_mask, one_shot_ids_k, slices
                )
                greedy_mask = mask_with_selected_patches(
                    empty_neighbor_mask, greedy_ids_k, slices
                )

                one_shot_output = model._forward_from_cached_context(
                    cached, override_communication_mask=one_shot_mask
                )
                greedy_output = model._forward_from_cached_context(
                    cached, override_communication_mask=greedy_mask
                )

                update_detection_stats(
                    dataset,
                    batch_data,
                    one_shot_output,
                    multik_stats[k]["standalone_one_shot"],
                )
                update_detection_stats(
                    dataset, batch_data, greedy_output, multik_stats[k]["greedy"]
                )

                multik_comm_rates[k]["standalone_one_shot"].append(
                    float(communication_rate_from_mask(one_shot_mask, cached["record_len"]).item())
                )
                multik_comm_rates[k]["greedy"].append(
                    float(communication_rate_from_mask(greedy_mask, cached["record_len"]).item())
                )
                overlap = len(set(one_shot_ids_k).intersection(greedy_ids_k)) / k
                multik_overlaps[k].append(overlap)

        processed += 1

    summary: Dict[str, object] = {
        "epoch": epoch_id,
        "samples": processed,
        "diagnostics": opt.diagnostics,
        "ks": ks,
        "original": calculate_ap_values(original_stat),
        "by_k": {},
    }
    summary["original"]["comm_rate"] = mean_or_zero(original_comm_rates)

    for k in ks:
        one_shot = calculate_ap_values(multik_stats[k]["standalone_one_shot"])
        greedy = calculate_ap_values(multik_stats[k]["greedy"])
        one_shot["comm_rate"] = mean_or_zero(multik_comm_rates[k]["standalone_one_shot"])
        greedy["comm_rate"] = mean_or_zero(multik_comm_rates[k]["greedy"])
        summary["by_k"]["K={}".format(k)] = {
            "K": k,
            "communication_rate": {
                "standalone_one_shot": one_shot["comm_rate"],
                "greedy": greedy["comm_rate"],
            },
            "standalone_one_shot": one_shot,
            "greedy": greedy,
            "greedy_minus_one_shot_ap": {
                "ap_30": greedy["ap_30"] - one_shot["ap_30"],
                "ap_50": greedy["ap_50"] - one_shot["ap_50"],
                "ap_70": greedy["ap_70"] - one_shot["ap_70"],
            },
            "mean_one_shot_greedy_overlap_at_K": mean_or_zero(multik_overlaps[k]),
        }

    with open(os.path.join(opt.output_dir, "multik_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

