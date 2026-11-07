# -*- coding: utf-8 -*-
"""Run one greedy trajectory to max K and evaluate multiple smaller K values."""

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
    scalar_loss,
    should_stop,
    update_detection_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate greedy K saturation from one max-K counterfactual search."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--eval_epoch", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--patch_h", type=int, default=2)
    parser.add_argument("--patch_w", type=int, default=3)
    parser.add_argument("--max_topk", type=int, default=30)
    parser.add_argument("--eval_ks", type=str, required=True)
    parser.add_argument("--sample_stride", type=int, default=1)
    return parser.parse_args()


def parse_eval_ks(eval_ks_text: str, max_topk: int) -> List[int]:
    eval_ks = [int(item.strip()) for item in eval_ks_text.split(",") if item.strip()]
    if not eval_ks:
        raise ValueError("--eval_ks must contain at least one K.")
    if any(k <= 0 for k in eval_ks):
        raise ValueError("All eval K values must be positive.")
    if any(k > max_topk for k in eval_ks):
        raise ValueError("All eval K values must be <= --max_topk.")
    return sorted(eval_ks)


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def new_multik_stats(eval_ks):
    return {
        k: {
            "standalone_one_shot": new_result_stat(),
            "greedy": new_result_stat(),
        }
        for k in eval_ks
    }


def main():
    opt = parse_args()
    if opt.sample_stride <= 0:
        raise ValueError("--sample_stride must be positive.")
    if opt.max_topk <= 0:
        raise ValueError("--max_topk must be positive.")

    os.makedirs(opt.output_dir, exist_ok=True)
    patch_size = (opt.patch_h, opt.patch_w)
    eval_ks = parse_eval_ks(opt.eval_ks, opt.max_topk)

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

    original_stat = new_result_stat()
    original_comm_rates = []
    multik_stats = new_multik_stats(eval_ks)
    multik_comm_rates = {
        k: {"standalone_one_shot": [], "greedy": []}
        for k in eval_ks
    }
    multik_overlaps = {k: [] for k in eval_ks}
    non_positive_by_k = {k: [] for k in eval_ks}
    greedy_best_by_round = [[] for _ in range(opt.max_topk)]
    diagnostics = []
    processed_batch_indices = []
    processed = 0

    for batch_idx, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if batch_idx % opt.sample_stride != 0:
            continue
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
            if int(cached["record_len"][0]) != 2:
                raise RuntimeError("This experiment expects DAIR-V2X two-agent batch=1 data.")

            original_output = model._forward_from_cached_context(
                cached, return_communication_debug=True
            )
            base_mask = original_output["communication_masks"]
            if base_mask is None:
                raise RuntimeError("Greedy K saturation requires original communication masks.")

            height, width = base_mask.shape[-2:]
            slices = patch_slices(height, width, patch_size)
            num_patches = len(slices)
            max_k = min(opt.max_topk, num_patches)

            empty_neighbor_mask = make_empty_neighbor_mask(base_mask)
            empty_output = model._forward_from_cached_context(
                cached, override_communication_mask=empty_neighbor_mask
            )
            loss_empty = scalar_loss(criterion, empty_output, label_dict)

            one_shot_utilities = []
            for patch_id in range(num_patches):
                only_patch_mask = mask_with_selected_patches(
                    empty_neighbor_mask, [patch_id], slices
                )
                output_only_patch = model._forward_from_cached_context(
                    cached, override_communication_mask=only_patch_mask
                )
                loss_only_patch = scalar_loss(criterion, output_only_patch, label_dict)
                one_shot_utilities.append(loss_empty - loss_only_patch)

            one_shot_utilities_tensor = torch.stack(one_shot_utilities)
            one_shot_ranked_patch_ids = (
                torch.argsort(one_shot_utilities_tensor, descending=True).cpu().tolist()
            )

            selected = []
            remaining = set(range(num_patches))
            current_loss = loss_empty
            greedy_round_selected_patch_ids = []
            greedy_round_best_marginal_utilities = []
            greedy_round_current_losses = [float(current_loss.item())]

            for _ in range(max_k):
                best_patch_id = None
                best_candidate_loss = None
                best_marginal_utility = None

                for patch_id in sorted(remaining):
                    candidate_ids = selected + [patch_id]
                    candidate_mask = mask_with_selected_patches(
                        empty_neighbor_mask, candidate_ids, slices
                    )
                    candidate_output = model._forward_from_cached_context(
                        cached, override_communication_mask=candidate_mask
                    )
                    candidate_loss = scalar_loss(criterion, candidate_output, label_dict)
                    marginal_utility = current_loss - candidate_loss

                    if (
                        best_marginal_utility is None
                        or marginal_utility.item() > best_marginal_utility.item()
                    ):
                        best_patch_id = patch_id
                        best_candidate_loss = candidate_loss
                        best_marginal_utility = marginal_utility

                selected.append(best_patch_id)
                remaining.remove(best_patch_id)
                current_loss = best_candidate_loss
                greedy_round_selected_patch_ids.append(int(best_patch_id))
                greedy_round_best_marginal_utilities.append(float(best_marginal_utility.item()))
                greedy_round_current_losses.append(float(current_loss.item()))
                greedy_best_by_round[len(selected) - 1].append(float(best_marginal_utility.item()))

            update_detection_stats(dataset, batch_data, original_output, original_stat)
            original_comm_rates.append(
                float(communication_rate_from_mask(base_mask, cached["record_len"]).item())
            )

            eval_outputs = {}
            for k in eval_ks:
                if k > max_k:
                    raise RuntimeError(
                        "Requested K {} exceeds available greedy trajectory length {}.".format(
                            k, max_k
                        )
                    )
                one_shot_ids_k = [int(x) for x in one_shot_ranked_patch_ids[:k]]
                greedy_ids_k = [int(x) for x in greedy_round_selected_patch_ids[:k]]

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

                one_shot_rate = float(
                    communication_rate_from_mask(one_shot_mask, cached["record_len"]).item()
                )
                greedy_rate = float(
                    communication_rate_from_mask(greedy_mask, cached["record_len"]).item()
                )
                multik_comm_rates[k]["standalone_one_shot"].append(one_shot_rate)
                multik_comm_rates[k]["greedy"].append(greedy_rate)
                overlap = len(set(one_shot_ids_k).intersection(greedy_ids_k)) / k
                multik_overlaps[k].append(overlap)
                non_positive_ratio = (
                    sum(1 for value in greedy_round_best_marginal_utilities[:k] if value <= 0)
                    / k
                )
                non_positive_by_k[k].append(non_positive_ratio)

                eval_outputs["K={}".format(k)] = {
                    "one_shot_patch_ids": one_shot_ids_k,
                    "greedy_patch_ids": greedy_ids_k,
                    "one_shot_greedy_overlap": float(overlap),
                    "greedy_non_positive_marginal_step_ratio": float(non_positive_ratio),
                    "communication_rate": {
                        "standalone_one_shot": one_shot_rate,
                        "greedy": greedy_rate,
                    },
                }

            diagnostics.append(
                {
                    "sample_id": sample_id,
                    "batch_idx": int(batch_idx),
                    "patch_size": list(patch_size),
                    "feature_size": [int(height), int(width)],
                    "num_patches": int(num_patches),
                    "max_topk": int(max_k),
                    "eval_ks": [int(k) for k in eval_ks],
                    "loss_empty": float(loss_empty.item()),
                    "one_shot_ranked_patch_ids": [int(x) for x in one_shot_ranked_patch_ids],
                    "one_shot_utilities": [
                        float(utility.item()) for utility in one_shot_utilities_tensor
                    ],
                    "greedy_round_selected_patch_ids": greedy_round_selected_patch_ids,
                    "greedy_round_best_marginal_utilities": greedy_round_best_marginal_utilities,
                    "greedy_round_current_losses": greedy_round_current_losses,
                    "eval_outputs": eval_outputs,
                    "original_comm_rate": original_comm_rates[-1],
                }
            )

        processed_batch_indices.append(int(batch_idx))
        processed += 1

    summary: Dict[str, object] = {
        "epoch": epoch_id,
        "samples": processed,
        "max_topk": int(opt.max_topk),
        "eval_ks": [int(k) for k in eval_ks],
        "patch_size": [int(opt.patch_h), int(opt.patch_w)],
        "sample_stride": int(opt.sample_stride),
        "processed_batch_indices": processed_batch_indices,
        "original": calculate_ap_values(original_stat),
        "by_k": {},
        "greedy_mean_best_marginal_utility_by_round": [
            mean_or_zero(values) for values in greedy_best_by_round
        ],
    }
    summary["original"]["comm_rate"] = mean_or_zero(original_comm_rates)

    for k in eval_ks:
        one_shot = calculate_ap_values(multik_stats[k]["standalone_one_shot"])
        greedy = calculate_ap_values(multik_stats[k]["greedy"])
        one_shot["comm_rate"] = mean_or_zero(multik_comm_rates[k]["standalone_one_shot"])
        greedy["comm_rate"] = mean_or_zero(multik_comm_rates[k]["greedy"])
        summary["by_k"]["K={}".format(k)] = {
            "K": int(k),
            "standalone_one_shot": one_shot,
            "greedy": greedy,
            "greedy_minus_one_shot": {
                "ap_30": greedy["ap_30"] - one_shot["ap_30"],
                "ap_50": greedy["ap_50"] - one_shot["ap_50"],
                "ap_70": greedy["ap_70"] - one_shot["ap_70"],
            },
            "one_shot_greedy_overlap": mean_or_zero(multik_overlaps[k]),
            "greedy_non_positive_marginal_step_ratio": mean_or_zero(non_positive_by_k[k]),
        }

    summary_path = os.path.join(opt.output_dir, "greedy_k_saturation_summary.json")
    diagnostics_path = os.path.join(opt.output_dir, "greedy_k_saturation_diagnostics.pt")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(diagnostics, diagnostics_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

