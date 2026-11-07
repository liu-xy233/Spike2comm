# -*- coding: utf-8 -*-
"""Compare one-shot and greedy patch counterfactual Oracle selection."""

import argparse
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.comm_modules.patch_utils import communication_rate_from_mask
from opencood.tools import train_utils
from opencood.utils import eval_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare standalone one-shot and greedy marginal patch Oracle selection."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--eval_epoch", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--patch_h", type=int, default=2)
    parser.add_argument("--patch_w", type=int, default=3)
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument(
        "--reg_weight",
        type=float,
        default=1.0,
        help="Extra regression weight used only for counterfactual utility."
    )
    return parser.parse_args()


def should_stop(processed, max_samples):
    return max_samples is not None and max_samples > 0 and processed >= max_samples


def new_result_stat():
    return {
        0.3: {"tp": [], "fp": [], "gt": 0},
        0.5: {"tp": [], "fp": [], "gt": 0},
        0.7: {"tp": [], "fp": [], "gt": 0},
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


def scalar_loss(criterion, output_dict, label_dict, reg_weight=1.0):
    # Run the original detection criterion so that PointPillarLoss
    # updates criterion.loss_dict.
    criterion(output_dict, label_dict)

    conf_loss = criterion.loss_dict["conf_loss"].detach()
    reg_loss = criterion.loss_dict["reg_loss"].detach()

    # reg_loss already contains the original reg_coe from PointPillarLoss.
    # reg_weight is an additional utility-only localization weight.
    weighted_loss = conf_loss + reg_weight * reg_loss
    return weighted_loss


def calculate_ap_values(result_stat):
    return {
        "ap_30": eval_utils.calculate_ap(result_stat, 0.3)[0],
        "ap_50": eval_utils.calculate_ap(result_stat, 0.5)[0],
        "ap_70": eval_utils.calculate_ap(result_stat, 0.7)[0],
    }


def patch_slices(height: int, width: int, patch_size: Tuple[int, int]) -> List[Tuple[slice, slice]]:
    patch_h, patch_w = patch_size
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch_h and patch_w must be positive.")
    slices = []
    for h_start in range(0, height, patch_h):
        h_end = min(h_start + patch_h, height)
        for w_start in range(0, width, patch_w):
            w_end = min(w_start + patch_w, width)
            slices.append((slice(h_start, h_end), slice(w_start, w_end)))
    return slices


def make_empty_neighbor_mask(base_mask: torch.Tensor) -> torch.Tensor:
    if base_mask.dim() != 4:
        raise ValueError("Expected flattened mask [N,1,H,W], got {}.".format(base_mask.shape))
    mask = base_mask.new_zeros(base_mask.shape)
    mask[0] = 1
    return mask


def mask_with_selected_patches(
    empty_neighbor_mask: torch.Tensor,
    selected_patch_ids: Sequence[int],
    slices: Sequence[Tuple[slice, slice]],
    neighbor_index: int = 1,
) -> torch.Tensor:
    mask = empty_neighbor_mask.clone()
    for patch_id in selected_patch_ids:
        h_slice, w_slice = slices[int(patch_id)]
        mask[neighbor_index, :, h_slice, w_slice] = 1
    return mask


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def main():
    opt = parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)
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

    result_stats = {
        "ego_only": new_result_stat(),
        "original": new_result_stat(),
        "standalone_one_shot": new_result_stat(),
        "greedy_topk": new_result_stat(),
    }
    comm_rates = {name: [] for name in result_stats}
    diagnostics = []
    overlap_ratios = []
    non_positive_steps = 0
    total_greedy_steps = 0
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
            if int(cached["record_len"][0]) != 2:
                raise RuntimeError("This experiment expects DAIR-V2X two-agent batch=1 data.")

            original_output = model._forward_from_cached_context(
                cached, return_communication_debug=True
            )
            base_mask = original_output["communication_masks"]
            if base_mask is None:
                raise RuntimeError("Greedy Oracle evaluation requires communication masks.")

            height, width = base_mask.shape[-2:]
            slices = patch_slices(height, width, patch_size)
            num_patches = len(slices)
            k = min(opt.topk, num_patches)

            empty_neighbor_mask = make_empty_neighbor_mask(base_mask)
            empty_output = model._forward_from_cached_context(
                cached, override_communication_mask=empty_neighbor_mask
            )
            loss_empty = scalar_loss(criterion, empty_output, label_dict, opt.reg_weight)

            one_shot_utilities = []
            one_shot_losses = []
            for patch_id in range(num_patches):
                only_patch_mask = mask_with_selected_patches(
                    empty_neighbor_mask, [patch_id], slices
                )
                output_only_patch = model._forward_from_cached_context(
                    cached, override_communication_mask=only_patch_mask
                )
                loss_only_patch = scalar_loss(criterion, output_only_patch, label_dict, opt.reg_weight)
                one_shot_losses.append(loss_only_patch)
                one_shot_utilities.append(loss_empty - loss_only_patch)

            one_shot_utilities_tensor = torch.stack(one_shot_utilities)
            one_shot_patch_ids = (
                torch.topk(one_shot_utilities_tensor, k=k, largest=True).indices.cpu().tolist()
                if k > 0
                else []
            )
            one_shot_mask = mask_with_selected_patches(
                empty_neighbor_mask, one_shot_patch_ids, slices
            )
            one_shot_output = model._forward_from_cached_context(
                cached, override_communication_mask=one_shot_mask
            )

            selected = []
            remaining = set(range(num_patches))
            current_loss = loss_empty
            greedy_round_patch_ids = []
            greedy_round_best_utilities = []
            greedy_round_current_losses = [float(current_loss.item())]

            for _ in range(k):
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
                    candidate_loss = scalar_loss(criterion, candidate_output, label_dict, opt.reg_weight)
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
                greedy_round_patch_ids.append(int(best_patch_id))
                greedy_round_best_utilities.append(float(best_marginal_utility.item()))
                greedy_round_current_losses.append(float(current_loss.item()))
                total_greedy_steps += 1
                if best_marginal_utility.item() <= 0:
                    non_positive_steps += 1

            greedy_patch_ids = [int(x) for x in selected]
            greedy_mask = mask_with_selected_patches(
                empty_neighbor_mask, greedy_patch_ids, slices
            )
            greedy_output = model._forward_from_cached_context(
                cached, override_communication_mask=greedy_mask
            )

            update_detection_stats(dataset, batch_data, empty_output, result_stats["ego_only"])
            update_detection_stats(dataset, batch_data, original_output, result_stats["original"])
            update_detection_stats(
                dataset, batch_data, one_shot_output, result_stats["standalone_one_shot"]
            )
            update_detection_stats(dataset, batch_data, greedy_output, result_stats["greedy_topk"])

            comm_rates["ego_only"].append(
                float(communication_rate_from_mask(empty_neighbor_mask, cached["record_len"]).item())
            )
            comm_rates["original"].append(
                float(communication_rate_from_mask(base_mask, cached["record_len"]).item())
            )
            comm_rates["standalone_one_shot"].append(
                float(communication_rate_from_mask(one_shot_mask, cached["record_len"]).item())
            )
            comm_rates["greedy_topk"].append(
                float(communication_rate_from_mask(greedy_mask, cached["record_len"]).item())
            )

            overlap_count = len(set(one_shot_patch_ids).intersection(greedy_patch_ids))
            overlap_ratio = overlap_count / k if k > 0 else 0.0
            overlap_ratios.append(overlap_ratio)

            diagnostics.append(
                {
                    "sample_id": sample_id,
                    "batch_idx": batch_idx,
                    "patch_size": list(patch_size),
                    "feature_size": [int(height), int(width)],
                    "num_patches": int(num_patches),
                    "topk": int(k),
                    "loss_empty": float(loss_empty.item()),
                    "one_shot_patch_ids": [int(x) for x in one_shot_patch_ids],
                    "greedy_patch_ids": greedy_patch_ids,
                    "one_shot_utilities": [
                        float(utility.item()) for utility in one_shot_utilities_tensor
                    ],
                    "greedy_round_selected_patch_ids": greedy_round_patch_ids,
                    "greedy_round_best_marginal_utilities": greedy_round_best_utilities,
                    "greedy_round_current_losses": greedy_round_current_losses,
                    "one_shot_greedy_overlap_count": int(overlap_count),
                    "one_shot_greedy_overlap_ratio": float(overlap_ratio),
                    "communication_rate": {
                        "ego_only": comm_rates["ego_only"][-1],
                        "original": comm_rates["original"][-1],
                        "standalone_one_shot": comm_rates["standalone_one_shot"][-1],
                        "greedy_topk": comm_rates["greedy_topk"][-1],
                    },
                }
            )

        processed += 1

    summary: Dict[str, object] = {
        "epoch": epoch_id,
        "samples": processed,
        "K": int(opt.topk),
        "patch_size": [int(opt.patch_h), int(opt.patch_w)],
        "mean_one_shot_greedy_overlap_at_K": mean_or_zero(overlap_ratios),
        "greedy_non_positive_selected_steps": int(non_positive_steps),
        "greedy_non_positive_selected_step_ratio": (
            non_positive_steps / total_greedy_steps if total_greedy_steps > 0 else 0.0
        ),
    }

    for method_name, stat in result_stats.items():
        method_summary = calculate_ap_values(stat)
        method_summary["comm_rate"] = mean_or_zero(comm_rates[method_name])
        summary[method_name] = method_summary

    with open(os.path.join(opt.output_dir, "greedy_oracle_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(opt.output_dir, "greedy_oracle_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    torch.save(diagnostics, os.path.join(opt.output_dir, "greedy_oracle_diagnostics.pt"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

