# -*- coding: utf-8 -*-
"""Generate sparse patch-level counterfactual communication utility labels."""

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.comm_modules.patch_utils import (
    apply_patch_state,
    patch_confidence_stats,
    stratified_sample_patch_ids,
)
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sparse patch-level counterfactual utility labels."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Checkpoint/log directory.")
    parser.add_argument("--eval_epoch", type=str, default=None, help="Checkpoint epoch.")
    parser.add_argument("--output", type=str, required=True, help="Output .pt or .jsonl path.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--patch_h", type=int, default=3)
    parser.add_argument("--patch_w", type=int, default=2)
    parser.add_argument("--bins", type=str, default="2,3,4,7")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save_every", type=int, default=50)
    return parser.parse_args()


def build_loader(hypes, split):
    train = split == "train"
    dataset = build_dataset(hypes, visualize=False, train=train)
    return dataset, DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_train if train else dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )


def save_records(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".jsonl"):
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
    else:
        torch.save(records, path)


def scalar_loss(criterion, output_dict, label_dict):
    return criterion(output_dict, label_dict).detach()


def build_confidence_map(psm_single, target_size):
    confidence = psm_single.sigmoid().max(dim=1, keepdim=True)[0]
    if confidence.shape[-2:] != target_size:
        confidence = F.interpolate(confidence, size=target_size, mode="nearest")
    return confidence


def main():
    opt = parse_args()
    random_gen = random.Random(opt.seed)
    bins = tuple(int(x.strip()) for x in opt.bins.split(",") if x.strip())
    patch_size = (opt.patch_h, opt.patch_w)

    hypes = yaml_utils.load_yaml(None, opt)
    dataset, data_loader = build_loader(hypes, opt.split)

    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _, model = train_utils.load_saved_model(opt.model_dir, model, opt.eval_epoch)
    model.eval()
    criterion = train_utils.create_loss(hypes)

    if not hasattr(model, "_extract_cached_context"):
        raise RuntimeError("This script requires PointPillarWhere2commV2 cache APIs.")

    records = []
    start_time = time.time()
    processed = 0

    for batch_idx, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if opt.max_samples is not None and processed >= opt.max_samples:
            break
        if batch_data is None:
            continue

        batch_data = train_utils.to_device(batch_data, device)
        ego_data = batch_data["ego"]
        sample_id = int(ego_data.get("sample_idx", batch_idx))
        label_dict = ego_data["label_dict"]

        with torch.no_grad():
            cached = model._extract_cached_context(ego_data)
            baseline_output = model._forward_from_cached_context(
                cached, return_communication_debug=True
            )
            base_mask = baseline_output["communication_masks"]
            if base_mask is None:
                raise RuntimeError("Counterfactual label generation requires communication masks.")
            target_size = base_mask.shape[-2:]
            confidence = build_confidence_map(cached["psm_single"], target_size)

            if int(cached["record_len"][0]) <= 1:
                processed += 1
                continue

            neighbor_confidence = confidence[1]
            stats = patch_confidence_stats(neighbor_confidence, patch_size=patch_size)
            sample_patch_ids = stratified_sample_patch_ids(
                stats["mean"], bins=bins, generator=random_gen
            )

            for patch_id in sample_patch_ids:
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
                utility = loss_off - loss_on

                records.append(
                    {
                        "sample_id": sample_id,
                        "batch_idx": batch_idx,
                        "patch_id": int(patch_id),
                        "context_id": "confidence_threshold_base",
                        "utility": float(utility.item()),
                        "loss_off": float(loss_off.item()),
                        "loss_on": float(loss_on.item()),
                        "patch_mean_confidence": float(stats["mean"][patch_id].item()),
                        "patch_max_confidence": float(stats["max"][patch_id].item()),
                        "patch_activation_ratio": float(
                            stats["activation_ratio"][patch_id].item()
                        ),
                        "patch_size": list(patch_size),
                        "feature_size": list(target_size),
                        "communication_source": baseline_output.get(
                            "communication_source", "unknown"
                        ),
                    }
                )

        processed += 1
        if opt.save_every > 0 and processed % opt.save_every == 0:
            save_records(opt.output, records)

    save_records(opt.output, records)
    elapsed = time.time() - start_time
    positives = sum(1 for item in records if item["utility"] > 0)
    negatives = sum(1 for item in records if item["utility"] < 0)
    print(
        "Generated {} labels from {} samples in {:.1f}s | positive={} negative={} zero={}".format(
            len(records), processed, elapsed, positives, negatives, len(records) - positives - negatives
        )
    )


if __name__ == "__main__":
    main()
