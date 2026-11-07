# -*- coding: utf-8 -*-
"""Real DAIR-V2X single-batch DFA smoke test.

This script performs one in-memory forward/loss/backward/optimizer.step audit.
It does not save checkpoints and is not a training launcher.
"""

import argparse
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(description="DFA real-batch smoke audit")
    parser.add_argument("--hypes_yaml", "-y", required=True)
    parser.add_argument("--pretrained_path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def tensor_norm(tensor):
    if tensor is None:
        return None
    return float(torch.linalg.vector_norm(tensor.detach()).cpu())


def first_named(named_items, contains):
    for name, value in named_items:
        if contains in name:
            return name, value
    return None, None


def clone_param(model, contains):
    name, param = first_named(model.named_parameters(), contains)
    if param is None:
        return name, None
    return name, param.detach().clone()


def max_abs_delta(before, after):
    if before is None or after is None:
        return None
    return float((after.detach() - before).abs().max().cpu())


def clone_bn_buffers(model):
    buffers = {}
    for name, buffer in model.named_buffers():
        if name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked"):
            buffers[name] = buffer.detach().clone()
    return buffers


def max_changed_bn(before, model):
    changed = []
    for name, old in before.items():
        current = dict(model.named_buffers())[name]
        delta = max_abs_delta(old, current)
        if delta and delta > 0:
            changed.append((name, delta))
    return changed


def grad_norms(model):
    groups = {
        "dfa": [],
        "offset_pred": [],
        "attn_pred": [],
        "backbone": [],
        "fusion": [],
        "head": [],
    }
    for name, param in model.named_parameters():
        norm = tensor_norm(param.grad)
        if "fusion_net.dfa" in name:
            groups["dfa"].append((name, norm))
        if "fusion_net.dfa.offset_pred" in name:
            groups["offset_pred"].append((name, norm))
        if "fusion_net.dfa.attn_pred" in name:
            groups["attn_pred"].append((name, norm))
        if name.startswith("backbone"):
            groups["backbone"].append((name, norm))
        if name.startswith("fusion_net") and "fusion_net.dfa" not in name:
            groups["fusion"].append((name, norm))
        if name.startswith("cls_head") or name.startswith("reg_head"):
            groups["head"].append((name, norm))
    return groups


def print_grad_summary(groups):
    for group_name in ["dfa", "offset_pred", "attn_pred"]:
        non_null = [(n, g) for n, g in groups[group_name] if g is not None]
        total = sum(g for _, g in non_null)
        print("{} grad tensors: {} total_norm: {:.8f}".format(group_name, len(non_null), total))
        for name, value in non_null[:8]:
            print("  {}: {:.8f}".format(name, value))
    for group_name in ["backbone", "fusion", "head"]:
        has_grad = [(n, g) for n, g in groups[group_name] if g is not None]
        print("{} frozen grad tensors with grad: {}".format(group_name, len(has_grad)))
        for name, value in has_grad[:8]:
            print("  {}: {}".format(name, value))


def main():
    args = parse_args()
    hypes = yaml_utils.load_yaml(args.hypes_yaml)
    if args.batch_size is not None:
        hypes["train_params"]["batch_size"] = args.batch_size

    print("Config source: {}".format(args.hypes_yaml))
    print("Dataset Building")
    dataset = build_dataset(hypes, visualize=False, train=True)
    loader = DataLoader(
        dataset,
        batch_size=hypes["train_params"]["batch_size"],
        num_workers=0,
        collate_fn=dataset.collate_batch_train,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    print("Creating Model")
    model = train_utils.create_model(hypes)
    device = torch.device(args.device)
    model.to(device)
    if args.pretrained_path:
        model = train_utils.load_pretrained_model(
            args.pretrained_path,
            model,
            allowed_missing_prefixes=("fusion_net.dfa.",),
            allow_unexpected=False,
        )
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total params: {}".format(total_params))
    print("trainable params: {}".format(trainable_params))
    print("trainable parameter names:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("  {}".format(name))

    batch_data = next(iter(loader))
    batch_data = train_utils.to_device(batch_data, device)
    batch_data["ego"]["epoch"] = 0
    batch_data["ego"]["return_communication_debug"] = True

    model.train()
    cached_context = model._extract_cached_context(batch_data["ego"])
    print("local_features.shape: {}".format(tuple(cached_context["local_features"].shape)))
    print("record_len: {}".format(cached_context["record_len"].detach().cpu().tolist()))

    old_use_dfa = model.fusion_net.use_dfa
    with torch.no_grad():
        model.fusion_net.use_dfa = False
        baseline_output = model._forward_from_cached_context(
            cached_context,
            return_communication_debug=True,
        )
        baseline_comm_rate = baseline_output["comm_rate"].detach().clone()

        # Compute the baseline loss using exactly the same criterion and
        # auxiliary single-agent losses as the DFA path.
        baseline_loss = criterion(
            baseline_output,
            batch_data["ego"]["label_dict"],
        )
        if len(baseline_output) > 2 and "label_dict_single_v" in batch_data["ego"]:
            baseline_loss = baseline_loss + criterion(
                baseline_output,
                batch_data["ego"]["label_dict_single_v"],
                prefix="_single_v",
            )
            baseline_loss = baseline_loss + criterion(
                baseline_output,
                batch_data["ego"]["label_dict_single_i"],
                prefix="_single_i",
            )

        baseline_loss = baseline_loss.detach().clone()
        model.fusion_net.use_dfa = old_use_dfa

    optimizer.zero_grad()
    output_dict = model._forward_from_cached_context(
        cached_context,
        return_communication_debug=True,
    )
    final_loss = criterion(output_dict, batch_data["ego"]["label_dict"])
    if len(output_dict) > 2 and "label_dict_single_v" in batch_data["ego"]:
        final_loss = final_loss + criterion(output_dict, batch_data["ego"]["label_dict_single_v"], prefix="_single_v")
        final_loss = final_loss + criterion(output_dict, batch_data["ego"]["label_dict_single_i"], prefix="_single_i")

    print("communication mask shape: {}".format(tuple(output_dict["communication_masks"].shape)))
    print("communication source: {}".format(output_dict["communication_source"]))
    print("baseline comm_rate: {}".format(baseline_comm_rate.detach().cpu().tolist()))
    print("dfa comm_rate: {}".format(output_dict["comm_rate"].detach().cpu().tolist()))
    print("comm_rate max abs delta: {:.10f}".format(max_abs_delta(baseline_comm_rate, output_dict["comm_rate"])))
    print("communication debug stats:")
    for item in output_dict.get("communication_debug_stats", []):
        print("  {}".format(item))
    print("DFA debug shapes:")
    for item in output_dict.get("dfa_debug_shapes", []):
        print("  {}".format(item))
    print("DFA stats:")
    for key, value in output_dict.get("dfa_stats", {}).items():
        print("  {}: {:.8f}".format(key, float(value.detach().cpu())))
    print("fusion output shape: {}".format(tuple(output_dict["psm"].shape)))
    print("baseline loss: {:.8f}".format(float(baseline_loss.detach().cpu())))
    print("dfa init loss: {:.8f}".format(float(final_loss.detach().cpu())))
    print(
        "dfa init loss delta: {:+.8f}".format(
            float((final_loss.detach() - baseline_loss).cpu())
        )
    )
    print(
        "dfa init loss relative delta: {:+.4f}%".format(
            float(
                (
                    (final_loss.detach() - baseline_loss)
                    / baseline_loss.clamp_min(1e-12)
                    * 100.0
                ).cpu()
            )
        )
    )

    tracked = {
        "dfa_offset": clone_param(model, "fusion_net.dfa.offset_pred.weight"),
        "dfa_attn": clone_param(model, "fusion_net.dfa.attn_pred.weight"),
        "dfa_out": clone_param(model, "fusion_net.dfa.out_proj.weight"),
        "e3dsnn": clone_param(model, "backbone"),
        "communication_selector": clone_param(model, "fusion_net.naive_communication.gaussian_filter.weight"),
        "head": clone_param(model, "cls_head.weight"),
    }
    bn_before = clone_bn_buffers(model)

    final_loss.backward()
    groups = grad_norms(model)
    print_grad_summary(groups)
    optimizer.step()

    print("optimizer step parameter deltas:")
    for label, (name, before) in tracked.items():
        if name is None:
            print("  {}: parameter not found".format(label))
            continue
        current = dict(model.named_parameters())[name]
        print("  {} {} max_abs_delta: {}".format(label, name, max_abs_delta(before, current)))

    bn_changed = max_changed_bn(bn_before, model)
    print("BN running-stat changed buffers: {}".format(len(bn_changed)))
    for name, delta in bn_changed[:20]:
        print("  {}: {}".format(name, delta))


if __name__ == "__main__":
    main()
