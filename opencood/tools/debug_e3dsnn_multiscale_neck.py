# -*- coding: utf-8 -*-
"""Real-batch smoke test for the E-3DSNN multi-scale enhancement neck."""

import argparse

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="E-3DSNN multi-scale enhancement real-batch smoke test"
    )
    parser.add_argument("--hypes_yaml", "-y", required=True)
    parser.add_argument("--pretrained_path", default="")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Non-zero residual scale used only by this smoke test.",
    )
    return parser.parse_args()


def require_finite(name, tensor):
    if tensor is None or not torch.is_tensor(tensor):
        raise RuntimeError("{} is not a tensor.".format(name))
    if not torch.isfinite(tensor).all():
        raise RuntimeError("{} contains NaN or Inf.".format(name))


def audit_spike_blocks(neck):
    """Smoke-test-only hooks: validate and print each main/shortcut branch."""
    handles = []
    block_names = ("downsample1", "downsample2", "output_conv")

    def branch_hook(label, quantized=False):
        def hook(module, inputs, output):
            require_finite(label, output)
            print("{}: {} -> {}".format(label, tuple(inputs[0].shape), tuple(output.shape)))
            if quantized:
                if not ((output >= 0) & (output <= 1) & (output * 4 == (output * 4).round())).all():
                    raise RuntimeError("Invalid four-level spike output at " + label)
                print("{} nonzero ratio: {:.6f}".format(
                    label, float((output != 0).float().mean().detach().cpu())
                ))
        return hook

    for name in block_names:
        block = getattr(neck, name)
        handles.append(block.spike.register_forward_hook(branch_hook(name + ".spike", True)))
        handles.append(block.bn.register_forward_hook(branch_hook(name + ".main_bn")))
        handles.append(block.shortcut.register_forward_hook(branch_hook(name + ".shortcut")))
        handles.append(block.register_forward_hook(branch_hook(name + ".residual_sum")))
    return handles


def main():
    args = parse_args()
    hypes = yaml_utils.load_yaml(args.hypes_yaml)
    hypes["train_params"]["batch_size"] = args.batch_size
    neck_cfg = hypes["model"]["args"]["base_bev_backbone"].setdefault(
        "multi_scale_enhancement", {}
    )
    neck_cfg["enabled"] = True
    neck_cfg["debug_shapes"] = True
    neck_cfg["alpha_init"] = args.alpha

    print("Config source: {}".format(args.hypes_yaml))
    dataset = build_dataset(hypes, visualize=False, train=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=dataset.collate_batch_train,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    device = torch.device(args.device)
    model = train_utils.create_model(hypes).to(device)
    if args.pretrained_path:
        model = train_utils.load_pretrained_model(
            args.pretrained_path,
            model,
            allowed_missing_prefixes=(
                "fusion_net.dfa.",
                "backbone.multi_scale_neck.",
            ),
            allow_unexpected=False,
        )

    # A loaded checkpoint can overwrite alpha_init with zero. Exercise all
    # branch weights in this diagnostic run without changing training config.
    neck = model.backbone.multi_scale_neck
    if args.alpha == 0:
        raise ValueError("The gradient smoke test requires a nonzero --alpha.")
    with torch.no_grad():
        neck.alpha.fill_(args.alpha)
    handles = audit_spike_blocks(neck)

    batch_data = train_utils.to_device(next(iter(loader)), device)
    batch_data["ego"]["return_communication_debug"] = True

    model.train()
    try:
        cached_context = model._extract_cached_context(batch_data["ego"])
    finally:
        for handle in handles:
            handle.remove()
    debug = cached_context.get("multi_scale_neck_debug", {})
    if not debug:
        raise RuntimeError("The multi-scale neck did not return debug shapes.")

    labels = [
        ("Stage2 sparse shape", "stage2_sparse"),
        ("Stage3 sparse shape", "stage3_sparse"),
        ("Stage4 sparse shape", "stage4_sparse"),
        ("Stage2 active features shape", "stage2_active_features"),
        ("Stage3 active features shape", "stage3_active_features"),
        ("Stage4 active features shape", "stage4_active_features"),
        ("Stage2 BEV shape", "stage2_bev"),
        ("Stage3 BEV shape", "stage3_bev"),
        ("Stage4 BEV shape", "stage4_bev"),
        ("L2 shape", "l2"),
        ("L3 shape", "l3"),
        ("L4 shape", "l4"),
        ("P3 shape", "p3"),
        ("P2 shape", "p2"),
        ("P_down1 shape", "p_down1"),
        ("P_down2 shape", "p_down2"),
        ("F_ms shape", "f_ms"),
        ("original_final_feature shape", "original_final_feature"),
        ("F_enhanced shape", "f_enhanced"),
    ]
    for label, key in labels:
        print("{}: {}".format(label, debug[key]))
    print("alpha: {}".format(debug["alpha"]))

    if debug["f_ms"] != debug["original_final_feature"]:
        raise RuntimeError("F_ms does not match original_final_feature.")
    if debug["f_enhanced"] != debug["original_final_feature"]:
        raise RuntimeError("F_enhanced does not match original_final_feature.")

    output_dict = model._forward_from_cached_context(
        cached_context, return_communication_debug=True
    )
    for name in ("psm", "rm", "psm_single", "rm_single"):
        require_finite(name, output_dict[name])
        print("{} shape: {}".format(name, tuple(output_dict[name].shape)))
    require_finite("comm_rate", output_dict["comm_rate"])
    print("communication mask shape: {}".format(
        tuple(output_dict["communication_masks"].shape)
    ))
    print("communication source: {}".format(output_dict["communication_source"]))
    print("communication rate: {}".format(output_dict["comm_rate"].detach().cpu()))

    criterion = train_utils.create_loss(hypes)
    loss = criterion(output_dict, batch_data["ego"]["label_dict"])
    require_finite("loss", loss)
    model.zero_grad()
    loss.backward()
    neck_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.multi_scale_neck.")
        and parameter.requires_grad
        and parameter.grad is not None
    ]
    if not neck_gradients:
        raise RuntimeError("No gradient reached the multi-scale neck.")
    for index, gradient in enumerate(neck_gradients):
        require_finite("neck_gradient_{}".format(index), gradient)
    for block_name in ("downsample1", "downsample2", "output_conv"):
        block = getattr(neck, block_name)
        for branch, weight in (("main", block.conv.weight), ("shortcut", block.shortcut[0].weight)):
            label = block_name + "." + branch
            require_finite(label + ".grad", weight.grad)
            grad_norm = float(weight.grad.abs().sum().detach().cpu())
            if grad_norm == 0:
                raise RuntimeError("Zero gradient in " + label)
            print("{} gradient L1: {:.8f}".format(label, grad_norm))
    print("loss: {:.8f}".format(float(loss.detach().cpu())))
    print("neck gradient tensors: {}".format(len(neck_gradients)))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
