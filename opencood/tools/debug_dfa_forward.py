# -*- coding: utf-8 -*-
"""Minimal DFA forward/backward smoke test.

Run inside the same Python environment used for OpenCOOD training:
python opencood/tools/debug_dfa_forward.py
"""

import torch

from opencood.models.fuse_modules.deformable_feature_alignment import (
    DeformableFeatureAlignment,
)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = DeformableFeatureAlignment(
        {
            "in_channels": 256,
            "dfa_num_levels": 4,
            "dfa_num_points": 5,
            "dfa_num_heads": 8,
            "dfa_mask_aware": True,
            "dfa_query_source": "where2comm_mask",
            "dfa_fill_mode": "replace",
            "dfa_use_multiscale": True,
        }
    ).to(device)

    neighbor_feature = torch.randn(2, 256, 32, 48, device=device, requires_grad=True)
    ego_feature = neighbor_feature[0:1]
    comm_mask = torch.zeros(2, 1, 32, 48, device=device)
    comm_mask[0] = 1
    comm_mask[1, :, 8:24, 12:36] = 1

    aligned = module(ego_feature, neighbor_feature, comm_mask)
    loss = aligned.square().mean()
    loss.backward()

    print("DFA debug shapes:")
    for key, value in module.latest_debug_shapes.items():
        print("  {}: {}".format(key, value))
    print("DFA stats:")
    for key, value in module.latest_stats.items():
        print("  {}: {:.6f}".format(key, float(value.detach().cpu())))
    print("checks:")
    print("  F_aligned equals F_tilde shape:", aligned.shape == neighbor_feature.shape)
    print("  aligned finite:", torch.isfinite(aligned).all().item())
    print("  grad finite:", torch.isfinite(neighbor_feature.grad).all().item())


if __name__ == "__main__":
    main()
