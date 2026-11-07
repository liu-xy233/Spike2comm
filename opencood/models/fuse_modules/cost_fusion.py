import torch
import torch.nn as nn

from opencood.models.cost_modules.deformable_detr import DeformableDETR


class CostFusion(nn.Module):
    """CoST fusion wrapper around deformable spatiotemporal attention."""

    def __init__(self, args):
        super().__init__()
        self.max_cav = args["max_cav"]
        self.use_temporal = args.get("multi_time", args.get("use_temporal", False))
        self.fusion_net = DeformableDETR(args)

    def forward(self, regroup_feature, mask, prior_encoding=None, prev_fused_feature=None):
        batch_size = regroup_feature.shape[0]
        if prior_encoding is None:
            prior_encoding = self.build_default_prior_encoding(
                batch_size, regroup_feature.shape[1], regroup_feature.device, regroup_feature.dtype
            )

        if self.use_temporal:
            if prev_fused_feature is None:
                add_mask = torch.zeros_like(mask[:, :1])
                add_feat = torch.zeros_like(regroup_feature[:, :1])
                add_prior = torch.zeros(batch_size, 1, 3, dtype=prior_encoding.dtype, device=prior_encoding.device)
            else:
                add_mask = torch.ones_like(mask[:, :1])
                add_feat = prev_fused_feature.unsqueeze(1)
                add_prior = torch.tensor(
                    [0, 100, 2], dtype=prior_encoding.dtype, device=prior_encoding.device
                ).reshape(1, 1, 3).repeat(batch_size, 1, 1)
            regroup_feature = torch.cat([regroup_feature, add_feat], dim=1)
            mask = torch.cat([mask, add_mask], dim=1)
            prior_encoding = torch.cat([prior_encoding, add_prior], dim=1)

        return self.fusion_net(regroup_feature, mask, prior_encoding)

    @staticmethod
    def build_default_prior_encoding(batch_size, cav_num, device, dtype):
        prior_encoding = torch.zeros(batch_size, cav_num, 3, dtype=dtype, device=device)
        if cav_num > 1:
            prior_encoding[:, 1:, 2] = 1
        return prior_encoding

