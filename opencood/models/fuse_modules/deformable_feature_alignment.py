# -*- coding: utf-8 -*-
"""Mask-aware deformable feature alignment for Where2comm.

The module is intentionally placed after Where2comm communication selection
and ego-coordinate warping. It never changes the communication mask or rate;
the mask is only used to decide which received features may be sampled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableFeatureAlignment(nn.Module):
    def __init__(self, args):
        super(DeformableFeatureAlignment, self).__init__()
        self.in_channels = int(args.get("in_channels", args.get("feature_dim", 256)))
        self.embed_dim = int(args.get("dfa_embed_dim", self.in_channels))
        self.num_levels = int(args.get("dfa_num_levels", 4))
        self.num_points = int(args.get("dfa_num_points", 5))
        self.num_heads = int(args.get("dfa_num_heads", 8))
        self.mask_aware = bool(args.get("dfa_mask_aware", True))
        self.fill_mode = args.get("dfa_fill_mode", "replace")
        self.use_multiscale = bool(args.get("dfa_use_multiscale", True))
        self.valid_threshold = float(args.get("dfa_valid_threshold", 0.5))
        self.max_offset = float(args.get("dfa_max_offset", 8.0))
        # Identity-friendly initialization:
        # initially favor the original-resolution collaborator feature (Level-1).
        self.init_level1_bias = float(args.get("dfa_init_level1_bias", 4.0))

        assert self.num_levels == 4, "DFA expects four sampling levels."
        assert self.fill_mode in ("replace", "residual")
        assert self.in_channels % self.num_heads == 0, (
            "DFA channels {} must be divisible by heads {}.".format(
                self.in_channels, self.num_heads
            )
        )
        self.head_dim = self.in_channels // self.num_heads

        self.ego_proj = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.neighbor_proj = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.naive_fusion = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1),
        )

        self.token_norm = nn.LayerNorm(self.in_channels * 2)
        self.token_proj = nn.Conv2d(self.in_channels * 2, self.embed_dim, kernel_size=1)
        self.offset_pred = nn.Conv2d(
            self.embed_dim,
            self.num_heads * self.num_levels * self.num_points * 2,
            kernel_size=1,
        )
        self.attn_pred = nn.Conv2d(
            self.embed_dim,
            self.num_heads * self.num_levels * self.num_points,
            kernel_size=1,
        )

        self.level2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.value_projs = nn.ModuleList(
            [nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1) for _ in range(4)]
        )
        self.out_proj = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        # ------------------------------------------------------------
        # Identity-friendly DFA initialization
        # ------------------------------------------------------------
        # Start with zero spatial displacement.
        nn.init.zeros_(self.offset_pred.weight)
        nn.init.zeros_(self.offset_pred.bias)

        # Keep attention prediction input-independent at initialization,
        # but strongly favor Level-1 (the original warped collaborator
        # feature) instead of uniformly mixing all four levels.
        nn.init.zeros_(self.attn_pred.weight)
        nn.init.zeros_(self.attn_pred.bias)
        with torch.no_grad():
            attn_bias = self.attn_pred.bias.view(
                self.num_heads,
                self.num_levels,
                self.num_points,
            )
            attn_bias[:, 0, :] = self.init_level1_bias

            # Level-1 value projection should initially preserve the
            # original collaborator feature exactly.
            nn.init.zeros_(self.value_projs[0].weight)
            nn.init.zeros_(self.value_projs[0].bias)
            eye = torch.eye(
                self.in_channels,
                dtype=self.value_projs[0].weight.dtype,
                device=self.value_projs[0].weight.device,
            )
            self.value_projs[0].weight.copy_(
                eye.view(self.in_channels, self.in_channels, 1, 1)
            )

            # Final output projection should also initially be identity.
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            self.out_proj.weight.copy_(
                eye.view(self.in_channels, self.in_channels, 1, 1)
            )

        self.latest_stats = {}
        self.latest_debug_shapes = {}

    @staticmethod
    def _normalize_grid(x_coord, y_coord, height, width):
        if width > 1:
            x_norm = x_coord / (width - 1) * 2 - 1
        else:
            x_norm = torch.zeros_like(x_coord)
        if height > 1:
            y_norm = y_coord / (height - 1) * 2 - 1
        else:
            y_norm = torch.zeros_like(y_coord)
        return torch.stack((x_norm, y_norm), dim=-1)

    @staticmethod
    def _downsample_mask_like_conv(mask, target_size):
        # Match the DFA level convs: kernel=3, stride=2, padding=1. A low-res
        # cell is valid if any communicated feature falls in that receptive field.
        pooled = F.max_pool2d(mask.float(), kernel_size=3, stride=2, padding=1)
        if pooled.shape[-2:] != target_size:
            pooled = F.interpolate(pooled, size=target_size, mode="nearest")
        return pooled

    @staticmethod
    def _sample(feature, grid):
        batch, channels = feature.shape[:2]
        num_heads, num_points, height, width = grid.shape[1:5]
        flat_grid = grid.reshape(batch, num_heads * num_points * height, width, 2)
        sampled = F.grid_sample(
            feature,
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.view(batch, channels, num_heads, num_points, height, width)
        return sampled.permute(0, 2, 3, 1, 4, 5).contiguous()

    def _build_levels(self, neighbor_feature, naive_feature, comm_mask):
        feature_l1 = neighbor_feature
        feature_l2 = self.level2(feature_l1)
        feature_l3 = self.level3(feature_l2)
        feature_l4 = naive_feature
        features = [feature_l1, feature_l2, feature_l3, feature_l4]

        neighbor_valid_masks = [comm_mask]
        level2_mask = self._downsample_mask_like_conv(comm_mask, feature_l2.shape[-2:])
        level3_mask = self._downsample_mask_like_conv(level2_mask, feature_l3.shape[-2:])
        neighbor_valid_masks.extend([level2_mask, level3_mask])
        # The naive level contains ego-neighbor joint context. It is a context
        # source, not evidence that uncommunicated neighbor cells were received.
        naive_valid_mask = torch.ones_like(comm_mask)
        sampling_valid_masks = neighbor_valid_masks + [naive_valid_mask]
        return features, sampling_valid_masks

    def forward(self, ego_feature, neighbor_feature, comm_mask):
        """Align already-warped collaborator features.

        Parameters
        ----------
        ego_feature : torch.Tensor
            Ego feature, shape [1, C, H, W].
        neighbor_feature : torch.Tensor
            Collaborator feature in ego coordinates, shape [N, C, H, W].
        comm_mask : torch.Tensor
            Warped communication mask, shape [N, 1, H, W].
        """
        num_agents, channels, height, width = neighbor_feature.shape
        assert channels == self.in_channels
        if num_agents <= 1:
            self.latest_stats = self._empty_stats(neighbor_feature)
            return neighbor_feature

        ego_repeat = ego_feature.expand(num_agents, -1, -1, -1)
        naive_feature = self.naive_fusion(
            torch.cat((self.ego_proj(ego_repeat), self.neighbor_proj(neighbor_feature)), dim=1)
        )

        token = torch.cat((naive_feature, neighbor_feature), dim=1)
        token = token.permute(0, 2, 3, 1)
        token = self.token_norm(token).permute(0, 3, 1, 2).contiguous()
        token = self.token_proj(token)

        offsets = self.offset_pred(token)
        offsets = offsets.view(
            num_agents,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
            height,
            width,
        )
        offsets = torch.tanh(offsets) * self.max_offset

        attn_logits = self.attn_pred(token)
        attn_logits = attn_logits.view(
            num_agents, self.num_heads, self.num_levels, self.num_points, height, width
        )

        levels, masks = self._build_levels(neighbor_feature, naive_feature, comm_mask)
        y_base, x_base = torch.meshgrid(
            torch.arange(height, device=neighbor_feature.device, dtype=neighbor_feature.dtype),
            torch.arange(width, device=neighbor_feature.device, dtype=neighbor_feature.dtype),
            indexing="ij",
        )

        sampled_features = []
        sampled_valids = []
        for level_idx, (level_feature, level_mask) in enumerate(zip(levels, masks)):
            level_h, level_w = level_feature.shape[-2:]
            scale_y = float(level_h) / float(height)
            scale_x = float(level_w) / float(width)
            level_offsets = offsets[:, :, level_idx]
            x_coord = (x_base.view(1, 1, 1, height, width) + level_offsets[:, :, :, 0]) * scale_x
            y_coord = (y_base.view(1, 1, 1, height, width) + level_offsets[:, :, :, 1]) * scale_y
            grid = self._normalize_grid(x_coord, y_coord, level_h, level_w)

            value = self.value_projs[level_idx](level_feature)
            value = self._sample(value, grid)
            value = value.view(
                num_agents,
                self.num_heads,
                self.num_points,
                self.num_heads,
                self.head_dim,
                height,
                width,
            )
            value = value.diagonal(dim1=1, dim2=3).permute(0, 5, 1, 2, 3, 4)

            valid = self._sample(level_mask, grid).squeeze(3)
            sampling_valid_mask = valid > self.valid_threshold
            sampled_features.append(value)
            sampled_valids.append(sampling_valid_mask)

        sampled_features = torch.stack(sampled_features, dim=2)
        sampled_valids = torch.stack(sampled_valids, dim=2)
        valid_float = sampled_valids.to(dtype=neighbor_feature.dtype)

        if self.mask_aware:
            masked_logits = attn_logits.masked_fill(~sampled_valids, -1e4)
            all_invalid = sampled_valids.flatten(1, 3).sum(dim=1, keepdim=True) == 0
            masked_logits = torch.where(
                all_invalid.unsqueeze(2).unsqueeze(3),
                torch.zeros_like(masked_logits),
                masked_logits,
            )
            attn = F.softmax(masked_logits.flatten(2, 3), dim=2).view_as(attn_logits)
            attn = attn * valid_float
            denom = attn.flatten(2, 3).sum(dim=2, keepdim=True).clamp_min(1e-6)
            attn = attn / denom.view(num_agents, self.num_heads, 1, 1, height, width)
        else:
            attn = F.softmax(attn_logits.flatten(2, 3), dim=2).view_as(attn_logits)

        aligned_heads = (sampled_features * attn.unsqueeze(4)).sum(dim=(2, 3))
        aligned = aligned_heads.reshape(num_agents, channels, height, width)
        aligned = self.out_proj(aligned)

        dfa_query_mask = comm_mask > self.valid_threshold
        all_invalid_query = sampled_valids.flatten(1, 3).sum(dim=1, keepdim=True) == 0
        fallback_mask = all_invalid_query | (~dfa_query_mask)
        if self.fill_mode == "replace":
            output = torch.where(fallback_mask.expand_as(aligned), neighbor_feature, aligned)
        else:
            output = torch.where(
                fallback_mask.expand_as(aligned), neighbor_feature, neighbor_feature + aligned
            )

        # The ego branch participates in the original fusion unchanged.
        output = output.clone()
        output[0:1] = neighbor_feature[0:1]
        self._update_stats(offsets, sampled_valids, dfa_query_mask, output, neighbor_feature, masks)
        self.latest_debug_shapes = {
            "F_e": tuple(ego_feature.shape),
            "F_tilde": tuple(neighbor_feature.shape),
            "M_comm": tuple(comm_mask.shape),
            "P_i": int(dfa_query_mask[1:].sum().detach().item()) if num_agents > 1 else 0,
            "F_naive": tuple(naive_feature.shape),
            "scales": [tuple(level.shape) for level in levels[:3]],
            "offset": tuple(offsets.shape),
            "attention": tuple(attn.shape),
            "valid_sampling_mask": tuple(sampled_valids.shape),
            "F_aligned": tuple(output.shape),
        }
        return output

    def _empty_stats(self, feature):
        zero = feature.new_tensor(0.0)
        return {
            "total_queries": zero,
            "valid_queries": zero,
            "zero_valid_queries": zero,
            "avg_valid_samples_per_query": zero,
            "valid_sampling_ratio": zero,
            "zero_valid_query_ratio": zero,
            "mean_offset_magnitude": zero,
            "median_offset_magnitude": zero,
            "mean_feature_delta": zero,
            "level1_valid_ratio": zero,
            "level2_valid_ratio": zero,
            "level3_valid_ratio": zero,
            "naive_valid_ratio": zero,
            "level1_avg_valid_samples_per_query": zero,
            "level2_avg_valid_samples_per_query": zero,
            "level3_avg_valid_samples_per_query": zero,
            "naive_avg_valid_samples_per_query": zero,
            "level_avg_valid_samples_sum": zero,
            "level_valid_samples_consistency_error": zero,
        }

    def _update_stats(self, offsets, valid_mask, query_mask, output, original, level_masks):
        with torch.no_grad():
            collaborator_query = query_mask.clone()
            collaborator_query[0:1] = False
            total_queries = collaborator_query.sum().to(dtype=output.dtype)
            if total_queries.item() == 0:
                self.latest_stats = self._empty_stats(output)
                return

            valid_per_query = valid_mask.flatten(1, 3).sum(dim=1)
            query_valid_counts = valid_per_query[collaborator_query.squeeze(1)]
            valid_queries = (query_valid_counts > 0).sum().to(dtype=output.dtype)
            zero_valid_queries = (query_valid_counts == 0).sum().to(dtype=output.dtype)

            # Per-level valid sampling count on collaborator DFA queries only.
            # valid_mask: [N, num_heads, num_levels, num_points, H, W]
            # Each level has at most num_heads * num_points valid samples/query.
            level_avg_valid_samples = []
            for level_idx in range(valid_mask.shape[2]):
                level_valid_per_query = valid_mask[:, :, level_idx].sum(dim=(1, 2))
                level_query_valid_counts = level_valid_per_query[
                    collaborator_query.squeeze(1)
                ]
                level_avg_valid_samples.append(
                    level_query_valid_counts.float().mean()
                )

            if len(level_avg_valid_samples) != 4:
                raise RuntimeError(
                    "Expected 4 DFA sampling levels, got {}".format(
                        len(level_avg_valid_samples)
                    )
                )

            level_avg_valid_samples_sum = torch.stack(
                level_avg_valid_samples
            ).sum()

            avg_valid_samples_per_query = query_valid_counts.float().mean()

            level_valid_samples_consistency_error = torch.abs(
                level_avg_valid_samples_sum - avg_valid_samples_per_query
            )

            offset_mag = torch.linalg.vector_norm(offsets, dim=4)
            query_offsets = offset_mag.permute(0, 4, 5, 1, 2, 3)[
                collaborator_query.squeeze(1)
            ]
            delta = torch.linalg.vector_norm(output - original, dim=1, keepdim=True)
            query_delta = delta[collaborator_query]

            self.latest_stats = {
                "total_queries": total_queries,
                "valid_queries": valid_queries,
                "zero_valid_queries": zero_valid_queries,
                "avg_valid_samples_per_query": avg_valid_samples_per_query,
                "valid_sampling_ratio": valid_mask.float().mean(),
                "level1_avg_valid_samples_per_query": level_avg_valid_samples[0],
                "level2_avg_valid_samples_per_query": level_avg_valid_samples[1],
                "level3_avg_valid_samples_per_query": level_avg_valid_samples[2],
                "naive_avg_valid_samples_per_query": level_avg_valid_samples[3],
                "level_avg_valid_samples_sum": level_avg_valid_samples_sum,
                "level_valid_samples_consistency_error": level_valid_samples_consistency_error,
                "zero_valid_query_ratio": zero_valid_queries / total_queries.clamp_min(1.0),
                "mean_offset_magnitude": query_offsets.float().mean(),
                "median_offset_magnitude": query_offsets.float().median(),
                "mean_feature_delta": query_delta.float().mean(),
                "level1_valid_ratio": (level_masks[0] > self.valid_threshold).float().mean(),
                "level2_valid_ratio": (level_masks[1] > self.valid_threshold).float().mean(),
                "level3_valid_ratio": (level_masks[2] > self.valid_threshold).float().mean(),
                "naive_valid_ratio": (level_masks[3] > self.valid_threshold).float().mean(),
            }
