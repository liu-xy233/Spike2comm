# -*- coding: utf-8 -*-
"""CoST feature selection/fusion with the existing E-3DSNN local backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.comm_modules.cost_compress_fuse import CostCompressFuse
from opencood.models.cost_modules.alignment import align_agents_to_ego, align_features
from opencood.models.fuse_modules.cost_fusion import CostFusion
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.e3dsnn_spconv_backbone import (
    E3DSNNSparseBEVBackbone,
)
from opencood.models.sub_modules.naive_compress import NaiveCompressor


class PointPillarCostE3dsnn(nn.Module):
    """CoST path that keeps the current E-3DSNN voxel/backbone stack intact."""

    def __init__(self, args):
        super(PointPillarCostE3dsnn, self).__init__()
        self.max_cav = args["max_cav"]
        self.backbone = E3DSNNSparseBEVBackbone(
            args["base_bev_backbone"],
            input_channels=args.get("input_channels", 4),
            grid_size=args["grid_size"],
        )

        self.shrink_flag = "shrink_header" in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args["shrink_header"])

        self.compression = False
        if args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])

        self.dcn = False
        if "dcn" in args:
            from opencood.models.sub_modules.dcn_net import DCNNet

            self.dcn = True
            self.dcn_net = DCNNet(args["dcn"])

        self.head_in_channels = args.get("head_in_channels", 256)
        self.cls_head = nn.Conv2d(self.head_in_channels, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(
            self.head_in_channels, 7 * args["anchor_number"], kernel_size=1
        )

        self.temporal_compression = "compression_module" in args
        if self.temporal_compression:
            self.temporal_compressor = CostCompressFuse(args["compression_module"], self.max_cav)
        self.fusion_net = CostFusion(args["deformable_transformer"])
        self.align_ratio = args.get("align_ratio", args["voxel_size"][0] * args.get("feature_stride", 8))
        self.spatial_align = args.get("spatial_align", True)

        self.pre_fused_feature = None
        self.prev_backbone = None
        self.prev_map = None

        if args["backbone_fix"]:
            self.backbone_fix()

    def backbone_fix(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if self.shrink_flag:
            for parameter in self.shrink_conv.parameters():
                parameter.requires_grad = False
        if self.compression:
            for parameter in self.naive_compressor.parameters():
                parameter.requires_grad = False

        for parameter in self.cls_head.parameters():
            parameter.requires_grad = False
        for parameter in self.reg_head.parameters():
            parameter.requires_grad = False

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        batch_size = len(split_x)
        _, channels, height, width = x.shape
        padded = x.new_zeros((batch_size, self.max_cav, channels, height, width))
        mask = torch.zeros(batch_size, self.max_cav, dtype=torch.bool, device=x.device)
        for batch_idx, feat in enumerate(split_x):
            cav_num = feat.shape[0]
            padded[batch_idx, :cav_num] = feat
            mask[batch_idx, :cav_num] = True
        return padded, mask

    @staticmethod
    def regroup_flat(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        return torch.tensor_split(x, cum_sum_len[:-1].cpu())

    def _apply_detection_neck(self, spatial_features_2d):
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)
        return spatial_features_2d

    def _extract_local_features(self, data_dict):
        processed_lidar = data_dict["processed_lidar"]
        record_len = data_dict["record_len"]
        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar.get("voxel_num_points", None),
            "record_len": record_len,
            "batch_size": int(record_len.sum().item()),
        }
        batch_dict = self.backbone(batch_dict)
        local_features = self._apply_detection_neck(batch_dict["spatial_features_2d"])
        psm_single = self.cls_head(local_features)
        rm_single = self.reg_head(local_features)
        return local_features, psm_single, rm_single

    def _build_confidence_map(self, psm_single, record_len, target_size):
        confidence = psm_single.sigmoid().max(dim=1, keepdim=True)[0]
        if confidence.shape[-2:] != target_size:
            confidence = F.interpolate(
                confidence, size=target_size, mode="bilinear", align_corners=False
            )
        confidence, _ = self.regroup(confidence, record_len)
        return confidence

    def _build_prior_encoding(self, data_dict, mask, dtype):
        if "prior_encoding" in data_dict:
            prior = data_dict["prior_encoding"].to(device=mask.device, dtype=dtype)
            if prior.dim() == 5:
                prior = prior[..., 0, 0]
            return prior
        prior = torch.zeros(mask.shape[0], mask.shape[1], 3, dtype=dtype, device=mask.device)
        prior[:, 1:, 2] = 1
        return prior

    def _align_temporal_pair(self, prev_feature, current_feature, prev_ego_pose, current_ego_pose):
        if prev_feature is None or prev_ego_pose is None or current_ego_pose is None:
            return prev_feature, current_feature

        if prev_feature.dim() == 5:
            batch_size, cav_num, channels, height, width = current_feature.shape
            temporal_features = torch.stack([prev_feature, current_feature], dim=2)
            temporal_features = temporal_features.flatten(0, 1)
            ego_poses = torch.stack([prev_ego_pose, current_ego_pose], dim=1)
            ego_poses = ego_poses.unsqueeze(1).repeat(1, cav_num, 1, 1, 1).flatten(0, 1)
            aligned = align_features(
                temporal_features, ego_poses, self.align_ratio
            ).reshape(batch_size, cav_num, 2, channels, height, width)
            return aligned[:, :, 0], aligned[:, :, 1]

        temporal_features = torch.stack([prev_feature, current_feature], dim=1)
        ego_poses = torch.stack([prev_ego_pose, current_ego_pose], dim=1)
        aligned = align_features(temporal_features, ego_poses, self.align_ratio)
        return aligned[:, 0], aligned[:, 1]

    def _spatially_align_current_agents(self, regroup_feature, confidence_map, record_len, pairwise_t_matrix):
        if not self.spatial_align or pairwise_t_matrix is None:
            return regroup_feature, confidence_map
        regroup_feature = align_agents_to_ego(
            regroup_feature, pairwise_t_matrix, record_len, self.align_ratio
        )
        confidence_map = align_agents_to_ego(
            confidence_map, pairwise_t_matrix, record_len, self.align_ratio
        )
        return regroup_feature, confidence_map

    def forward_single_frame(self, data_dict, prev_state=None, use_prev=False):
        local_features, psm_single, rm_single = self._extract_local_features(data_dict)
        record_len = data_dict["record_len"]
        regroup_feature, mask = self.regroup(local_features, record_len)
        confidence_map = self._build_confidence_map(
            psm_single, record_len, regroup_feature.shape[-2:]
        )
        confidence_map = confidence_map * mask.reshape(mask.shape[0], mask.shape[1], 1, 1, 1)
        regroup_feature, confidence_map = self._spatially_align_current_agents(
            regroup_feature,
            confidence_map,
            record_len,
            data_dict.get("pairwise_t_matrix"),
        )

        prev_backbone = prev_map = prev_fused_feature = None
        if prev_state is not None and use_prev:
            prev_backbone, regroup_feature = self._align_temporal_pair(
                prev_state.get("backbone"),
                regroup_feature,
                prev_state.get("ego_pose"),
                data_dict.get("ego_pose"),
            )
            prev_map, confidence_map = self._align_temporal_pair(
                prev_state.get("confidence"),
                confidence_map,
                prev_state.get("ego_pose"),
                data_dict.get("ego_pose"),
            )
            prev_fused_feature, _ = self._align_temporal_pair(
                prev_state.get("fused_feature"),
                regroup_feature[:, 0],
                prev_state.get("ego_pose"),
                data_dict.get("ego_pose"),
            )

        if self.temporal_compression:
            regroup_feature, communication_rate = self.temporal_compressor(
                regroup_feature,
                confidence_map,
                prev_feat=prev_backbone,
                pre_map=prev_map,
                reset=not use_prev,
            )
        else:
            communication_rate = torch.tensor(0.0, device=local_features.device)

        prior_encoding = self._build_prior_encoding(data_dict, mask, regroup_feature.dtype)
        fused_feature = self.fusion_net(regroup_feature, mask, prior_encoding, prev_fused_feature)
        state = {
            "backbone": regroup_feature.detach(),
            "confidence": confidence_map.detach(),
            "fused_feature": fused_feature.detach(),
            "ego_pose": data_dict.get("ego_pose"),
        }
        return fused_feature, psm_single, rm_single, communication_rate, state

    def _split_single_outputs(self, psm_single, rm_single, record_len):
        split_psm = self.regroup_flat(psm_single, record_len)
        split_rm = self.regroup_flat(rm_single, record_len)
        return {
            "psm_single_v": torch.cat([item[0:1] for item in split_psm], dim=0),
            "psm_single_i": torch.cat([item[1:2] for item in split_psm], dim=0),
            "rm_single_v": torch.cat([item[0:1] for item in split_rm], dim=0),
            "rm_single_i": torch.cat([item[1:2] for item in split_rm], dim=0),
        }

    def forward(self, data_dict, inference=False):
        history = data_dict.get("history", [])
        if isinstance(history, dict):
            history = [history]

        prev_state = None
        for history_dict in history:
            with torch.no_grad():
                _, _, _, _, prev_state = self.forward_single_frame(
                    history_dict, prev_state, use_prev=prev_state is not None
                )

        if inference and not history and self.prev_backbone is not None:
            prev_state = {
                "backbone": self.prev_backbone,
                "confidence": self.prev_map,
                "fused_feature": self.pre_fused_feature,
            }

        fused_feature, psm_single, rm_single, communication_rate, state = self.forward_single_frame(
            data_dict, prev_state, use_prev=prev_state is not None
        )
        if inference:
            self.prev_backbone = state["backbone"]
            self.prev_map = state["confidence"]
            self.pre_fused_feature = state["fused_feature"]

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {"psm": psm, "rm": rm}
        output_dict.update(self._split_single_outputs(psm_single, rm_single, data_dict["record_len"]))
        output_dict["comm_rate"] = communication_rate
        return output_dict
