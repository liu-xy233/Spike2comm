# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from opencood.models.fuse_modules.where2comm_attn_v2 import Where2commV2
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.e_spikeformer_bev_backbone_v2 import (
    ESpikeFormerBEVBackboneV2,
)
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class PointPillarWhere2commV2(nn.Module):
    """Full SDTv3 + Where2comm v2 integration path."""

    def __init__(self, args):
        super(PointPillarWhere2commV2, self).__init__()

        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])

        backbone_args = args["base_bev_backbone"]
        backbone_type = backbone_args.get("type", "").lower()
        supported_backbone_types = [
            "full_sdtv3_qfpn_v2",
            "e_spikeformer_full_sdtv3_qfpn_v2",
            "e_spikeformer_sdtv3_qfpn_v2",
        ]
        if backbone_type not in supported_backbone_types:
            raise ValueError(
                "PointPillarWhere2commV2 only supports the new full SDTv3 backbone, "
                "got '{}'.".format(backbone_type)
            )
        self.backbone = ESpikeFormerBEVBackboneV2(backbone_args, 64)

        self.shrink_flag = "shrink_header" in args
        if not self.shrink_flag:
            raise ValueError(
                "The v2 SDTv3 path requires shrink_header so that 384-channel BEV "
                "features are reduced to the original 256-channel head input."
            )
        shrink_cfg = args["shrink_header"]
        assert shrink_cfg["input_dim"] == self.backbone.num_bev_features, (
            "shrink_header.input_dim must match backbone output channels {}, got {}.".format(
                self.backbone.num_bev_features, shrink_cfg["input_dim"]
            )
        )
        self.shrink_conv = DownsampleConv(shrink_cfg)
        self.head_in_channels = shrink_cfg["dim"][-1]
        assert self.head_in_channels == 256, (
            "The v2 path keeps the original 256-channel detection heads, got {}.".format(
                self.head_in_channels
            )
        )

        self.compression = False
        if args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(self.head_in_channels, args["compression"])

        self.dcn = False
        if "dcn" in args:
            from opencood.models.sub_modules.dcn_net import DCNNet

            self.dcn = True
            self.dcn_net = DCNNet(args["dcn"])

        self.fusion_net = Where2commV2(args["fusion_args"])
        self.multi_scale = args["fusion_args"]["multi_scale"]
        assert self.multi_scale, "The v2 path only supports multi_scale=True."
        self.backbone_input_norm_eps = args.get("backbone_input_norm_eps", 1e-6)

        self.cls_head = nn.Conv2d(
            self.head_in_channels, args["anchor_number"], kernel_size=1
        )
        self.reg_head = nn.Conv2d(
            self.head_in_channels, 7 * args["anchor_number"], kernel_size=1
        )

        if args["backbone_fix"]:
            self.backbone_fix()

    def backbone_fix(self):
        for parameter in self.pillar_vfe.parameters():
            parameter.requires_grad = False

        for parameter in self.scatter.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if self.compression:
            for parameter in self.naive_compressor.parameters():
                parameter.requires_grad = False

        for parameter in self.shrink_conv.parameters():
            parameter.requires_grad = False

        for parameter in self.cls_head.parameters():
            parameter.requires_grad = False

        for parameter in self.reg_head.parameters():
            parameter.requires_grad = False

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def _apply_detection_neck(self, spatial_features_2d):
        assert spatial_features_2d.shape[1] == self.backbone.num_bev_features, (
            "Pre-shrink local/collaborative features must have {} channels, got {}.".format(
                self.backbone.num_bev_features, spatial_features_2d.shape[1]
            )
        )
        spatial_features_2d = self.shrink_conv(spatial_features_2d)
        assert spatial_features_2d.shape[1] == self.head_in_channels, (
            "shrink_conv output must have {} channels, got {}.".format(
                self.head_in_channels, spatial_features_2d.shape[1]
            )
        )

        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)
        return spatial_features_2d

    def _normalize_backbone_input(self, spatial_features):
        assert spatial_features.dim() == 4, (
            "backbone input must be a 4D tensor [N, C, H, W], got {} dims.".format(
                spatial_features.dim()
            )
        )
        channel_mean = spatial_features.mean(dim=(2, 3), keepdim=True)
        channel_std = spatial_features.std(dim=(2, 3), keepdim=True, unbiased=False)
        channel_std = channel_std.clamp_min(self.backbone_input_norm_eps)
        return (spatial_features - channel_mean) / channel_std

    def _extract_cached_context(self, data_dict):
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "record_len": record_len,
        }

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict["spatial_features"] = self._normalize_backbone_input(
            batch_dict["spatial_features"]
        )
        batch_dict = self.backbone(batch_dict)

        local_spatial_features_2d = batch_dict["spatial_features_2d"]
        assert local_spatial_features_2d.shape[1] == self.backbone.num_bev_features, (
            "Local backbone output must have {} channels, got {}.".format(
                self.backbone.num_bev_features, local_spatial_features_2d.shape[1]
            )
        )

        local_head_input = self._apply_detection_neck(local_spatial_features_2d)
        psm_single = self.cls_head(local_head_input)
        rm_single = self.reg_head(local_head_input)

        return {
            "spatial_features": batch_dict["spatial_features"],
            "sdtv3_features": batch_dict["sdtv3_features"],
            "psm_single": psm_single,
            "rm_single": rm_single,
            "record_len": record_len,
            "pairwise_t_matrix": pairwise_t_matrix,
        }

    def _forward_from_cached_context(
        self,
        cached_context,
        override_communication_mask=None,
        return_communication_debug=False,
    ):
        record_len = cached_context["record_len"]
        pairwise_t_matrix = cached_context["pairwise_t_matrix"]
        psm_single = cached_context["psm_single"]
        rm_single = cached_context["rm_single"]

        fused_feature, communication_rates, result_dict = self.fusion_net(
            cached_context["spatial_features"],
            psm_single,
            record_len,
            pairwise_t_matrix,
            backbone=self.backbone,
            local_backbone_features=cached_context["sdtv3_features"],
            override_communication_mask=override_communication_mask,
        )
        fused_feature = self._apply_detection_neck(fused_feature)
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {"psm": psm, "rm": rm}
        if return_communication_debug:
            output_dict.update(result_dict)
        elif "communication_source" in result_dict:
            output_dict["communication_source"] = result_dict["communication_source"]

        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []
        for batch_idx in range(len(split_psm_single)):
            psm_single_v.append(split_psm_single[batch_idx][0:1])
            psm_single_i.append(split_psm_single[batch_idx][1:2])
            rm_single_v.append(split_rm_single[batch_idx][0:1])
            rm_single_i.append(split_rm_single[batch_idx][1:2])

        psm_single_v = torch.cat(psm_single_v, dim=0)
        psm_single_i = torch.cat(psm_single_i, dim=0)
        rm_single_v = torch.cat(rm_single_v, dim=0)
        rm_single_i = torch.cat(rm_single_i, dim=0)

        output_dict.update(
            {
                "psm_single_v": psm_single_v,
                "psm_single_i": psm_single_i,
                "rm_single_v": rm_single_v,
                "rm_single_i": rm_single_i,
                "comm_rate": communication_rates,
            }
        )
        return output_dict

    def forward(self, data_dict):
        cached_context = self._extract_cached_context(data_dict)
        return self._forward_from_cached_context(
            cached_context,
            override_communication_mask=data_dict.get("override_communication_mask"),
            return_communication_debug=data_dict.get("return_communication_debug", False),
        )
