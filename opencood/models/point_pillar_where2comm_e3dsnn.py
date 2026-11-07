# -*- coding: utf-8 -*-
"""Where2comm with an E-3DSNN-style sparse spike local BEV encoder."""

import torch
import torch.nn as nn

from opencood.models.fuse_modules.where2comm_attn import Where2comm
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.e3dsnn_spconv_backbone import (
    E3DSNNSparseBEVBackbone,
)
from opencood.models.sub_modules.naive_compress import NaiveCompressor


class PointPillarWhere2commE3dsnn(nn.Module):
    """Drop-in Where2comm model with sparse spike local feature extraction.

    The post-backbone communication, feature selection, fusion, and detection
    heads follow the current PointPillarWhere2comm path.
    """

    def __init__(self, args):
        super(PointPillarWhere2commE3dsnn, self).__init__()

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

        self.fusion_net = Where2comm(args["fusion_args"])
        self.multi_scale = args["fusion_args"]["multi_scale"]

        self.cls_head = nn.Conv2d(256, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(256, 7 * args["anchor_number"], kernel_size=1)
        self.iou_head = nn.Conv2d(256, args["anchor_number"], kernel_size=1)

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

        for parameter in self.iou_head.parameters():
            parameter.requires_grad = False

    @staticmethod
    def regroup(x, record_len):
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

    def _split_single_outputs(self, psm_single, rm_single, iou_single,
                              record_len):
        split_psm = self.regroup(psm_single, record_len)
        split_rm = self.regroup(rm_single, record_len)
        split_iou = self.regroup(iou_single, record_len)
        psm_single_v, psm_single_i = [], []
        rm_single_v, rm_single_i = [], []
        iou_single_v, iou_single_i = [], []

        for batch_idx in range(len(split_psm)):
            psm_single_v.append(split_psm[batch_idx][0:1])
            psm_single_i.append(split_psm[batch_idx][1:2])
            rm_single_v.append(split_rm[batch_idx][0:1])
            rm_single_i.append(split_rm[batch_idx][1:2])
            iou_single_v.append(split_iou[batch_idx][0:1])
            iou_single_i.append(split_iou[batch_idx][1:2])

        return {
            "psm_single_v": torch.cat(psm_single_v, dim=0),
            "psm_single_i": torch.cat(psm_single_i, dim=0),
            "rm_single_v": torch.cat(rm_single_v, dim=0),
            "rm_single_i": torch.cat(rm_single_i, dim=0),
            "iou_single_v": torch.cat(iou_single_v, dim=0),
            "iou_single_i": torch.cat(iou_single_i, dim=0),
        }

    def _extract_cached_context(self, data_dict):
        processed_lidar = data_dict["processed_lidar"]
        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]

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
        iou_single = self.iou_head(local_features)

        return {
            "spatial_features": batch_dict["spatial_features"],
            "spatial_features_2d": batch_dict["spatial_features_2d"],
            "original_spatial_features": batch_dict["original_spatial_features"],
            "multi_scale_neck_debug": batch_dict.get("multi_scale_neck_debug", {}),
            "local_features": local_features,
            "psm_single": psm_single,
            "rm_single": rm_single,
            "iou_single": iou_single,
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
        iou_single = cached_context["iou_single"]

        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                cached_context["spatial_features"],
                psm_single,
                record_len,
                pairwise_t_matrix,
                self.backbone,
                [self.shrink_conv, self.cls_head, self.reg_head],
                override_communication_mask=override_communication_mask,
                return_communication_debug=return_communication_debug,
            )
            fused_feature = self._apply_detection_neck(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                cached_context["spatial_features_2d"],
                psm_single,
                record_len,
                pairwise_t_matrix,
                override_communication_mask=override_communication_mask,
                return_communication_debug=return_communication_debug,
            )

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        iou = self.iou_head(fused_feature)

        output_dict = {
            "psm": psm,
            "rm": rm,
            "iou": iou,
            # Keep the packed local predictions for inference-time consumers
            # that need every non-ego agent.  The legacy vehicle/infrastructure
            # aliases below remain unchanged for the existing auxiliary losses.
            "psm_single": psm_single,
            "rm_single": rm_single,
            "iou_single": iou_single,
        }
        if return_communication_debug and cached_context["multi_scale_neck_debug"]:
            output_dict["multi_scale_neck_debug"] = cached_context[
                "multi_scale_neck_debug"
            ]
        output_dict.update(result_dict)
        output_dict.update(
            self._split_single_outputs(
                psm_single, rm_single, iou_single, record_len
            )
        )
        # OPV2V packed multi-agent prediction contract.
        # Keep every agent prediction instead of exposing only V/I slices.
        output_dict["psm_single"] = psm_single
        output_dict["rm_single"] = rm_single
        output_dict["iou_single"] = iou_single

        # Preserve the fused IoU-quality prediction for post-processing.
        if "iou" not in output_dict:
            output_dict["iou"] = self.iou_head(fused_feature)

        output_dict["comm_rate"] = communication_rates
        return output_dict

    def forward(self, data_dict):
        cached_context = self._extract_cached_context(data_dict)
        return self._forward_from_cached_context(
            cached_context,
            override_communication_mask=data_dict.get("override_communication_mask"),
            return_communication_debug=data_dict.get("return_communication_debug", False),
        )
