import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.comm_modules.where2comm import Communication
from opencood.models.comm_modules.patch_utils import (
    communication_rate_from_mask,
    flatten_agent_mask,
    resize_mask_nearest,
)
from opencood.models.fuse_modules.where2comm_attn import (
    AttenFusion,
    MaxFusion,
    TransformerFusion,
)
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


class Where2commV2(nn.Module):
    """Where2comm fusion path dedicated to the full SDTv3 v2 backbone."""

    def __init__(self, args):
        super(Where2commV2, self).__init__()

        self.communication = False
        self.round = 1
        if "communication" in args:
            self.communication = True
            self.naive_communication = Communication(args["communication"])
            if "round" in args["communication"]:
                self.round = args["communication"]["round"]

        self.discrete_ratio = args["voxel_size"][0]
        self.downsample_rate = args["downsample_rate"]
        self.agg_mode = args["agg_operator"]["mode"]
        self.multi_scale = args["multi_scale"]
        self.backbone_mode = args.get("backbone_mode", "")
        self.num_levels = args.get("num_levels", 0)
        self.num_filters = args.get("num_filters", [])

        assert self.multi_scale, "Where2commV2 only supports multi_scale=True."
        assert self.backbone_mode == "full_sdtv3_qfpn_v2", (
            "Where2commV2 is only intended for full_sdtv3_qfpn_v2, got '{}'.".format(
                self.backbone_mode
            )
        )
        assert self.num_levels == len(self.num_filters), (
            "fusion num_levels must match num_filters length, got {} and {}.".format(
                self.num_levels, len(self.num_filters)
            )
        )

        self.fuse_modules = nn.ModuleList()
        for idx in range(self.num_levels):
            if self.agg_mode == "ATTEN":
                fuse_network = AttenFusion(self.num_filters[idx])
            elif self.agg_mode == "MAX":
                fuse_network = MaxFusion()
            elif self.agg_mode == "Transformer":
                fuse_network = TransformerFusion(
                    channels=self.num_filters[idx],
                    n_head=args["agg_operator"]["n_head"],
                    with_spe=args["agg_operator"]["with_spe"],
                    with_scm=args["agg_operator"]["with_scm"],
                )
            else:
                raise ValueError("Unsupported aggregation mode '{}'.".format(self.agg_mode))
            self.fuse_modules.append(fuse_network)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def _prepare_pairwise_t_matrix(self, x, pairwise_t_matrix):
        _, _, height, width = x.shape
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * height / width
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * width / height
        pairwise_t_matrix[..., 0, 2] = (
            pairwise_t_matrix[..., 0, 2]
            / (self.downsample_rate * self.discrete_ratio * width)
            * 2
        )
        pairwise_t_matrix[..., 1, 2] = (
            pairwise_t_matrix[..., 1, 2]
            / (self.downsample_rate * self.discrete_ratio * height)
            * 2
        )
        return pairwise_t_matrix

    def _validate_backbone(self, backbone):
        assert backbone is not None, "backbone must be provided for Where2commV2."
        backbone_mode = backbone.get_backbone_mode()
        assert backbone_mode == self.backbone_mode, (
            "Backbone mode mismatch, expected '{}', got '{}'.".format(
                self.backbone_mode, backbone_mode
            )
        )

    def _build_communication_mask(
        self,
        rm,
        record_len,
        pairwise_t_matrix,
        target_size,
        override_communication_mask=None,
    ):
        if override_communication_mask is not None:
            communication_masks = override_communication_mask.to(device=rm.device, dtype=rm.dtype)
            communication_masks = resize_mask_nearest(communication_masks, target_size)
            communication_masks = flatten_agent_mask(communication_masks, record_len)
            communication_rates = communication_rate_from_mask(communication_masks, record_len)
            return communication_masks, communication_rates, {
                "communication_masks": communication_masks,
                "communication_source": "override",
            }

        if not self.communication:
            return None, torch.tensor(0.0, device=rm.device), {
                "communication_masks": None,
                "communication_source": "disabled",
            }

        batch_confidence_maps = self.regroup(rm, record_len)
        _, communication_masks, communication_rates = self.naive_communication(
            batch_confidence_maps, record_len, pairwise_t_matrix
        )
        communication_masks = F.interpolate(
            communication_masks.float(), size=target_size, mode="nearest"
        )
        assert communication_masks.shape[-2:] == target_size, (
            "Communication mask resize failed, expected {}, got {}.".format(
                target_size, communication_masks.shape[-2:]
            )
        )
        return communication_masks, communication_rates, {
            "communication_masks": communication_masks,
            "communication_source": "confidence_threshold",
        }

    def _fuse_single_scale(self, feature, record_len, pairwise_t_matrix, fuse_module):
        batch_node_features = self.regroup(feature, record_len)
        fused_feature = []
        batch_size = pairwise_t_matrix.shape[0]

        for batch_idx in range(batch_size):
            cav_num = int(record_len[batch_idx])
            t_matrix = pairwise_t_matrix[batch_idx][:cav_num, :cav_num, :, :]
            node_features = batch_node_features[batch_idx]
            _, channels, height, width = node_features.shape
            assert channels > 0 and height > 0 and width > 0
            neighbor_feature = warp_affine_simple(
                node_features, t_matrix[0, :, :, :], (height, width)
            )
            fused_feature.append(fuse_module(neighbor_feature))

        return torch.stack(fused_feature)

    def forward(
        self,
        x,
        rm,
        record_len,
        pairwise_t_matrix,
        backbone=None,
        local_backbone_features=None,
        override_communication_mask=None,
    ):
        self._validate_backbone(backbone)
        pairwise_t_matrix = self._prepare_pairwise_t_matrix(x, pairwise_t_matrix)

        if local_backbone_features is None:
            stage1_features = backbone.extract_stage1_features(x)
            x1 = stage1_features["x1"]
            x2 = stage1_features["x2"]
        else:
            assert "x1" in local_backbone_features and "x2" in local_backbone_features, (
                "local_backbone_features must at least contain x1 and x2."
            )
            x1 = local_backbone_features["x1"]
            x2 = local_backbone_features["x2"]

        communication_masks, communication_rates, communication_debug = self._build_communication_mask(
            rm,
            record_len,
            pairwise_t_matrix,
            x2.shape[-2:],
            override_communication_mask=override_communication_mask,
        )
        if communication_masks is not None:
            x2 = x2 * communication_masks

        feature_dict = backbone.extract_sdtv3_features_from_x2(x2, x1=x1)

        fused_multiscale = {}
        fusion_feature_names = backbone.get_fusion_feature_names()
        assert len(fusion_feature_names) == self.num_levels, (
            "Backbone fusion feature count {} does not match configured num_levels {}.".format(
                len(fusion_feature_names), self.num_levels
            )
        )

        for idx, feature_name in enumerate(fusion_feature_names):
            assert feature_name in feature_dict, "Missing feature '{}'.".format(feature_name)
            fused_multiscale[feature_name] = self._fuse_single_scale(
                feature=feature_dict[feature_name],
                record_len=record_len,
                pairwise_t_matrix=pairwise_t_matrix,
                fuse_module=self.fuse_modules[idx],
            )

        fused_pyramid = backbone.forward_qfpn(fused_multiscale)
        fused_feature = backbone.adapt_bev_output(fused_pyramid, x1)

        assert fused_feature.shape[0] == pairwise_t_matrix.shape[0], (
            "The fused collaborative feature must keep batch size {}, got {}.".format(
                pairwise_t_matrix.shape[0], fused_feature.shape[0]
            )
        )
        return fused_feature, communication_rates, communication_debug
