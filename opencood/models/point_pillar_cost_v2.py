import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.comm_modules.cost_compress_fuse import CostCompressFuse
from opencood.models.fuse_modules.cost_fusion import CostFusion
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.e_spikeformer_bev_backbone_v2 import ESpikeFormerBEVBackboneV2
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


class PointPillarCostV2(nn.Module):
    """CoST feature selection and fusion on top of the local SDTv3 PointPillar path."""

    def __init__(self, args):
        super().__init__()
        self.max_cav = args["max_cav"]
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
            raise ValueError("PointPillarCostV2 expects the SDTv3-QFPN-v2 backbone.")
        self.backbone = ESpikeFormerBEVBackboneV2(backbone_args, 64)

        self.shrink_flag = "shrink_header" in args
        if not self.shrink_flag:
            raise ValueError("PointPillarCostV2 requires shrink_header for 256-channel CoST fusion.")
        self.shrink_conv = DownsampleConv(args["shrink_header"])
        self.head_in_channels = args["shrink_header"]["dim"][-1]

        self.compression = args["compression"] > 0
        if self.compression:
            self.naive_compressor = NaiveCompressor(self.head_in_channels, args["compression"])

        self.dcn = "dcn" in args
        if self.dcn:
            from opencood.models.sub_modules.dcn_net import DCNNet

            self.dcn_net = DCNNet(args["dcn"])

        self.cls_head = nn.Conv2d(self.head_in_channels, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(self.head_in_channels, 7 * args["anchor_number"], kernel_size=1)

        self.temporal_compression = "compression_module" in args
        if self.temporal_compression:
            self.temporal_compressor = CostCompressFuse(args["compression_module"], self.max_cav)
        self.fusion_net = CostFusion(args["deformable_transformer"])

        self.backbone_input_norm_eps = args.get("backbone_input_norm_eps", 1e-6)
        self.align_ratio = args.get("align_ratio", args.get("feature_stride", 4) * args["voxel_size"][0])

        self.pre_fused_feature = None
        self.prev_backbone = None
        self.prev_map = None
        self.communication_rates = []

        if args["backbone_fix"]:
            self.backbone_fix()

    def backbone_fix(self):
        for module in [self.pillar_vfe, self.scatter, self.backbone, self.shrink_conv, self.cls_head, self.reg_head]:
            for parameter in module.parameters():
                parameter.requires_grad = False
        if self.compression:
            for parameter in self.naive_compressor.parameters():
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

    def regroup_flat(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        return torch.tensor_split(x, cum_sum_len[:-1].cpu())

    def _apply_detection_neck(self, spatial_features_2d):
        spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        if self.dcn:
            spatial_features_2d = self.dcn_net(spatial_features_2d)
        return spatial_features_2d

    def _normalize_backbone_input(self, spatial_features):
        channel_mean = spatial_features.mean(dim=(2, 3), keepdim=True)
        channel_std = spatial_features.std(dim=(2, 3), keepdim=True, unbiased=False)
        return (spatial_features - channel_mean) / channel_std.clamp_min(self.backbone_input_norm_eps)

    def _extract_local_features(self, data_dict):
        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        record_len = data_dict["record_len"]
        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "record_len": record_len,
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict["spatial_features"] = self._normalize_backbone_input(batch_dict["spatial_features"])
        batch_dict = self.backbone(batch_dict)
        local_feature = self._apply_detection_neck(batch_dict["spatial_features_2d"])
        psm_single = self.cls_head(local_feature)
        rm_single = self.reg_head(local_feature)
        return local_feature, psm_single, rm_single

    def _build_confidence_map(self, psm_single, record_len, target_size):
        confidence = psm_single.sigmoid().max(dim=1, keepdim=True)[0]
        if confidence.shape[-2:] != target_size:
            confidence = F.interpolate(confidence, size=target_size, mode="bilinear", align_corners=False)
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

    def _align_temporal_feature(self, prev_feature, prev_ego_pose, current_ego_pose):
        if prev_feature is None or prev_ego_pose is None or current_ego_pose is None:
            return None
        original_shape = prev_feature.shape
        if prev_feature.dim() == 5:
            batch_size, cav_num, channels, height, width = prev_feature.shape
            flat_feature = prev_feature.flatten(0, 1)
            transform = self._ego_motion_to_affine(prev_ego_pose, current_ego_pose, height, width)
            transform = transform.unsqueeze(1).repeat(1, cav_num, 1, 1).flatten(0, 1)
            aligned = warp_affine_simple(flat_feature, transform, (height, width))
            return aligned.reshape(original_shape)

        batch_size, channels, height, width = prev_feature.shape
        transform = self._ego_motion_to_affine(prev_ego_pose, current_ego_pose, height, width)
        return warp_affine_simple(prev_feature, transform, (height, width))

    def _ego_motion_to_affine(self, prev_ego_pose, current_ego_pose, height, width):
        # ego_pose is T_world_ego. We need T_current_ego_previous_ego.
        transform = torch.linalg.solve(current_ego_pose, prev_ego_pose)
        affine = transform[:, [0, 1], :][:, :, [0, 1, 3]].clone()
        affine[..., 0, 1] = affine[..., 0, 1] * height / width
        affine[..., 1, 0] = affine[..., 1, 0] * width / height
        affine[..., 0, 2] = affine[..., 0, 2] / (self.align_ratio * width) * 2
        affine[..., 1, 2] = affine[..., 1, 2] / (self.align_ratio * height) * 2
        return affine

    def forward_single_frame(self, data_dict, prev_state=None, use_prev=False):
        local_feature, psm_single, rm_single = self._extract_local_features(data_dict)
        record_len = data_dict["record_len"]
        regroup_feature, mask = self.regroup(local_feature, record_len)
        confidence_map = self._build_confidence_map(psm_single, record_len, regroup_feature.shape[-2:])
        confidence_map = confidence_map * mask.reshape(mask.shape[0], mask.shape[1], 1, 1, 1)

        prev_backbone = prev_map = prev_fused_feature = None
        if prev_state is not None and use_prev:
            prev_backbone = self._align_temporal_feature(
                prev_state.get("backbone"), prev_state.get("ego_pose"), data_dict.get("ego_pose")
            )
            prev_map = self._align_temporal_feature(
                prev_state.get("confidence"), prev_state.get("ego_pose"), data_dict.get("ego_pose")
            )
            prev_fused_feature = self._align_temporal_feature(
                prev_state.get("fused_feature"), prev_state.get("ego_pose"), data_dict.get("ego_pose")
            )
            if prev_backbone is None:
                prev_backbone = prev_state.get("backbone")
            if prev_map is None:
                prev_map = prev_state.get("confidence")
            if prev_fused_feature is None:
                prev_fused_feature = prev_state.get("fused_feature")

        if self.temporal_compression:
            regroup_feature, communication_rate = self.temporal_compressor(
                regroup_feature,
                confidence_map,
                prev_feat=prev_backbone,
                pre_map=prev_map,
                reset=not use_prev,
            )
        else:
            communication_rate = torch.tensor(0.0, device=local_feature.device)

        prior_encoding = self._build_prior_encoding(data_dict, mask, regroup_feature.dtype)
        fused_feature = self.fusion_net(regroup_feature, mask, prior_encoding, prev_fused_feature)
        state = {
            "backbone": regroup_feature.detach(),
            "confidence": confidence_map.detach(),
            "fused_feature": fused_feature.detach(),
            "ego_pose": data_dict.get("ego_pose"),
        }
        return fused_feature, psm_single, rm_single, communication_rate, state

    def forward(self, data_dict, inference=False):
        history = data_dict.get("history", [])
        if isinstance(history, dict):
            history = [history]

        prev_state = None
        for history_dict in history:
            with torch.no_grad():
                _, _, _, _, prev_state = self.forward_single_frame(history_dict, prev_state, use_prev=prev_state is not None)

        if inference and not history and self.prev_backbone is not None:
            prev_state = {
                "backbone": self.prev_backbone,
                "confidence": self.prev_map,
                "fused_feature": self.pre_fused_feature,
            }

        use_prev = prev_state is not None
        fused_feature, psm_single, rm_single, communication_rate, state = self.forward_single_frame(
            data_dict, prev_state, use_prev=use_prev
        )
        if inference:
            self.prev_backbone = state["backbone"]
            self.prev_map = state["confidence"]
            self.pre_fused_feature = state["fused_feature"]

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict = {"psm": psm, "rm": rm}

        split_psm_single = self.regroup_flat(psm_single, data_dict["record_len"])
        split_rm_single = self.regroup_flat(rm_single, data_dict["record_len"])
        output_dict.update(
            {
                "psm_single_v": torch.cat([item[0:1] for item in split_psm_single], dim=0),
                "psm_single_i": torch.cat([item[1:2] for item in split_psm_single], dim=0),
                "rm_single_v": torch.cat([item[0:1] for item in split_rm_single], dim=0),
                "rm_single_i": torch.cat([item[1:2] for item in split_rm_single], dim=0),
                "comm_rate": communication_rate,
            }
        )
        return output_dict
