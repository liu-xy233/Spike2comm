import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from opencood.models.sub_modules.e_spikeformer_bev_backbone import (
    MSBlockSpikeSepConv,
    MSConvBlockSpikeSepConv,
    MSDownSampling,
    MultiSpikeNorm4,
)


def _conv_output_size(size, kernel_size, stride, padding):
    return int(math.floor((size + 2 * padding - kernel_size) / stride + 1))


def _as_stage_value(value, stage_index, name):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return value[0]
        if len(value) == 2 and stage_index >= 2:
            return value[stage_index - 2]
        if len(value) == 4:
            return value[stage_index]
        raise ValueError(
            "{} only supports scalar values or lists with 1, 2 or 4 items, got {}.".format(
                name, len(value)
            )
        )
    return value


class SpikeFPNBlock(nn.Module):
    """A minimal SpikeFPN block: MultiSpike -> Conv -> BN."""

    def __init__(self, in_channels, out_channels, kernel_size, padding, T):
        super().__init__()
        self.spike = MultiSpikeNorm4(T=T)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        x = self.spike(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class SpikeQFPN(nn.Module):
    """
    SDTv3-style SpikeFPN:
    1. MultiSpike -> 1x1 Conv -> BN
    2. Top-down nearest resize + element-wise add
    3. MultiSpike -> 3x3 Conv -> BN
    """

    def __init__(self, input_features, in_channels, out_channels, T):
        super().__init__()
        assert len(input_features) == len(in_channels), (
            "FPN input feature names and channels must have the same length, "
            "got {} and {}.".format(len(input_features), len(in_channels))
        )
        self.input_features = list(input_features)
        self.output_names = ["P{}".format(idx + 2) for idx in range(len(input_features))]
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for channels in in_channels:
            self.lateral_convs.append(
                SpikeFPNBlock(
                    in_channels=channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                    T=T,
                )
            )
            self.output_convs.append(
                SpikeFPNBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    T=T,
                )
            )

    def forward(self, feature_dict):
        assert isinstance(feature_dict, dict), "feature_dict must be a dict."
        for name in self.input_features:
            assert name in feature_dict, "Missing FPN input feature '{}'.".format(name)

        laterals = []
        for idx, name in enumerate(self.input_features):
            laterals.append(self.lateral_convs[idx](feature_dict[name]))

        for idx in range(len(laterals) - 1, 0, -1):
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx],
                size=laterals[idx - 1].shape[-2:],
                mode="nearest",
            )

        outputs = {}
        for idx, name in enumerate(self.output_names):
            outputs[name] = self.output_convs[idx](laterals[idx])
        return outputs


class BEVOutputAdapter(nn.Module):
    """Resize all QFPN outputs to x1 resolution with nearest and concatenate."""

    def __init__(self, pyramid_names):
        super().__init__()
        self.pyramid_names = list(pyramid_names)

    def forward(self, pyramid_features, target_size):
        resized = []
        for name in self.pyramid_names:
            assert name in pyramid_features, "Missing pyramid feature '{}'.".format(name)
            resized.append(
                F.interpolate(
                    pyramid_features[name], size=target_size, mode="nearest"
                )
            )
        return torch.cat(resized, dim=1)


class ESpikeFormerBEVBackboneV2(nn.Module):
    """
    Full SDTv3-style 4-stage encoder used only by the new v2 integration path.
    """

    backbone_mode = "full_sdtv3_qfpn_v2"

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.T = model_cfg.get("T", 4)

        embed_dims = model_cfg.get("embed_dims", [64, 128, 256, 360])
        conv_depths = model_cfg.get("conv_depths", [2, 2])
        transformer_depths = model_cfg.get("transformer_depths", [6, 2])
        fpn_cfg = model_cfg.get("fpn", {})

        assert len(embed_dims) == 4, "embed_dims must be [64, 128, 256, 360]."
        assert len(conv_depths) == 2, "conv_depths must contain two stages."
        assert len(transformer_depths) == 2, (
            "transformer_depths must contain Stage3 and Stage4 depths."
        )
        assert conv_depths[0] == 2, (
            "Stage1 is defined as two conv blocks (x1 and x2 branches); "
            "conv_depths[0] must be 2."
        )

        self.embed_dims = list(embed_dims)
        self.conv_depths = list(conv_depths)
        self.transformer_depths = list(transformer_depths)
        self.stage1_mid_channels = model_cfg.get(
            "stage1_mid_channels", self.embed_dims[0] // 2
        )
        assert self.stage1_mid_channels == 32, (
            "For the current DAIR setup Stage1 mid channels must stay 32, "
            "got {}.".format(self.stage1_mid_channels)
        )

        self.fpn_input_features = fpn_cfg.get("input_features", ["x2", "x3", "x4"])
        self.fpn_in_channels = fpn_cfg.get("in_channels", [64, 128, 360])
        self.fpn_out_channels = fpn_cfg.get("out_channels", 128)

        assert list(self.fpn_input_features) == ["x2", "x3", "x4"], (
            "The v2 path only supports x2/x3/x4 as FPN inputs, got {}.".format(
                self.fpn_input_features
            )
        )
        assert list(self.fpn_in_channels) == [64, 128, 360], (
            "The v2 path expects FPN channels [64, 128, 360], got {}.".format(
                self.fpn_in_channels
            )
        )
        assert self.fpn_out_channels == 128, (
            "The v2 path expects FPN out_channels=128, got {}.".format(
                self.fpn_out_channels
            )
        )

        self.num_heads = model_cfg.get("num_heads", 8)
        self.mlp_ratio = model_cfg.get("mlp_ratio", model_cfg.get("mlp_ratios", 4))
        self.qkv_bias = model_cfg.get("qkv_bias", False)
        self.drop_rate = model_cfg.get("drop_rate", 0.0)
        self.attn_drop_rate = model_cfg.get("attn_drop_rate", 0.0)
        self.drop_path_rate = model_cfg.get("drop_path_rate", 0.0)
        self.sr_ratios = model_cfg.get("sr_ratios", 1)

        total_transformer_depth = sum(self.transformer_depths)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, total_transformer_depth)]

        # Stage 1
        self.downsample1_1 = MSDownSampling(
            in_channels=input_channels,
            embed_dims=self.stage1_mid_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
            T=self.T,
        )
        self.stage1_block1 = MSConvBlockSpikeSepConv(
            dim=self.stage1_mid_channels,
            mlp_ratio=self.mlp_ratio,
            T=self.T,
        )
        self.downsample1_2 = MSDownSampling(
            in_channels=self.stage1_mid_channels,
            embed_dims=self.embed_dims[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            T=self.T,
        )
        self.stage1_block2 = MSConvBlockSpikeSepConv(
            dim=self.embed_dims[0],
            mlp_ratio=self.mlp_ratio,
            T=self.T,
        )

        # Stage 2
        self.downsample2 = MSDownSampling(
            in_channels=self.embed_dims[0],
            embed_dims=self.embed_dims[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            T=self.T,
        )
        self.stage2_blocks = nn.ModuleList(
            [
                MSConvBlockSpikeSepConv(
                    dim=self.embed_dims[1],
                    mlp_ratio=self.mlp_ratio,
                    T=self.T,
                )
                for _ in range(self.conv_depths[1])
            ]
        )

        # Stage 3
        stage3_num_heads = _as_stage_value(self.num_heads, 2, "num_heads")
        stage3_sr_ratio = _as_stage_value(self.sr_ratios, 2, "sr_ratios")
        self.downsample3 = MSDownSampling(
            in_channels=self.embed_dims[1],
            embed_dims=self.embed_dims[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            T=self.T,
        )
        self.stage3_blocks = nn.ModuleList(
            [
                MSBlockSpikeSepConv(
                    dim=self.embed_dims[2],
                    num_heads=stage3_num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[idx],
                    sr_ratio=stage3_sr_ratio,
                    T=self.T,
                )
                for idx in range(self.transformer_depths[0])
            ]
        )

        # Stage 4
        stage4_num_heads = _as_stage_value(self.num_heads, 3, "num_heads")
        stage4_sr_ratio = _as_stage_value(self.sr_ratios, 3, "sr_ratios")
        stage4_offset = self.transformer_depths[0]
        self.downsample4 = MSDownSampling(
            in_channels=self.embed_dims[2],
            embed_dims=self.embed_dims[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
            T=self.T,
        )
        self.stage4_blocks = nn.ModuleList(
            [
                MSBlockSpikeSepConv(
                    dim=self.embed_dims[3],
                    num_heads=stage4_num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[stage4_offset + idx],
                    sr_ratio=stage4_sr_ratio,
                    T=self.T,
                )
                for idx in range(self.transformer_depths[1])
            ]
        )

        self.spike_qfpn = SpikeQFPN(
            input_features=self.fpn_input_features,
            in_channels=self.fpn_in_channels,
            out_channels=self.fpn_out_channels,
            T=self.T,
        )
        self.bev_adapter = BEVOutputAdapter(self.spike_qfpn.output_names)

        self.num_levels = len(self.fpn_input_features)
        self.num_bev_features = self.fpn_out_channels * self.num_levels
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_backbone_mode(self):
        return self.backbone_mode

    def get_fusion_feature_names(self):
        return list(self.fpn_input_features)

    def _assert_stage_shapes(self, feature_dict, input_shape):
        height, width = input_shape
        expected_x1 = (
            _conv_output_size(height, kernel_size=7, stride=2, padding=3),
            _conv_output_size(width, kernel_size=7, stride=2, padding=3),
        )
        expected_x2 = (
            _conv_output_size(expected_x1[0], kernel_size=3, stride=2, padding=1),
            _conv_output_size(expected_x1[1], kernel_size=3, stride=2, padding=1),
        )
        expected_x3 = (
            _conv_output_size(expected_x2[0], kernel_size=3, stride=2, padding=1),
            _conv_output_size(expected_x2[1], kernel_size=3, stride=2, padding=1),
        )
        expected_stage3 = (
            _conv_output_size(expected_x3[0], kernel_size=3, stride=2, padding=1),
            _conv_output_size(expected_x3[1], kernel_size=3, stride=2, padding=1),
        )
        expected_x4 = (
            _conv_output_size(expected_stage3[0], kernel_size=3, stride=1, padding=1),
            _conv_output_size(expected_stage3[1], kernel_size=3, stride=1, padding=1),
        )

        assert feature_dict["x1"].shape[1] == self.stage1_mid_channels
        assert feature_dict["x2"].shape[1] == self.embed_dims[0]
        assert feature_dict["x3"].shape[1] == self.embed_dims[1]
        assert feature_dict["stage3_out"].shape[1] == self.embed_dims[2]
        assert feature_dict["x4"].shape[1] == self.embed_dims[3]

        assert feature_dict["x1"].shape[-2:] == expected_x1, (
            "x1 shape mismatch, expected {}, got {}.".format(
                expected_x1, feature_dict["x1"].shape[-2:]
            )
        )
        assert feature_dict["x2"].shape[-2:] == expected_x2, (
            "x2 shape mismatch, expected {}, got {}.".format(
                expected_x2, feature_dict["x2"].shape[-2:]
            )
        )
        assert feature_dict["x3"].shape[-2:] == expected_x3, (
            "x3 shape mismatch, expected {}, got {}.".format(
                expected_x3, feature_dict["x3"].shape[-2:]
            )
        )
        assert feature_dict["stage3_out"].shape[-2:] == expected_stage3, (
            "stage3_out shape mismatch, expected {}, got {}.".format(
                expected_stage3, feature_dict["stage3_out"].shape[-2:]
            )
        )
        assert feature_dict["x4"].shape[-2:] == expected_x4, (
            "x4 shape mismatch, expected {}, got {}.".format(
                expected_x4, feature_dict["x4"].shape[-2:]
            )
        )

    def extract_stage1_features(self, spatial_features):
        assert spatial_features.dim() == 4, (
            "spatial_features must be a 4D tensor, got {} dims.".format(
                spatial_features.dim()
            )
        )
        x1 = self.downsample1_1(spatial_features)
        x1 = self.stage1_block1(x1)

        x2 = self.downsample1_2(x1)
        x2 = self.stage1_block2(x2)

        return {"x1": x1, "x2": x2}

    def extract_sdtv3_features_from_x2(self, x2, x1=None):
        assert x2.dim() == 4, "x2 must be a 4D tensor."
        assert x2.shape[1] == self.embed_dims[0], (
            "x2 must have {} channels, got {}.".format(
                self.embed_dims[0], x2.shape[1]
            )
        )
        if x1 is not None:
            assert x1.shape[1] == self.stage1_mid_channels, (
                "x1 must have {} channels, got {}.".format(
                    self.stage1_mid_channels, x1.shape[1]
                )
            )

        x3 = self.downsample2(x2)
        for block in self.stage2_blocks:
            x3 = block(x3)

        stage3_out = self.downsample3(x3)
        for block in self.stage3_blocks:
            stage3_out = block(stage3_out)

        x4 = self.downsample4(stage3_out)
        for block in self.stage4_blocks:
            x4 = block(x4)

        features = {
            "x2": x2,
            "x3": x3,
            "stage3_out": stage3_out,
            "x4": x4,
        }
        if x1 is not None:
            features["x1"] = x1
        return features

    def extract_sdtv3_features(self, spatial_features):
        stage1_features = self.extract_stage1_features(spatial_features)
        features = self.extract_sdtv3_features_from_x2(
            stage1_features["x2"], x1=stage1_features["x1"]
        )
        self._assert_stage_shapes(features, spatial_features.shape[-2:])
        return features

    def forward_qfpn(self, feature_dict):
        pyramid_features = self.spike_qfpn(feature_dict)
        expected_names = self.spike_qfpn.output_names
        for name in expected_names:
            assert name in pyramid_features, "Missing QFPN output '{}'.".format(name)
            assert pyramid_features[name].shape[1] == self.fpn_out_channels, (
                "{} must have {} channels, got {}.".format(
                    name, self.fpn_out_channels, pyramid_features[name].shape[1]
                )
            )
        return pyramid_features

    def adapt_bev_output(self, pyramid_features, x1):
        target_size = x1.shape[-2:]
        spatial_features_2d = self.bev_adapter(pyramid_features, target_size)
        assert spatial_features_2d.shape[1] == self.num_bev_features, (
            "BEV adapter output must have {} channels, got {}.".format(
                self.num_bev_features, spatial_features_2d.shape[1]
            )
        )
        assert spatial_features_2d.shape[-2:] == target_size, (
            "BEV adapter output size mismatch, expected {}, got {}.".format(
                target_size, spatial_features_2d.shape[-2:]
            )
        )
        return spatial_features_2d

    def build_local_bev_features(self, spatial_features):
        feature_dict = self.extract_sdtv3_features(spatial_features)
        pyramid_features = self.forward_qfpn(feature_dict)
        spatial_features_2d = self.adapt_bev_output(pyramid_features, feature_dict["x1"])
        return feature_dict, pyramid_features, spatial_features_2d

    def forward(self, data_dict):
        spatial_features = data_dict["spatial_features"]
        feature_dict, pyramid_features, spatial_features_2d = self.build_local_bev_features(
            spatial_features
        )

        data_dict["spatial_features_2x"] = feature_dict["x1"]
        data_dict["spatial_features_4x"] = feature_dict["x2"]
        data_dict["spatial_features_8x"] = feature_dict["x3"]
        data_dict["spatial_features_16x"] = feature_dict["x4"]
        data_dict["stage3_out"] = feature_dict["stage3_out"]
        data_dict["sdtv3_features"] = feature_dict
        data_dict["spike_qfpn_features"] = pyramid_features
        data_dict["spatial_features_2d"] = spatial_features_2d
        return data_dict
