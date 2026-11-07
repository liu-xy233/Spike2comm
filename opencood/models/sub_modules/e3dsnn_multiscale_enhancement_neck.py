# -*- coding: utf-8 -*-
"""Lightweight multi-scale BEV enhancement for the E-3DSNN backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.e3dsnn_bev_adapter import (
    SparseToBEVScatterAdapter,
)
from opencood.models.sub_modules.spike_residual_conv import SpikeResidualConvBlock


def _conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


class MultiScaleEnhancementNeck(nn.Module):
    """Fuse Stage2/3/4 sparse features into one residual BEV feature.

    Each sparse feature is first projected to BEV using the same dense-height
    collapse used by the existing E-3DSNN output adapter.  A lightweight FPN
    top-down path combines the three resolutions, and two learned stride-2
    convolutions restore the spatial resolution of ``original_final_feature``.
    """

    def __init__(
        self,
        stage_channels,
        stage_depths,
        output_channels,
        neck_channels=64,
        alpha_init=0.0,
        alpha_learnable=True,
        interpolation_mode="nearest",
    ):
        super(MultiScaleEnhancementNeck, self).__init__()
        if len(stage_channels) != 3 or len(stage_depths) != 3:
            raise ValueError("Stage2/3/4 channels and depths must each have three items.")
        if neck_channels <= 0 or output_channels <= 0:
            raise ValueError("neck_channels and output_channels must be positive.")

        self.stage_channels = tuple(int(value) for value in stage_channels)
        self.stage_depths = tuple(int(value) for value in stage_depths)
        self.output_channels = int(output_channels)
        self.neck_channels = int(neck_channels)
        self.interpolation_mode = interpolation_mode

        # Keep projection and lateral convolution separate: projection only
        # removes the height dimension; lateral layers unify FPN channels.
        self.stage2_projection = SparseToBEVScatterAdapter(
            self.stage_channels[0], self.stage_depths[0], self.stage_channels[0]
        )
        self.stage3_projection = SparseToBEVScatterAdapter(
            self.stage_channels[1], self.stage_depths[1], self.stage_channels[1]
        )
        self.stage4_projection = SparseToBEVScatterAdapter(
            self.stage_channels[2], self.stage_depths[2], self.stage_channels[2]
        )

        self.lateral2 = _conv_bn_relu(self.stage_channels[0], self.neck_channels, 1)
        self.lateral3 = _conv_bn_relu(self.stage_channels[1], self.neck_channels, 1)
        self.lateral4 = _conv_bn_relu(self.stage_channels[2], self.neck_channels, 1)

        self.downsample1 = SpikeResidualConvBlock(
            self.neck_channels, self.neck_channels, stride=2
        )
        self.downsample2 = SpikeResidualConvBlock(
            self.neck_channels, self.neck_channels, stride=2
        )
        self.output_conv = SpikeResidualConvBlock(
            self.neck_channels, self.output_channels, stride=1
        )

        alpha = torch.tensor(float(alpha_init), dtype=torch.float32)
        if alpha_learnable:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

    @staticmethod
    def _sparse_shape(sparse_tensor):
        return (
            int(sparse_tensor.batch_size),
            int(sparse_tensor.features.shape[1]),
            *tuple(int(value) for value in sparse_tensor.spatial_shape),
        )

    def _resize(self, feature, target_size):
        return F.interpolate(
            feature,
            size=target_size,
            mode=self.interpolation_mode,
        )

    def forward(
        self,
        stage2,
        stage3,
        stage4,
        original_final_feature,
        return_debug=False,
    ):
        if original_final_feature.dim() != 4:
            raise ValueError("original_final_feature must be [B, C, H, W].")
        if original_final_feature.shape[1] != self.output_channels:
            raise ValueError(
                "Expected original_final_feature to have {} channels, got {}.".format(
                    self.output_channels, original_final_feature.shape[1]
                )
            )

        f2 = self.stage2_projection(stage2)
        f3 = self.stage3_projection(stage3)
        f4 = self.stage4_projection(stage4)

        l2 = self.lateral2(f2)
        l3 = self.lateral3(f3)
        l4 = self.lateral4(f4)

        p4 = l4
        p3 = l3 + self._resize(p4, l3.shape[-2:])
        p2 = l2 + self._resize(p3, l2.shape[-2:])

        p_down1 = self.downsample1(p2)
        p_down2 = self.downsample2(p_down1)
        target_size = original_final_feature.shape[-2:]
        if p_down2.shape[-2:] != target_size:
            p_down2 = self._resize(p_down2, target_size)

        f_ms = self.output_conv(p_down2)
        if f_ms.shape != original_final_feature.shape:
            raise RuntimeError(
                "Multi-scale feature shape {} does not match original BEV shape {}.".format(
                    tuple(f_ms.shape), tuple(original_final_feature.shape)
                )
            )
        f_enhanced = original_final_feature + self.alpha * f_ms
        if f_enhanced.shape != original_final_feature.shape:
            raise RuntimeError("Enhanced BEV feature changed the backbone interface shape.")

        debug = {}
        if return_debug:
            debug = {
                "stage2_sparse": self._sparse_shape(stage2),
                "stage3_sparse": self._sparse_shape(stage3),
                "stage4_sparse": self._sparse_shape(stage4),
                "stage2_active_features": tuple(stage2.features.shape),
                "stage3_active_features": tuple(stage3.features.shape),
                "stage4_active_features": tuple(stage4.features.shape),
                "stage2_bev": tuple(f2.shape),
                "stage3_bev": tuple(f3.shape),
                "stage4_bev": tuple(f4.shape),
                "l2": tuple(l2.shape),
                "l3": tuple(l3.shape),
                "l4": tuple(l4.shape),
                "p3": tuple(p3.shape),
                "p2": tuple(p2.shape),
                "p_down1": tuple(p_down1.shape),
                "p_down2": tuple(p_down2.shape),
                "f_ms": tuple(f_ms.shape),
                "original_final_feature": tuple(original_final_feature.shape),
                "f_enhanced": tuple(f_enhanced.shape),
                "alpha": float(self.alpha.detach().cpu()),
            }
        return f_enhanced, debug
