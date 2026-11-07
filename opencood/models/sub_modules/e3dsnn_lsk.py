# -*- coding: utf-8 -*-
"""Large Selective Kernel feature enhancement for E-3DSNN BEV features."""

import torch
import torch.nn as nn


class LSKBlock(nn.Module):
    """The spatial selective-kernel mechanism from LSKNet.

    This block accepts and returns a dense ``[A, C, H, W]`` BEV tensor.  It is
    intentionally independent of MMCV/MMRotate and contains no outer residual.
    """

    def __init__(self, channels, spatial_dilation=3):
        super(LSKBlock, self).__init__()
        channels = int(channels)
        spatial_dilation = int(spatial_dilation)
        if channels <= 0 or channels % 2:
            raise ValueError("LSKBlock channels must be a positive even integer.")
        if spatial_dilation <= 0:
            raise ValueError("spatial_dilation must be positive.")

        hidden_channels = channels // 2
        self.channels = channels
        self.conv0 = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels
        )
        self.conv_spatial = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            stride=1,
            padding=3 * spatial_dilation,
            dilation=spatial_dilation,
            groups=channels,
        )
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.conv_squeeze = nn.Conv2d(2, 2, kernel_size=7, padding=3)
        self.conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != self.channels:
            raise ValueError(
                "Expected LSK input [A, {}, H, W], got {}.".format(
                    self.channels, tuple(x.shape)
                )
            )
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        branch1 = self.conv1(attn1)
        branch2 = self.conv2(attn2)
        branches = torch.cat((branch1, branch2), dim=1)
        spatial_mean = branches.mean(dim=1, keepdim=True)
        spatial_max = branches.amax(dim=1, keepdim=True)
        weights = torch.sigmoid(
            self.conv_squeeze(torch.cat((spatial_mean, spatial_max), dim=1))
        )
        selected = branch1 * weights[:, 0:1] + branch2 * weights[:, 1:2]
        attn = self.conv(selected)
        return x * attn


class LSKResidualAdapter(nn.Module):
    """Small-gamma residual adapter suitable for old-checkpoint fine-tuning."""

    def __init__(self, channels, gamma_init=1e-3, spatial_dilation=3):
        super(LSKResidualAdapter, self).__init__()
        channels = int(channels)
        self.channels = channels
        self.input_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.lsk = LSKBlock(channels, spatial_dilation=spatial_dilation)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        transformed = self.input_proj(x)
        transformed = self.activation(transformed)
        transformed = self.lsk(transformed)
        transformed = self.output_proj(transformed)
        return x + self.gamma * transformed
