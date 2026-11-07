# -*- coding: utf-8 -*-
"""Sparse tensor to BEV tensor adapters for E-3DSNN local encoders."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseToBEVAdapter(nn.Module):
    """Collapse spconv dense output from [B, C, D, H, W] to [B, C*D, H, W]."""

    def __init__(self, input_channels, output_channels):
        super(SparseToBEVAdapter, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, sparse_tensor):
        dense = sparse_tensor.dense()
        if dense.dim() == 5:
            batch_size, channels, depth, height, width = dense.shape
            dense = dense.view(batch_size, channels * depth, height, width)
        elif dense.dim() != 4:
            raise ValueError(
                "Unexpected sparse tensor dense shape {}.".format(tuple(dense.shape))
            )
        return self.project(dense)


class SparseToBEVScatterAdapter(nn.Module):
    """Memory-efficient equivalent of dense CxZ collapse plus a 1x1 conv.

    A dense adapter applies an independent weight to every input channel and
    height plane after reshaping ``[B, C, D, H, W]`` to ``[B, C*D, H, W]``.
    This module keeps exactly that learnable mapping but evaluates it only on
    active sparse voxels, then sums projected voxels into their BEV cells.  It
    therefore avoids materializing the very large Stage2 five-dimensional
    dense tensor.
    """

    def __init__(self, input_channels, input_depth, output_channels):
        super(SparseToBEVScatterAdapter, self).__init__()
        self.input_channels = int(input_channels)
        self.input_depth = int(input_depth)
        self.output_channels = int(output_channels)
        self.weight = nn.Parameter(
            torch.empty(
                self.output_channels,
                self.input_channels,
                self.input_depth,
            )
        )
        self.bn = nn.BatchNorm2d(
            self.output_channels, eps=1e-3, momentum=0.01
        )
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        # Match nn.Conv2d(..., bias=False)'s default initialization with C*D
        # effective input channels.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, sparse_tensor):
        features = sparse_tensor.features
        indices = sparse_tensor.indices.long()
        if features.dim() != 2 or features.shape[1] != self.input_channels:
            raise ValueError(
                "Expected sparse features [N, {}], got {}.".format(
                    self.input_channels, tuple(features.shape)
                )
            )
        depth, height, width = tuple(
            int(value) for value in sparse_tensor.spatial_shape
        )
        if depth != self.input_depth:
            raise ValueError(
                "Expected sparse depth {}, got {}.".format(
                    self.input_depth, depth
                )
            )

        batch_size = int(sparse_tensor.batch_size)
        output_flat = features.new_zeros(
            (batch_size * height * width, self.output_channels)
        )
        flat_bev_indices = (
            indices[:, 0] * (height * width)
            + indices[:, 2] * width
            + indices[:, 3]
        )

        # Slicing by Z avoids gathering an [N, Cout, Cin] weight tensor while
        # remaining algebraically identical to Conv2d(Cin*D, Cout, 1).
        for z_index in range(depth):
            active = indices[:, 1] == z_index
            projected = F.linear(
                features[active], self.weight[:, :, z_index]
            )
            output_flat.index_add_(0, flat_bev_indices[active], projected)

        output = output_flat.view(
            batch_size, height, width, self.output_channels
        ).permute(0, 3, 1, 2).contiguous()
        return self.relu(self.bn(output))
