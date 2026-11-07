# -*- coding: utf-8 -*-
"""E-3DSNN sparse spike backbone adapted to OpenCOOD.

The sparse 3D encoder mirrors E-3DSNN's ``VoxelBackBone8x_3dv_snn`` structure:
``conv_input -> conv1 -> DonwBlock(conv2/3/4) -> conv_out``.  A small BEV neck
is kept outside that encoder only to expose the OpenCOOD ``blocks/deblocks``
interface used by Where2comm fusion.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.e3dsnn_bev_adapter import SparseToBEVAdapter
from opencood.models.sub_modules.e3dsnn_multiscale_enhancement_neck import (
    MultiScaleEnhancementNeck,
)


def _load_spconv():
    try:
        import spconv.pytorch as spconv

        return spconv
    except ImportError:
        try:
            import spconv

            return spconv
        except ImportError as exc:
            raise ImportError(
                "E3DSNNSparseBEVBackbone requires spconv. Install a compatible "
                "spconv/cumm build before using this model."
            ) from exc


spconv = _load_spconv()


def replace_feature(sparse_tensor, features):
    if hasattr(sparse_tensor, "replace_feature"):
        return sparse_tensor.replace_feature(features)
    sparse_tensor.features = features
    return sparse_tensor


class ReLUX(nn.Module):
    def __init__(self, threshold=4.0):
        super(ReLUX, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.clamp(x, 0.0, self.threshold)


class MultiSpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lens=4.0):
        ctx.save_for_backward(x)
        ctx.lens = lens
        return torch.floor(torch.clamp(x, 0.0, lens) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        surrogate = ((x > 0.0) & (x < ctx.lens)).to(grad_output.dtype)
        return grad_output * surrogate, None


class MultiSpike(nn.Module):
    def __init__(self, lens=4.0):
        super(MultiSpike, self).__init__()
        self.lens = lens

    def forward(self, x):
        return MultiSpikeFunction.apply(x, self.lens)


def post_act_block(
    in_channels,
    out_channels,
    kernel_size,
    indice_key=None,
    stride=1,
    padding=0,
    conv_type="subm",
    norm_fn=None,
):
    if conv_type == "subm":
        conv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size, padding=padding,
            bias=False, indice_key=indice_key
        )
    elif conv_type == "spconv":
        conv = spconv.SparseConv3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=False, indice_key=indice_key
        )
    else:
        raise NotImplementedError("Unsupported sparse conv type '{}'.".format(conv_type))

    return spconv.SparseSequential(MultiSpike(), conv, norm_fn(out_channels))


class BasicBlock(nn.Module):
    """E-3DSNN sparse voxel residual block used inside DonwBlock."""

    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_fn=None,
        indice_key=None,
        depth=4,
        groups=None,
        grid_size=None,
        bias=False,
    ):
        super(BasicBlock, self).__init__()
        assert groups is None or embed_channels % groups == 0
        self.groups = groups
        self.embed_channels = embed_channels
        self.proj = nn.ModuleList()
        self.grid_size = grid_size
        self.block = spconv.SparseSequential(
            MultiSpike(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
        )
        self.voxel_block = spconv.SparseSequential(
            MultiSpike(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
            MultiSpike(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
        )

    def forward(self, x):
        feat = self.block(x)
        feat = replace_feature(feat, feat.features + x.features)
        res = feat
        out = self.voxel_block(feat)
        out = replace_feature(out, out.features + res.features)
        return out


class DonwBlock(nn.Module):
    """Keep E-3DSNN's original class name and sparse stride-2 downsampling."""

    def __init__(
        self,
        in_channels,
        embed_channels,
        depth,
        sp_indice_key,
        point_grid_size,
        num_ref=16,
        groups=None,
        norm_fn=None,
        sub_indice_key=None,
    ):
        super(DonwBlock, self).__init__()
        self.num_ref = num_ref
        self.depth = depth
        self.point_grid_size = point_grid_size
        self.down = spconv.SparseSequential(
            MultiSpike(),
            spconv.SparseConv3d(
                in_channels,
                embed_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(embed_channels),
        )
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                BasicBlock(
                    in_channels=embed_channels,
                    embed_channels=embed_channels,
                    depth=len(point_grid_size) + 1,
                    groups=groups,
                    grid_size=point_grid_size,
                    norm_fn=norm_fn,
                    indice_key=sub_indice_key,
                )
            )

    def forward(self, x):
        x = self.down(x)
        for block in self.blocks:
            x = block(x)
        return x


class SparseSpikeEncoder(nn.Module):
    """E-3DSNN VoxelBackBone8x_3dv_snn without debug print/exit side effects."""

    def __init__(self, model_cfg, input_channels, grid_size):
        super(SparseSpikeEncoder, self).__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        channels = model_cfg.get("sparse_channels", [16, 32, 64, 64])
        self.sparse_shape = (
            np.array(grid_size[::-1], dtype=np.int64) + np.array([1, 0, 0])
        ).tolist()
        self.in_channels = input_channels
        self.num_stages = 4
        self.embed_channels = model_cfg.get("embed_channels", 16)
        self.out_channels = model_cfg.get("out_channels", 128)
        enc_depth = model_cfg.get("enc_depth", [2, 2, 2, 2])
        groups = model_cfg.get("groups", [4, 4, 8, 16])
        enc_num_ref = model_cfg.get("enc_num_ref", [16, 16, 16, 16])
        point_grid_size = model_cfg.get(
            "point_grid_size",
            [[8, 12, 16, 16], [6, 9, 12, 12], [4, 6, 8, 8], [3, 4, 6, 6]],
        )

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channels,
                self.embed_channels,
                3,
                padding=1,
                bias=False,
                indice_key="subm1",
            ),
            norm_fn(self.embed_channels),
            MultiSpike(),
            spconv.SubMConv3d(
                self.embed_channels,
                self.embed_channels,
                3,
                padding=1,
                bias=False,
                indice_key="subm1",
            ),
            norm_fn(self.embed_channels),
            MultiSpike(),
            spconv.SubMConv3d(
                self.embed_channels,
                self.embed_channels,
                3,
                padding=1,
                bias=False,
                indice_key="subm1",
            ),
            norm_fn(self.embed_channels),
        )

        self.conv1 = spconv.SparseSequential(
            post_act_block(
                self.embed_channels,
                channels[0],
                3,
                norm_fn=norm_fn,
                padding=1,
                indice_key="subm1",
            )
        )
        self.conv2 = DonwBlock(
            in_channels=channels[0],
            embed_channels=channels[1],
            depth=enc_depth[1],
            norm_fn=norm_fn,
            groups=groups[1],
            point_grid_size=point_grid_size[1],
            num_ref=enc_num_ref[1],
            sp_indice_key="spconv2",
            sub_indice_key="subm2",
        )
        self.conv3 = DonwBlock(
            in_channels=channels[1],
            embed_channels=channels[2],
            depth=enc_depth[2],
            norm_fn=norm_fn,
            groups=groups[2],
            point_grid_size=point_grid_size[2],
            num_ref=enc_num_ref[2],
            sp_indice_key="spconv3",
            sub_indice_key="subm3",
        )
        self.conv4 = DonwBlock(
            in_channels=channels[2],
            embed_channels=channels[3],
            depth=enc_depth[3],
            norm_fn=norm_fn,
            groups=groups[3],
            point_grid_size=point_grid_size[3],
            num_ref=enc_num_ref[3],
            sp_indice_key="spconv4",
            sub_indice_key="subm4",
        )

        last_pad = model_cfg.get("last_pad", 0)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(
                channels[3],
                self.out_channels,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=last_pad,
                bias=False,
                indice_key="spconv_down2",
            ),
            norm_fn(self.out_channels),
            MultiSpike(),
        )
        self.num_point_features = self.out_channels
        self.backbone_channels = {
            "x_conv1": channels[0],
            "x_conv2": channels[1],
            "x_conv3": channels[2],
            "x_conv4": channels[3],
        }
        self.stage_spatial_shapes = self._stage_spatial_shapes()
        self.encoded_depth = self._encoded_depth()

    @staticmethod
    def _conv_output_size(size, kernel_size, stride, padding):
        return int(np.floor((size + 2 * padding - kernel_size) / stride) + 1)

    def _encoded_depth(self):
        depth = int(self.sparse_shape[0])
        depth = self._conv_output_size(depth, 3, 2, 1)
        depth = self._conv_output_size(depth, 3, 2, 1)
        depth = self._conv_output_size(depth, 3, 2, 1)
        depth = self._conv_output_size(depth, 3, 2, self.model_cfg.get("last_pad", 0))
        return depth

    def _stage_spatial_shapes(self):
        shape = tuple(int(value) for value in self.sparse_shape)
        shapes = [shape]
        for _ in range(3):
            shape = tuple(
                self._conv_output_size(value, 3, 2, 1) for value in shape
            )
            shapes.append(shape)
        return {
            "x_conv1": shapes[0],
            "x_conv2": shapes[1],
            "x_conv3": shapes[2],
            "x_conv4": shapes[3],
        }

    def forward(self, batch_dict):
        input_sp_tensor = spconv.SparseConvTensor(
            features=batch_dict["voxel_features"].float(),
            indices=batch_dict["voxel_coords"].int(),
            spatial_shape=self.sparse_shape,
            batch_size=int(batch_dict["batch_size"]),
        )
        x = self.conv_input(input_sp_tensor)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = self.conv_out(x4)
        batch_dict.update(
            {
                "encoded_spconv_tensor": out,
                "encoded_spconv_tensor_stride": 8,
                "multi_scale_3d_features": {
                    "x_conv1": x1,
                    "x_conv2": x2,
                    "x_conv3": x3,
                    "x_conv4": x4,
                },
                "multi_scale_3d_strides": {
                    "x_conv1": 1,
                    "x_conv2": 2,
                    "x_conv3": 4,
                    "x_conv4": 8,
                },
            }
        )
        return batch_dict


class E3DSNNSparseBEVBackbone(nn.Module):
    """Sparse spike local encoder plus BEV neck compatible with Where2comm."""

    def __init__(self, model_cfg, input_channels, grid_size):
        super(E3DSNNSparseBEVBackbone, self).__init__()
        self.model_cfg = model_cfg
        self.sparse_encoder = SparseSpikeEncoder(model_cfg, input_channels, grid_size)
        sparse_depth = int(self.sparse_encoder.encoded_depth)
        self.bev_in_channels = model_cfg.get(
            "bev_in_channels", self.sparse_encoder.out_channels
        )
        self.bev_adapter = SparseToBEVAdapter(
            self.sparse_encoder.out_channels * sparse_depth,
            self.bev_in_channels,
        )
        self.blocks, self.deblocks, self.num_bev_features = self._build_bev_neck(
            model_cfg, self.bev_in_channels
        )
        enhancement_cfg = model_cfg.get("multi_scale_enhancement", {})
        self.multi_scale_enhancement_enabled = bool(
            enhancement_cfg.get("enabled", False)
        )
        self.multi_scale_enhancement_debug = bool(
            enhancement_cfg.get("debug_shapes", False)
        )
        if self.multi_scale_enhancement_enabled:
            stage_channels = [
                self.sparse_encoder.backbone_channels["x_conv2"],
                self.sparse_encoder.backbone_channels["x_conv3"],
                self.sparse_encoder.backbone_channels["x_conv4"],
            ]
            stage_depths = [
                self.sparse_encoder.stage_spatial_shapes[name][0]
                for name in ("x_conv2", "x_conv3", "x_conv4")
            ]
            self.multi_scale_neck = MultiScaleEnhancementNeck(
                stage_channels=stage_channels,
                stage_depths=stage_depths,
                output_channels=self.bev_in_channels,
                neck_channels=int(enhancement_cfg.get("neck_channels", 64)),
                alpha_init=float(enhancement_cfg.get("alpha_init", 0.0)),
                alpha_learnable=bool(
                    enhancement_cfg.get("alpha_learnable", True)
                ),
                interpolation_mode=enhancement_cfg.get(
                    "interpolation_mode", "nearest"
                ),
            )

    @staticmethod
    def _build_bev_neck(model_cfg, input_channels):
        layer_nums = model_cfg.get("layer_nums", [1, 2, 6])
        layer_strides = model_cfg.get("layer_strides", [2, 2, 2])
        num_filters = model_cfg.get("num_filters", [64, 128, 256])
        upsample_strides = model_cfg.get("upsample_strides", [1, 2, 4])
        num_upsample_filter = model_cfg.get("num_upsample_filter", [128, 128, 128])

        assert len(layer_nums) == len(layer_strides) == len(num_filters)
        assert len(upsample_strides) == len(num_upsample_filter)

        blocks = nn.ModuleList()
        deblocks = nn.ModuleList()
        c_in_list = [input_channels, *num_filters[:-1]]

        for idx, layer_num in enumerate(layer_nums):
            block_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], 3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_num):
                block_layers.extend(
                    [
                        nn.Conv2d(
                            num_filters[idx], num_filters[idx], 3,
                            padding=1, bias=False
                        ),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True),
                    ]
                )
            blocks.append(nn.Sequential(*block_layers))

            stride = upsample_strides[idx]
            if stride >= 1:
                deblocks.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filter[idx],
                            stride, stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(
                            num_upsample_filter[idx], eps=1e-3, momentum=0.01
                        ),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                stride = int(np.round(1 / stride))
                deblocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filter[idx],
                            stride, stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(
                            num_upsample_filter[idx], eps=1e-3, momentum=0.01
                        ),
                        nn.ReLU(inplace=True),
                    )
                )

        return blocks, deblocks, sum(num_upsample_filter)

    def _sparse_to_bev(self, sparse_tensor):
        return self.bev_adapter(sparse_tensor)

    def forward(self, data_dict):
        batch_dict = self.sparse_encoder(data_dict)
        spatial_features = self._sparse_to_bev(batch_dict["encoded_spconv_tensor"])
        batch_dict["original_spatial_features"] = spatial_features
        if self.multi_scale_enhancement_enabled:
            sparse_features = batch_dict["multi_scale_3d_features"]
            spatial_features, neck_debug = self.multi_scale_neck(
                sparse_features["x_conv2"],
                sparse_features["x_conv3"],
                sparse_features["x_conv4"],
                spatial_features,
                return_debug=self.multi_scale_enhancement_debug,
            )
            if self.multi_scale_enhancement_debug:
                batch_dict["multi_scale_neck_debug"] = neck_debug
        batch_dict["spatial_features"] = spatial_features

        ups = []
        x = spatial_features
        for idx, block in enumerate(self.blocks):
            x = block(x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            batch_dict["spatial_features_%dx" % stride] = x
            ups.append(self.deblocks[idx](x) if len(self.deblocks) > 0 else x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        batch_dict["spatial_features_2d"] = x
        return batch_dict
