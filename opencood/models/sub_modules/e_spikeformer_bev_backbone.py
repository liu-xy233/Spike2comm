import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from opencood.tools.monitor_spike import print_value_bins


class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value=0, max_value=4):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        (i,) = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None


class MultiSpikeNorm4(nn.Module):
    def __init__(self, Vth=1.0, T=4.0):
        super().__init__()
        self.spike = Quant()
        self.Vth = Vth
        self.T = T

    def forward(self, x):
        if self.training:
            return self.spike.apply(x) / self.T
        return torch.clamp(x, min=0, max=self.T).round_() / self.T


class SepConvSpike(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
        T=None,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = MultiSpikeNorm4(T=T)
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels),
        )
        self.spike2 = MultiSpikeNorm4(T=T)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                med_channels,
                med_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=med_channels,
                bias=bias,
            ),
            nn.BatchNorm2d(med_channels),
        )
        self.spike3 = MultiSpikeNorm4(T=T)
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = self.spike1(x)
        x = self.pwconv1(x)
        x = self.spike2(x)
        x = self.dwconv(x)
        x = self.spike3(x)
        x = self.pwconv2(x)
        return x


class MSConvBlockSpikeSepConv(nn.Module):
    def __init__(self, dim, mlp_ratio=4, T=None):
        super().__init__()
        self.Conv = SepConvSpike(dim=dim, T=T)
        self.mlp_ratio = mlp_ratio
        self.spike1 = MultiSpikeNorm4(T=T)
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)
        self.spike2 = MultiSpikeNorm4(T=T)
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x)).reshape(B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)
        x = self.bn2(self.conv2(x)).reshape(B, C, H, W)
        x = x_feat + x

        return x


class MSMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        layer=0,
        T=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = MultiSpikeNorm4(T=T)

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = MultiSpikeNorm4(T=T)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2)
        x = self.fc1_spike(x)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).reshape(B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(B, C, H, W).contiguous()

        return x


class MSAttentionLinear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        T=None,
        lamda_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} should be divided by num_heads {num_heads}."
        )
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio

        self.head_spike = MultiSpikeNorm4(T=T)
        self.q_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )
        self.q_spike = MultiSpikeNorm4(T=T)
        self.k_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )
        self.k_spike = MultiSpikeNorm4(T=T)
        self.v_conv = nn.Sequential(
            nn.Conv2d(dim, int(dim * lamda_ratio), 1, 1, bias=False),
            nn.BatchNorm2d(int(dim * lamda_ratio)),
        )
        self.v_spike = MultiSpikeNorm4(T=T)
        self.attn_spike = MultiSpikeNorm4(T=T)
        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim * lamda_ratio, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        C_v = int(C * self.lamda_ratio)

        x = self.head_spike(x)

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = self.q_spike(q)
        q = q.flatten(2)
        q = (
            q.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        k = self.k_spike(k)
        k = k.flatten(2)
        k = (
            k.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        v = self.v_spike(v)
        v = v.flatten(2)
        v = (
            v.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C_v // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        x = q @ k.transpose(-2, -1)
        x = (x @ v) * (self.scale * 2)

        x = x.transpose(2, 3).reshape(B, C_v, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(B, C_v, H, W)
        x = self.proj_conv(x).reshape(B, C, H, W)

        return x


class MSBlockSpikeSepConv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        T=None,
    ):
        super().__init__()
        self.conv = SepConvSpike(dim=dim, kernel_size=3, padding=1, T=T)
        self.attn = MSAttentionLinear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            T=T,
            lamda_ratio=4,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MSMLP(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, T=T
        )

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class MSDownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        T=None,
    ):
        super().__init__()
        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = MultiSpikeNorm4(T=T)

    def forward(self, x):
        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x)
        x = self.encode_bn(x)
        return x


class ESpikeFormerBEVStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        depth,
        block_type,
        num_heads,
        mlp_ratio,
        qkv_bias,
        drop_rate,
        attn_drop_rate,
        drop_path_rates,
        sr_ratio,
        first_layer,
        T,
        stage_idx=None,
    ):
        super().__init__()
        self.stage_idx = stage_idx
        layers = [
            MSDownSampling(
                in_channels=in_channels,
                embed_dims=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                first_layer=first_layer,
                T=T,
            )
        ]
        for idx in range(depth):
            drop_path = drop_path_rates[idx] if idx < len(drop_path_rates) else 0.0
            if block_type == "attention":
                layers.append(
                    MSBlockSpikeSepConv(
                        dim=out_channels,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path,
                        sr_ratio=sr_ratio,
                        T=T,
                    )
                )
            else:
                layers.append(
                    MSConvBlockSpikeSepConv(
                        dim=out_channels,
                        mlp_ratio=mlp_ratio,
                        T=T,
                    )
                )
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        for idx, layer in enumerate(self.stage):
            x = layer(x)
            if idx == 0:
                continue

            block_idx = idx - 1
            if self.stage_idx is None:
                monitor_name = f"ESpikeFormerBEVStage.layer_{block_idx}"
            else:
                monitor_name = (
                    f"ESpikeFormerBEVStage[{self.stage_idx}].layer_{block_idx}"
                )
            # print_value_bins(monitor_name, x)
        return x


class ESpikeFormerBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        layer_nums = model_cfg.get("layer_nums", [1, 2, 6])
        layer_strides = model_cfg.get("layer_strides", [2, 2, 2])
        num_filters = model_cfg.get("num_filters", [64, 128, 256])
        upsample_strides = model_cfg.get("upsample_strides", [1, 2, 4])
        num_upsample_filters = model_cfg.get("num_upsample_filter", [128, 128, 128])
        stage_types = model_cfg.get("stage_types", ["conv", "conv", "attention"])

        assert len(layer_nums) == len(layer_strides) == len(num_filters)
        assert len(upsample_strides) == len(num_upsample_filters)

        self.T = model_cfg.get("T", 4)
        self.num_levels = len(layer_nums)
        mlp_ratio = model_cfg.get("mlp_ratios", model_cfg.get("mlp_ratio", 4))
        num_heads_cfg = model_cfg.get("num_heads", 8)
        qkv_bias = model_cfg.get("qkv_bias", False)
        drop_rate = model_cfg.get("drop_rate", 0.0)
        attn_drop_rate = model_cfg.get("attn_drop_rate", 0.0)
        drop_path_rate = model_cfg.get("drop_path_rate", 0.0)
        sr_ratios_cfg = model_cfg.get("sr_ratios", 1)

        total_depth = sum(layer_nums)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.blocks = nn.ModuleList()
        c_in_list = [input_channels, *num_filters[:-1]]
        dpr_offset = 0
        for idx in range(self.num_levels):
            num_heads = (
                num_heads_cfg[idx] if isinstance(num_heads_cfg, list) else num_heads_cfg
            )
            sr_ratio = (
                sr_ratios_cfg[idx] if isinstance(sr_ratios_cfg, list) else sr_ratios_cfg
            )
            block_type = stage_types[idx] if idx < len(stage_types) else "attention"
            stage_drop_rates = dpr[dpr_offset:dpr_offset + layer_nums[idx]]
            dpr_offset += layer_nums[idx]
            self.blocks.append(
                ESpikeFormerBEVStage(
                    in_channels=c_in_list[idx],
                    out_channels=num_filters[idx],
                    stride=layer_strides[idx],
                    depth=layer_nums[idx],
                    block_type=block_type,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rates=stage_drop_rates,
                    sr_ratio=sr_ratio,
                    first_layer=(idx == 0),
                    T=self.T,
                    stage_idx=idx,
                )
            )

        self.deblocks = nn.ModuleList()
        for idx in range(self.num_levels):
            stride = upsample_strides[idx]
            if stride >= 1:
                self.deblocks.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx],
                            num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx],
                            bias=False,
                        ),
                        nn.BatchNorm2d(
                            num_upsample_filters[idx], eps=1e-3, momentum=0.01
                        ),
                        nn.ReLU(),
                    )
                )
            else:
                stride = int(np.round(1 / stride))
                self.deblocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx],
                            num_upsample_filters[idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(
                            num_upsample_filters[idx], eps=1e-3, momentum=0.01
                        ),
                        nn.ReLU(),
                    )
                )

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c_in,
                        c_in,
                        upsample_strides[-1],
                        stride=upsample_strides[-1],
                        bias=False,
                    ),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )

        self.num_bev_features = c_in
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data_dict):
        spatial_features = data_dict["spatial_features"]
        ups = []
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            data_dict["spatial_features_%dx" % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        # TODO
        print_value_bins("ESpikeFormerBEVBackbone", x)
        data_dict["spatial_features_2d"] = x
        return data_dict
