"""Torch-only SDT V3 primitives for spatial-patch value prediction.

Reference: BICLab/Spike-Driven-Transformer-V3, revision
00fcd1fd5d9c65fe9a26579ed512a20c6a303deb,
SDT_V3/Classification/Model_Base/models.py (Quant, MultiSpike,
SepConv_Spike, MS_ConvBlock_spike_SepConv, MS_MLP,
MS_Attention_linear, MS_Block_Spike_SepConv).

Branches start with spikes and end with continuous membrane currents.
Only the owning block adds residuals. No post-add spike or LayerNorm.
This is stateless SFA simulation, not a stateful LIF time-step simulator.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _SpikeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, maximum):
        ctx.save_for_backward(x)
        ctx.maximum = maximum
        return x.clamp(0, maximum).round()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Inclusive STE window matches the fixed upstream reference.
        return grad_output.masked_fill((x < 0) | (x > ctx.maximum), 0), None


class MultiSpike(nn.Module):
    def __init__(self, spike_max=4, spike_norm=4.0):
        super().__init__()
        if int(spike_max) != spike_max or spike_max <= 0:
            raise ValueError("spike_max must be a positive integer")
        if not math.isfinite(spike_norm) or spike_norm <= 0:
            raise ValueError("spike_norm must be finite and positive")
        self.spike_max = int(spike_max)
        self.spike_norm = float(spike_norm)
        self.record_stats = False
        self.last_stats = None

    def forward(self, x):
        # Division also scales the STE derivative by 1/spike_norm.
        result = _SpikeQuantize.apply(x, self.spike_max) / self.spike_norm
        if self.record_stats:
            with torch.no_grad():
                self.last_stats = {
                    "zero_fraction": (result == 0).float().mean(),
                    "saturation_fraction": (result == self.spike_max / self.spike_norm).float().mean(),
                    "mean_spike_count": result.float().mean() * self.spike_norm,
                }
        return result


class SingletonSafeBatchNorm1d(nn.BatchNorm1d):
    """Use running statistics for a single metadata observation in train mode.

No fake spatial copies are used to estimate metadata variance. Affine weights
still receive gradients; normal batches update the usual BN statistics.
"""
    def forward(self, x):
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


def _conv_bn(cin, cout, kernel=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, padding=kernel // 2,
                  groups=groups, bias=False),
        nn.BatchNorm2d(cout),
    )


class SpikeSepConv(nn.Module):
    """S -> PW/BN -> S -> DW/BN -> S -> PW/BN; no residual inside."""
    def __init__(self, dim, kernel=7, expansion=2, spike_max=4, spike_norm=4.0):
        super().__init__()
        hidden = dim * expansion
        self.spike1 = MultiSpike(spike_max, spike_norm)
        self.pwconv1 = _conv_bn(dim, hidden)
        self.spike2 = MultiSpike(spike_max, spike_norm)
        self.dwconv = _conv_bn(hidden, hidden, kernel, hidden)
        self.spike3 = MultiSpike(spike_max, spike_norm)
        self.pwconv2 = _conv_bn(hidden, dim)

    def forward(self, u):
        u = self.pwconv1(self.spike1(u))
        u = self.dwconv(self.spike2(u))
        return self.pwconv2(self.spike3(u))


class SpikeConvBlock(nn.Module):
    """Full Conv-based SNN block with TWO sequential membrane residuals."""
    def __init__(self, dim, ratio=4, kernel=7, expansion=2,
                 spike_max=4, spike_norm=4.0):
        super().__init__()
        self.sepconv = SpikeSepConv(dim, kernel, expansion, spike_max, spike_norm)
        self.channel_conv = nn.Sequential(
            MultiSpike(spike_max, spike_norm),
            _conv_bn(dim, dim * ratio, 3),
            MultiSpike(spike_max, spike_norm),
            _conv_bn(dim * ratio, dim, 3),
        )

    def forward(self, u):
        # The identity always receives the unquantized current membrane.
        u = u + self.sepconv(u)
        u = u + self.channel_conv(u)
        return u


class SpikeProjectionMLP(nn.Module):
    """Dimension-changing S/Conv1d/BN/S/Conv1d/BN; NO residual.

Input [B,Cin,N], output [B,Cout,N]. Also works for metadata with N=1.
Unlike upstream MS_MLP, reshape is not hard-coded to the input channels.
"""
    def __init__(self, cin, hidden, cout, spike_max=4, spike_norm=4.0):
        super().__init__()
        self.spike1 = MultiSpike(spike_max, spike_norm)
        self.fc1 = nn.Conv1d(cin, hidden, 1)
        self.bn1 = SingletonSafeBatchNorm1d(hidden)
        self.spike2 = MultiSpike(spike_max, spike_norm)
        self.fc2 = nn.Conv1d(hidden, cout, 1)
        self.bn2 = SingletonSafeBatchNorm1d(cout)

    def forward(self, u):
        u = self.bn1(self.fc1(self.spike1(u)))
        return self.bn2(self.fc2(self.spike2(u)))


class SpikeChannelMLP(SpikeProjectionMLP):
    """Same-dimension channel MLP; the Transformer owns its residual."""
    def __init__(self, dim, ratio=4, spike_max=4, spike_norm=4.0):
        super().__init__(dim, dim * ratio, dim, spike_max, spike_norm)

    def forward(self, u):
        shape = u.shape
        return super().forward(u.flatten(2)).reshape(shape)


class SpikePatchAttention(nn.Module):
    """SDTv3 MS_Attention_linear with SPATIAL PATCHES as the attention axis.

Q/K have D channels, V has v_expansion*D. The softmax-free upstream
expression is ((Qs @ Ks.T) @ Vs) * (2/sqrt(D/heads)). kv_first reassociates
the products; it introduces no intermediate spike, normalization or mask.
FP32 accumulation avoids overflow before the attention spike under AMP.
"""
    def __init__(self, dim, heads=4, v_expansion=4, spike_max=4,
                 spike_norm=4.0, attention_impl="kv_first", qkv_bn_init=0.5):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by attention heads")
        if attention_impl not in ("kv_first", "qk_first"):
            raise ValueError("attention_impl must be kv_first or qk_first")
        self.dim, self.heads = dim, heads
        self.value_dim = dim * v_expansion
        self.attention_impl = attention_impl
        self.scale = 2.0 / math.sqrt(dim // heads)
        self.head_spike = MultiSpike(spike_max, spike_norm)
        self.q_conv = _conv_bn(dim, dim)
        self.k_conv = _conv_bn(dim, dim)
        self.v_conv = _conv_bn(dim, self.value_dim)
        self.q_spike = MultiSpike(spike_max, spike_norm)
        self.k_spike = MultiSpike(spike_max, spike_norm)
        self.v_spike = MultiSpike(spike_max, spike_norm)
        self.attn_spike = MultiSpike(spike_max, spike_norm)
        self.proj_conv = _conv_bn(self.value_dim, dim)
        # Initializer only: lowers initial firing density for N=1050.
        # Upstream operator/scaling and trainable BN are retained.
        for conv in (self.q_conv, self.k_conv, self.v_conv):
            nn.init.constant_(conv[1].weight, qkv_bn_init)

    def _tokens(self, x):
        b, c, h, w = x.shape
        return x.reshape(b, self.heads, c // self.heads, h * w).transpose(2, 3)

    def forward(self, u):
        b, _, h, w = u.shape
        spikes = self.head_spike(u)
        q = self._tokens(self.q_spike(self.q_conv(spikes)))
        k = self._tokens(self.k_spike(self.k_conv(spikes)))
        v = self._tokens(self.v_spike(self.v_conv(spikes)))
        # Q/K: [B,heads,P,D/heads]; V: [B,heads,P,value_dim/heads].
        with torch.autocast(device_type=u.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            if self.attention_impl == "kv_first":
                context = q @ (k.transpose(-2, -1) @ v)
            else:
                context = (q @ k.transpose(-2, -1)) @ v
            context = context * self.scale
        # [B,heads,P,Dv/heads] -> [B,Dv,grid_h,grid_w].
        context = context.transpose(2, 3).reshape(b, self.value_dim, h, w)
        return self.proj_conv(self.attn_spike(context))


class SpikeTransformerBlock(nn.Module):
    """THREE sequential membrane residuals; no spike/LN after an addition."""
    def __init__(self, dim, heads=4, ratio=4, v_expansion=4, expansion=2,
                 spike_max=4, spike_norm=4.0, attention_impl="kv_first",
                 qkv_bn_init=0.5):
        super().__init__()
        self.sepconv = SpikeSepConv(dim, 3, expansion, spike_max, spike_norm)
        self.attention = SpikePatchAttention(
            dim, heads, v_expansion, spike_max, spike_norm,
            attention_impl, qkv_bn_init,
        )
        self.channel_mlp = SpikeChannelMLP(dim, ratio, spike_max, spike_norm)

    def forward(self, u):
        u = u + self.sepconv(u)
        u = u + self.attention(u)
        u = u + self.channel_mlp(u)
        return u


class SignedInputCurrent(nn.Module):
    """Fixed-scale ON/OFF input coding; keeps signed metadata/coordinates.

Each external channel becomes two internal current channels before the first
spike node. Scales are physical ranges (or training-calibrated ranges), NOT
statistics recomputed from the inference batch.
"""
    def __init__(self, scales, spike_max=4):
        super().__init__()
        scale = torch.tensor(scales, dtype=torch.float32)
        if scale.ndim != 1 or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Signed input scales must be finite positive values")
        self.register_buffer("scales", scale.reshape(1, -1, 1))
        self.spike_max = spike_max

    def forward(self, x):
        # [B,C,N] -> [B,2C,N]; actual quantization is in the following MLP.
        x = x / self.scales.to(dtype=x.dtype)
        return torch.cat((x.clamp_min(0), (-x).clamp_min(0)), dim=1) * self.spike_max
