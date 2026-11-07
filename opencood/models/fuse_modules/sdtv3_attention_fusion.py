"""SDTv3 attention with a membrane-valued residual for BEV agent fusion.

Adapted from BICLab/Spike-Driven-Transformer-V3, inspected 2026-10-21:
Upstream revision: 00fcd1fd5d9c65fe9a26579ed512a20c6a303deb.
SDT_V3/Classification/Model_Base/models.py, Quant, MultiSpike,
MS_Attention_linear and the attention residual of MS_Block_Spike_SepConv.
https://github.com/BICLab/Spike-Driven-Transformer-V3

The attention sequence is the agents at ONE aligned BEV location, rather than
the spatial positions of one image. Q/K/V projections, BN, five spike nodes,
expanded V, the softmax-free product and output projection are retained.
No temporal state, SepConv or MLP is introduced by this fusion module.
"""

import math

import torch
from torch import nn


class _QuantizeSpike(torch.autograd.Function):
    """Rounded, bounded spikes with SDTv3's inclusive straight-through window."""

    @staticmethod
    def forward(ctx, x, min_value, max_value):
        ctx.save_for_backward(x)
        ctx.min_value = min_value
        ctx.max_value = max_value
        return x.clamp(min_value, max_value).round()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        outside = (x < ctx.min_value) | (x > ctx.max_value)
        return grad_output.masked_fill(outside, 0), None, None


class MultiSpike(nn.Module):
    """Stateless SFA activation; the divisor is not a simulation time axis."""

    def __init__(self, max_value=4, norm=None):
        super().__init__()
        if not isinstance(max_value, int) or max_value <= 0:
            raise ValueError("spike_max must be a positive integer")
        self.max_value = max_value
        self.norm = float(max_value if norm is None else norm)
        if not math.isfinite(self.norm) or self.norm <= 0:
            raise ValueError("spike_norm must be finite and positive")

    def forward(self, x):
        return _QuantizeSpike.apply(x, 0, self.max_value) / self.norm

    def extra_repr(self):
        return "max_value={}, norm={}".format(self.max_value, self.norm)


class SpikeDrivenAgentAttention(nn.Module):
    """Complete MS_Attention_linear, with agents as tokens at each BEV cell.

    Input/output: [A, C, H, W] for one scene, with ego at index zero.
    valid_mask: optional [A, 1, H, W] boolean communication/FOV support.
    BN runs on the real agent maps, never on padded agents. Unreceived cells
    remain zero inputs to BN and hence still affect its training statistics.
    Masking after Q/K/V spiking prevents BN offsets from creating messages.
    """

    def __init__(self, dim, num_heads=8, v_expansion=4,
                 spike_max=4, spike_norm=None):
        super().__init__()
        for name, value in (("dim", dim), ("num_heads", num_heads),
                            ("v_expansion", v_expansion)):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.value_dim = dim * v_expansion
        self.v_expansion = v_expansion
        self.scale = (dim // num_heads) ** -0.5

        self.head_spike = MultiSpike(spike_max, spike_norm)
        self.q_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim)
        )
        self.k_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim)
        )
        self.v_conv = nn.Sequential(
            nn.Conv2d(dim, self.value_dim, 1, bias=False),
            nn.BatchNorm2d(self.value_dim),
        )
        self.q_spike = MultiSpike(spike_max, spike_norm)
        self.k_spike = MultiSpike(spike_max, spike_norm)
        self.v_spike = MultiSpike(spike_max, spike_norm)
        self.attn_spike = MultiSpike(spike_max, spike_norm)
        self.proj_conv = nn.Sequential(
            nn.Conv2d(self.value_dim, dim, 1, bias=False), nn.BatchNorm2d(dim)
        )

    def _agent_tokens(self, x):
        agents, channels, height, width = x.shape
        return x.reshape(
            agents, self.num_heads, channels // self.num_heads, height * width
        ).permute(3, 1, 0, 2).contiguous()

    def forward(self, x, valid_mask=None):
        if x.ndim != 4 or x.shape[1] != self.dim or min(x.shape) <= 0:
            raise ValueError(
                "features must have nonempty shape [A, {}, H, W]".format(self.dim)
            )
        agents, _, height, width = x.shape
        if valid_mask is not None:
            if valid_mask.shape != (agents, 1, height, width):
                raise ValueError("valid_mask must have shape [A, 1, H, W]")
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
            # Also sanitize masked NaN/Inf rather than multiplying them by zero.
            x = x.masked_fill(~valid_mask, 0)

        spike_x = self.head_spike(x)
        q = self.q_spike(self.q_conv(spike_x))
        k = self.k_spike(self.k_conv(spike_x))
        v = self.v_spike(self.v_conv(spike_x))
        if valid_mask is not None:
            q = q.masked_fill(~valid_mask, 0)
            k = k.masked_fill(~valid_mask, 0)
            v = v.masked_fill(~valid_mask, 0)

        # [H*W, heads, A, head_channels]; no spatial all-to-all attention.
        q, k, v = (self._agent_tokens(t) for t in (q, k, v))
        weights = q @ k.transpose(-2, -1)
        context = (weights @ v) * (self.scale * 2)
        context = context.permute(2, 1, 3, 0).reshape(
            agents, self.value_dim, height, width
        ).contiguous()
        context = self.attn_spike(context)
        output = self.proj_conv(context)
        if valid_mask is not None:
            output = output.masked_fill(~valid_mask, 0)
        return output


class Sdtv3AttentionFusion(nn.Module):
    """Return ego after the full attention branch and unquantized residual."""

    def __init__(self, dim, num_heads=8, v_expansion=4,
                 spike_max=4, spike_norm=None):
        super().__init__()
        self.essa = SpikeDrivenAgentAttention(
            dim, num_heads, v_expansion, spike_max, spike_norm
        )

    def forward(self, x, valid_mask=None):
        delta = self.essa(x, valid_mask)
        residual = x
        if valid_mask is not None:
            residual = residual.masked_fill(
                ~valid_mask.to(device=x.device, dtype=torch.bool), 0
            )
        # Project all real agents before selecting ego, preserving BN semantics.
        return (residual + delta)[0]


def sdtv3_fusion_state_prefixes(model):
    """Exact module namespaces allowed to be new in a baseline initialization."""
    return tuple(
        (name + "." if name else "")
        for name, module in model.named_modules()
        if isinstance(module, Sdtv3AttentionFusion)
    )
