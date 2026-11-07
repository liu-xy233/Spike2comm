"""SDTv3-style four-level spike convolution with a projected residual."""

import torch
from torch import nn


class _QuantizeFour(torch.autograd.Function):
    """Match e_spikeformer_bev_backbone.Quant at T=4, including boundaries."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.round(torch.clamp(x, min=0, max=4))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = grad_output.clone()
        grad[(x < 0) | (x > 4)] = 0
        return grad


class MultiSpikeNorm4(nn.Module):
    """Stateless SDTv3 multi-spike activation, fixed T=4 in train and eval.

    The division by four also scales the surrogate gradient by 1/4. No time
    dimension or neuron state is introduced. Kept independent of the SDTv3
    backbone's timm/monitor imports so the E-3DSNN neck only requires torch.
    """

    def forward(self, x):
        return _QuantizeFour.apply(x) / 4.0


class SpikeResidualConvBlock(nn.Module):
    """y = BN(Conv3x3(Spike(x))) + shortcut(x), with no output activation.

    Main conv/BN are registered as ``0``/``1`` to retain the original ANN
    Sequential state_dict names and shapes. The shortcut consumes the original
    continuous input; a 1x1 projection aligns stride/channels when needed.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(SpikeResidualConvBlock, self).__init__()
        self.add_module("0", nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        ))
        self.add_module("1", nn.BatchNorm2d(
            out_channels, eps=1e-3, momentum=0.01
        ))
        self.spike = MultiSpikeNorm4()
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            )

    @property
    def conv(self):
        return self._modules["0"]

    @property
    def bn(self):
        return self._modules["1"]

    def forward(self, x):
        main = self.bn(self.conv(self.spike(x)))
        residual = self.shortcut(x)
        if main.shape != residual.shape:
            raise RuntimeError("Spike residual shape mismatch: {} vs {}".format(
                tuple(main.shape), tuple(residual.shape)
            ))
        return main + residual
