import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn


def _is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    n_batch, _, n_heads, head_dim = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes.tolist()):
        value_l = value_list[level_id].flatten(2).transpose(1, 2)
        value_l = value_l.reshape(n_batch * n_heads, head_dim, height, width)
        sampling_grid_l = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        sampling_value_l = F.grid_sample(
            value_l,
            sampling_grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        n_batch * n_heads, 1, len_q, n_levels * n_points
    )
    output = (
        torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    ).sum(-1)
    output = output.view(n_batch, n_heads * head_dim, len_q)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention with a pure PyTorch execution path."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, dif_offset=False, cav_num=2):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = d_model // n_heads
        if not _is_power_of_2(head_dim):
            warnings.warn("For CUDA efficiency, each deformable-attention head dim should be a power of 2.")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.im2col_step = 64

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for point_id in range(self.n_points):
            grid_init[:, :, point_id, :] *= point_id + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        n_batch, len_q, _ = query.shape
        _, len_in, _ = input_flatten.shape
        expected_len = (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()
        if len_in != expected_len:
            raise AssertionError("input_flatten spatial length mismatch")

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(n_batch, len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            n_batch, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            n_batch, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            n_batch, len_q, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError("reference_points last dimension must be 2 or 4")

        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)
