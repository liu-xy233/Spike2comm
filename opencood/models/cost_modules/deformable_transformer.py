import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_

from opencood.models.cost_modules.mln import AT
from opencood.models.cost_modules.ms_deform_attn import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        dif_offset=False,
        cav_num=2,
    ):
        super().__init__()
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
            dif_offset=dif_offset,
            cav_num=cav_num,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        for module in self.modules():
            if isinstance(module, MSDeformAttn):
                module._reset_parameters()
        normal_(self.level_embed)

    @staticmethod
    def get_valid_ratio(mask):
        _, height, width = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_w.float() / width, valid_h.float() / height], -1)

    def forward(self, srcs, masks, pos_embeds, prior_encoding=None):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level_id, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            batch_size, channels, height, width = src.shape
            spatial_shapes.append((height, width))
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level_id].view(1, 1, -1)
            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(mask) for mask in masks], 1)
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
            prior_encoding=prior_encoding,
        )
        return memory, (spatial_shapes, level_start_index)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        dif_offset=False,
        cav_num=2,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, cav_num=cav_num)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dif_offset = dif_offset
        if self.dif_offset:
            self.query_embed = AT(1, f_dim=d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, prior_encoding=None):
        if self.dif_offset:
            batch_size, cav_num, _ = prior_encoding.shape
            _, length, channels = src.shape
            spatial_len = length // cav_num
            src = src + self.query_embed(src.reshape(batch_size, cav_num, spatial_len, channels), prior_encoding).reshape(
                batch_size, length, channels
            )
        src2 = self.self_attn(
            self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask
        )
        src = self.norm1(src + self.dropout1(src2))
        return self.forward_ffn(src)


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for level_id, (height, width) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level_id, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level_id, 0] * width)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, prior_encoding=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(
                output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, prior_encoding=prior_encoding
            )
        return output


def _get_clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError("activation should be relu/gelu/glu, not {}.".format(activation))


def build_deforamble_transformer(args, num_feature_levels, num_encoder_layers, dif_offset=False, cav_num=2):
    return DeformableTransformer(
        d_model=args["hidden_dim"],
        nhead=args["nheads"],
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=args["dec_layers"],
        dim_feedforward=args["dim_feedforward"],
        dropout=args["dropout"],
        activation="relu",
        num_feature_levels=num_feature_levels,
        dec_n_points=args["dec_n_points"],
        enc_n_points=args["enc_n_points"],
        dif_offset=dif_offset,
        cav_num=cav_num,
    )
