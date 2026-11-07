import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce

from opencood.models.cost_modules.deformable_transformer import build_deforamble_transformer
from opencood.models.cost_modules.mln import AT
from opencood.models.cost_modules.position_encoding import PositionEmbeddingSine


class DeformableDETR(nn.Module):
    """CoST deformable spatiotemporal BEV fusion."""

    def __init__(self, args):
        super().__init__()
        hidden_dim = args["hidden_dim"]
        self.hidden_dim = hidden_dim
        self.pos_emb = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.multi_time = args.get("multi_time", False)
        self.dif_offset = args.get("dif_offset", False)
        self.num_layers = args["enc_layers"]

        if self.dif_offset:
            self.query_embed = AT(1, f_dim=hidden_dim)

        max_agents = args["max_cav"] + 1 if self.multi_time else args["max_cav"]
        self.transformer = build_deforamble_transformer(
            args,
            num_feature_levels=max_agents,
            num_encoder_layers=args["enc_layers"],
            dif_offset=False,
        )

        self.mlp_head = nn.Sequential(
            Reduce("b l h w c -> b h w c", "mean"),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            Rearrange("b h w c -> b c h w"),
        )

    def forward(self, src, mask=None, prior_encoding=None):
        # src: [B, L, C, H, W], mask: [B, L], prior_encoding: [B, L, 3]
        batch_size, cav_num, channels, height, width = src.size()
        if channels != self.hidden_dim:
            raise AssertionError(
                "CoST DeformableDETR expects {} channels, got {}.".format(
                    self.hidden_dim, channels
                )
            )
        if mask is None:
            mask = torch.ones(batch_size, cav_num, dtype=torch.bool, device=src.device)
        if prior_encoding is None:
            prior_encoding = torch.zeros(batch_size, cav_num, 3, dtype=src.dtype, device=src.device)

        if self.dif_offset:
            src = src + self.query_embed(src, prior_encoding)

        src = src.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        mask = mask.reshape(cav_num, batch_size, 1, 1).repeat(1, 1, height, width)
        pos = self.pos_emb(mask.flatten(0, 1)).reshape(cav_num, batch_size, channels, height, width)
        mask = ~mask.bool()

        memory, _ = self.transformer(src, mask, pos, prior_encoding=prior_encoding)
        return self.mlp_head(memory.reshape(batch_size, cav_num, height, width, channels))

