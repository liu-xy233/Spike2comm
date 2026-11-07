import torch
import torch.nn as nn
from einops import rearrange


class AT(nn.Module):
    """Agent/time conditioning used by CoST before deformable fusion."""

    def __init__(self, c_dim, f_dim=256, emb_size=3):
        super().__init__()
        self.reduce = nn.Sequential(nn.Linear(c_dim, f_dim), nn.ReLU())
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Embedding(emb_size, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)

    def forward(self, feat, prior_encoding):
        # feat: [B, L, C, H, W], prior_encoding: [B, L, 3]
        batch_size, cav_num, _, _, _ = feat.shape
        feat = rearrange(feat, "b l c h w -> b l h w c")
        time_delay = prior_encoding[..., 1:2].reshape(batch_size, cav_num, 1, 1, 1)
        dis_flag = prior_encoding[..., -1].reshape(batch_size, cav_num, 1, 1).long()
        time_delay = self.reduce(time_delay)
        feat = self.ln(feat)
        out = self.gamma(time_delay) * feat + self.beta(dis_flag.clamp_min(0))
        return rearrange(out, "b l h w c -> b l c h w")

