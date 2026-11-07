"""Independent SDT V3-style counterfactual value predictor.

Only ego BEV and one neighbor's metadata are inputs. Output is signed utility
on the ego patch grid. Membrane residuals and continuous final readout follow
the documented SFA design; no temporal simulation axis is invented.
"""

import math

import torch
from torch import nn

from opencood.models.comm_modules.patch_utils import get_num_patches, patchify_map
from opencood.models.comm_modules.value_predictor import generate_patch_coordinates
from opencood.models.comm_modules.spiking_value_blocks import (
    MultiSpike, SignedInputCurrent, SpikeConvBlock, SpikeProjectionMLP,
    SpikeTransformerBlock,
)


def resolve_spiking_config(config):
    """Canonical, checkpoint-verifiable architecture and input calibration."""
    config = config.get("model", config.get("value_predictor", config))
    defaults = {
        "architecture": "sdtv3_snn", "in_channels": 128, "embed_dim": 256,
        "feature_size": [50, 126], "patch_size": [2, 3], "num_patch": 1050,
        "meta_dim": 4, "meta_hidden_dim": 128, "meta_embed_dim": 64,
        "metadata_scales": [100.8, 40.0, math.pi, 120.0],
        "coord_embed_dim": 64, "use_coord_embedding": True,
        "use_position_embedding": True, "conv_blocks": 1,
        "conv_mlp_ratio": 4, "stem_kernel": 7, "sep_expansion": 2,
        "transformer_layers": 2, "transformer_heads": 4, "mlp_ratio": 4,
        "v_expansion": 4, "spike_max": 4, "spike_norm": 4.0,
        "attention_impl": "kv_first", "qkv_bn_init": 0.5,
        "value_hidden_dim": 128,
    }
    result = {key: config.get(key, value) for key, value in defaults.items()}
    for key, default in defaults.items():
        if isinstance(default, bool):
            if not isinstance(result[key], bool):
                raise ValueError(key + " must be a boolean")
        elif isinstance(default, int):
            if isinstance(result[key], bool) or int(result[key]) != result[key] or result[key] <= 0:
                raise ValueError(key + " must be a positive integer")
            result[key] = int(result[key])
        elif isinstance(default, float):
            result[key] = float(result[key])
            if not math.isfinite(result[key]) or result[key] <= 0:
                raise ValueError(key + " must be finite and positive")
    for key in ("feature_size", "patch_size"):
        value = result[key]
        if len(value) != 2 or any(int(v) != v or v <= 0 for v in value):
            raise ValueError(key + " must contain two positive integers")
        result[key] = list(map(int, value))
    result["metadata_scales"] = list(map(float, result["metadata_scales"]))
    if len(result["metadata_scales"]) != result["meta_dim"] or any(
        not math.isfinite(v) or v <= 0 for v in result["metadata_scales"]
    ):
        raise ValueError("metadata_scales must match meta_dim and be positive")
    if result["embed_dim"] % result["transformer_heads"]:
        raise ValueError("embed_dim must be divisible by transformer_heads")
    if result["stem_kernel"] % 2 != 1:
        raise ValueError("stem_kernel must be odd to preserve the BEV grid")
    h, w = result["feature_size"]
    if get_num_patches(h, w, tuple(result["patch_size"])) != result["num_patch"]:
        raise ValueError("feature_size/patch_size disagree with configured num_patch")
    if result["architecture"] != "sdtv3_snn":
        raise ValueError("Spiking predictor requires architecture=sdtv3_snn")
    if config.get("use_output_norm", False):
        raise ValueError("SNN membrane path has no output LayerNorm; set use_output_norm=false")
    if float(config.get("dropout", 0)) != 0:
        raise ValueError("This reference SNN block requires dropout=0")
    if "transformer_ffn_dim" in config and int(config["transformer_ffn_dim"]) != result["embed_dim"] * result["mlp_ratio"]:
        raise ValueError("Use mlp_ratio; transformer_ffn_dim conflicts with the SNN width")
    return result


class SpikingCounterfactualValuePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        raw = config.get("model", config.get("value_predictor", config))
        c = self.architecture_config = resolve_spiking_config(config)
        self.in_channels, self.embed_dim = c["in_channels"], c["embed_dim"]
        self.meta_dim, self.num_patch = c["meta_dim"], c["num_patch"]
        self.patch_size, self.feature_size = tuple(c["patch_size"]), tuple(c["feature_size"])
        self.grid_size = tuple(s // p for s, p in zip(self.feature_size, self.patch_size))
        maximum, norm, dim = c["spike_max"], c["spike_norm"], self.embed_dim
        self.ego_encoder = nn.Sequential(*[
            SpikeConvBlock(self.in_channels, c["conv_mlp_ratio"], c["stem_kernel"],
                           c["sep_expansion"], maximum, norm)
            for _ in range(c["conv_blocks"])
        ])
        patch_channels = self.in_channels * self.patch_size[0] * self.patch_size[1]
        self.patch_embedding = nn.Sequential(
            MultiSpike(maximum, norm), nn.Conv2d(patch_channels, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.metadata_current = SignedInputCurrent(c["metadata_scales"], maximum)
        self.metadata_encoder = SpikeProjectionMLP(
            self.meta_dim * 2, c["meta_hidden_dim"], c["meta_embed_dim"], maximum, norm
        )
        self.feature_fusion = nn.Sequential(
            MultiSpike(maximum, norm),
            nn.Conv2d(dim + c["meta_embed_dim"], dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        if c["use_position_embedding"]:
            self.position_embedding = nn.Parameter(torch.empty(1, self.num_patch, dim))
            nn.init.trunc_normal_(self.position_embedding, std=0.02)
        else:
            self.register_parameter("position_embedding", None)
        # Reuse exactly the existing Oracle-aligned coordinate generator.
        coords = generate_patch_coordinates(*self.feature_size, self.patch_size)
        self.register_buffer("patch_coordinates", coords, persistent=False)
        if c["use_coord_embedding"]:
            self.coordinate_current = SignedInputCurrent([1.0, 1.0], maximum)
            self.coord_encoder = SpikeProjectionMLP(4, c["coord_embed_dim"], dim, maximum, norm)
        else:
            self.coordinate_current = self.coord_encoder = None
        self.transformer = nn.Sequential(*[
            SpikeTransformerBlock(
                dim, c["transformer_heads"], c["mlp_ratio"], c["v_expansion"],
                c["sep_expansion"], maximum, norm, c["attention_impl"], c["qkv_bn_init"]
            ) for _ in range(c["transformer_layers"])
        ])
        self.output_norm = nn.Identity()
        self.value_head = nn.Sequential(
            MultiSpike(maximum, norm),
            nn.Conv2d(dim, c["value_hidden_dim"], 1, bias=False),
            nn.BatchNorm2d(c["value_hidden_dim"]),
            MultiSpike(maximum, norm),
            # Signed real-valued utility readout: no BN/spike/sigmoid after it.
            nn.Conv2d(c["value_hidden_dim"], 1, 1, bias=True),
        )
        self.collect_spike_stats = False
        self.set_spike_monitoring(bool(raw.get("collect_spike_stats", False)))

    def set_spike_monitoring(self, enabled=True):
        self.collect_spike_stats = bool(enabled)
        for module in self.modules():
            if isinstance(module, MultiSpike):
                module.record_stats = self.collect_spike_stats
                module.last_stats = None

    def get_spike_stats(self):
        """Detached measurements of the last forward; no CPU sync in hot path."""
        return {
            name: {key: float(value.cpu()) for key, value in module.last_stats.items()}
            for name, module in self.named_modules()
            if isinstance(module, MultiSpike) and module.last_stats is not None
        }

    def patchify_feature(self, feature):
        """[B,C,H,W] -> [B,6C,25,42], preserving row-major patch ids."""
        b, c, h, w = feature.shape
        ph, pw = self.patch_size
        patches = patchify_map(feature, self.patch_size)  # [B,C,P,ph,pw]
        tokens = patches.permute(0, 2, 1, 3, 4).contiguous().flatten(2)
        return tokens.transpose(1, 2).reshape(b, c * ph * pw, h // ph, w // pw)

    def forward(self, ego_feature, neighbor_metadata, return_dict=False):
        if ego_feature.ndim != 4 or tuple(ego_feature.shape[1:]) != (self.in_channels, *self.feature_size):
            raise ValueError("Expected ego_feature [B,{}, {}, {}] for configured num_patch={}".format(
                self.in_channels, *self.feature_size, self.num_patch))
        b = ego_feature.shape[0]
        if b == 0 or neighbor_metadata.shape != (b, self.meta_dim):
            raise ValueError("Expected nonempty neighbor_metadata [B,{}]".format(self.meta_dim))
        if ego_feature.device != neighbor_metadata.device:
            raise ValueError("ego_feature and neighbor_metadata must be on the same device")
        if not ego_feature.is_floating_point() or not neighbor_metadata.is_floating_point():
            raise ValueError("Features and metadata must be floating-point tensors")
        if not torch.isfinite(ego_feature).all() or not torch.isfinite(neighbor_metadata).all():
            raise ValueError("Features and metadata must be finite")

        # [B,C,50,126] -> [B,C,50,126]. Residuals are INSIDE the encoder.
        refined = self.ego_encoder(ego_feature)
        # [B,C,1050,2,3] -> [B,6C,25,42] -> [B,256,25,42].
        patch_membrane = self.patch_embedding(self.patchify_feature(refined))
        # [B,4] -> signed input [B,8,1] -> [B,64,1] -> [B,64,25,42].
        meta_current = self.metadata_current(neighbor_metadata.unsqueeze(-1))
        meta = self.metadata_encoder(meta_current).unsqueeze(-1)
        meta = meta.expand(-1, -1, *self.grid_size)
        # Concatenate channels: [B,320,25,42] -> [B,256,25,42].
        fused = self.feature_fusion(torch.cat((patch_membrane, meta), dim=1))
        if self.position_embedding is not None:
            fused = fused + self.position_embedding.transpose(1, 2).reshape(1, self.embed_dim, *self.grid_size)
        if self.coord_encoder is not None:
            # [P,2] -> [B,2,P] -> signed [B,4,P] -> [B,256,P].
            coords = self.patch_coordinates.to(dtype=ego_feature.dtype).T.unsqueeze(0).expand(b, -1, -1)
            coordinate_embedding = self.coord_encoder(self.coordinate_current(coords))
            fused = fused + coordinate_embedding.reshape(b, self.embed_dim, *self.grid_size)
        # Keep continuous fused membrane here: do NOT globally spike the skip.
        modeled = self.transformer(fused)  # [B,256,25,42], 3 residuals/block.
        value_map = self.value_head(self.output_norm(modeled)).flatten(1)  # [B,1050]
        if return_dict:
            result = {
                "value_map": value_map,
                "patch_tokens": modeled.flatten(2).transpose(1, 2),  # [B,1050,256] membrane
            }
            if self.collect_spike_stats:
                result["spike_stats"] = self.get_spike_stats()
            return result
        return value_map
