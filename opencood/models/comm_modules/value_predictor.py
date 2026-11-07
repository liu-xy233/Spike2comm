"""Counterfactual communication-value prediction from ego-side inputs."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from opencood.models.comm_modules.patch_utils import (
    get_num_patches,
    patchify_map,
)


def generate_patch_coordinates(
    height: int,
    width: int,
    patch_size: Tuple[int, int],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return row-major normalized patch centers shaped [num_patch, 2].

    Coordinates are ordered as ``(x, y)`` and normalized to ``[-1, 1]``.
    Patch ids therefore follow exactly the same row-major order as
    :func:`patchify_map` and the Oracle utility labels.
    """
    patch_h, patch_w = patch_size
    num_patch = get_num_patches(height, width, patch_size)
    grid_h, grid_w = height // patch_h, width // patch_w
    dtype = dtype or torch.float32

    y = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5)
    y = y * (2.0 / grid_h) - 1.0
    x = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5)
    x = x * (2.0 / grid_w) - 1.0

    # [grid_h,grid_w,2] -> [P,2], with width changing fastest.
    grid_x = x.unsqueeze(0).expand(grid_h, -1)
    grid_y = y.unsqueeze(1).expand(-1, grid_w)
    coordinates = torch.stack((grid_x, grid_y), dim=-1).reshape(
        num_patch, 2
    )
    return coordinates


class CounterfactualValuePredictor(nn.Module):
    """Predict the utility of every neighbor patch without seeing its feature.

    The predictor is intentionally independent from Where2comm's communication
    selector.  It consumes only information available before communication:
    the ego BEV feature and metadata describing one neighbor.

    Args:
        config: Value-predictor configuration.  A full configuration containing
            a ``value_predictor`` subsection is accepted as a convenience too.
    """

    def __init__(self, config: Dict[str, Any]):
        super(CounterfactualValuePredictor, self).__init__()
        if "value_predictor" in config:
            config = config["value_predictor"]

        self.enabled = bool(config.get("enabled", True))
        self.in_channels = int(config.get("in_channels", 256))
        self.embed_dim = int(config.get("embed_dim", 256))
        self.meta_dim = int(config.get("meta_dim", 4))
        self.num_patch = int(config.get("num_patch", 1050))
        self.patch_size = self._parse_patch_size(
            config.get("patch_size", (2, 3))
        )

        transformer_layers = int(config.get("transformer_layers", 2))
        transformer_heads = int(config.get("transformer_heads", 4))
        transformer_ffn_dim = int(
            config.get("transformer_ffn_dim", self.embed_dim)
        )
        dropout = float(config.get("dropout", 0.0))
        meta_hidden_dim = int(config.get("meta_hidden_dim", 128))
        meta_embed_dim = int(config.get("meta_embed_dim", 64))
        value_hidden_dim = int(config.get("value_hidden_dim", 128))
        coord_embed_dim = int(config.get("coord_embed_dim", 64))
        self.use_coord_embedding = bool(
            config.get("use_coord_embedding", True)
        )
        self.use_output_norm = bool(config.get("use_output_norm", True))

        self._validate_config(
            transformer_layers, transformer_heads, coord_embed_dim
        )

        # Spatial refinement preserves C, H and W so Oracle patch boundaries
        # remain unchanged.  The residual path also retains the raw BEV signal.
        self.ego_encoder = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                groups=self.in_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, 1),
            nn.ReLU(inplace=True),
        )

        patch_area = self.patch_size[0] * self.patch_size[1]
        self.patch_embedding = nn.Linear(
            self.in_channels * patch_area, self.embed_dim
        )

        self.metadata_encoder = nn.Sequential(
            nn.Linear(self.meta_dim, meta_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(meta_hidden_dim, meta_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_fusion = nn.Linear(
            self.embed_dim + meta_embed_dim, self.embed_dim
        )

        # Patch ids are row-major in patch_utils.  A learnable embedding gives
        # the Transformer explicit access to this spatial ordering.
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patch, self.embed_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        if self.use_coord_embedding:
            self.coord_encoder = nn.Sequential(
                nn.Linear(2, coord_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(coord_embed_dim, self.embed_dim),
            )
        else:
            self.coord_encoder = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.output_norm = (
            nn.LayerNorm(self.embed_dim)
            if self.use_output_norm else nn.Identity()
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, value_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden_dim, 1),
        )

    @staticmethod
    def _parse_patch_size(value: Any) -> Tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("patch_size must contain [patch_h, patch_w].")
        patch_size = (int(value[0]), int(value[1]))
        if patch_size[0] <= 0 or patch_size[1] <= 0:
            raise ValueError("patch_size values must be positive.")
        return patch_size

    def _validate_config(self, transformer_layers: int,
                         transformer_heads: int,
                         coord_embed_dim: int) -> None:
        positive_values = {
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "meta_dim": self.meta_dim,
            "num_patch": self.num_patch,
            "transformer_layers": transformer_layers,
            "transformer_heads": transformer_heads,
            "coord_embed_dim": coord_embed_dim,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError("{} must be positive, got {}.".format(name, value))
        if self.embed_dim % transformer_heads != 0:
            raise ValueError(
                "embed_dim ({}) must be divisible by transformer_heads ({}).".format(
                    self.embed_dim, transformer_heads
                )
            )

    def build_coordinate_embedding(
        self,
        batch_size: int,
        height: int,
        width: int,
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Build explicit coordinate context shaped [B, P, embed_dim]."""
        if self.coord_encoder is None:
            return None
        # [P,2], in the same row-major order as patchify_map.
        coordinates = generate_patch_coordinates(
            height,
            width,
            self.patch_size,
            device=reference.device,
            dtype=reference.dtype,
        )
        # [P,2] -> [B,P,2] -> [B,P,D]
        coordinates = coordinates.unsqueeze(0).expand(batch_size, -1, -1)
        return self.coord_encoder(coordinates)

    def forward(self, ego_feature: torch.Tensor,
                neighbor_metadata: torch.Tensor,
                return_dict: bool = False) -> Union[
                    torch.Tensor, Dict[str, torch.Tensor]
                ]:
        """Predict one scalar utility per neighbor patch.

        Tensor flow (P is ``num_patch`` and D is ``embed_dim``):
            ego_feature:       [B, C, H, W]
            refined feature:   [B, C, H, W]
            raw patches:       [B, C, P, patch_h, patch_w]
            patch tokens:      [B, P, D]
            metadata context:  [B, P, meta_embed_dim]
            patch coordinates: [B, P, 2]
            coordinate embed:  [B, P, D]
            fused tokens:      [B, P, D]
            transformer output:[B, P, D]
            normalized output: [B, P, D]
            value_map:         [B, P]

        When ``return_dict`` is true, the raw Transformer output is returned
        as ``patch_tokens`` alongside ``value_map`` for downstream analysis.
        """
        if ego_feature.dim() != 4:
            raise ValueError(
                "ego_feature must be [B,C,H,W], got {}.".format(
                    tuple(ego_feature.shape)
                )
            )
        if neighbor_metadata.dim() != 2:
            raise ValueError(
                "neighbor_metadata must be [B,meta_dim], got {}.".format(
                    tuple(neighbor_metadata.shape)
                )
            )

        batch_size, channels, height, width = ego_feature.shape
        if channels != self.in_channels:
            raise ValueError(
                "Expected {} ego channels, got {}.".format(
                    self.in_channels, channels
                )
            )
        if neighbor_metadata.shape != (batch_size, self.meta_dim):
            raise ValueError(
                "Expected neighbor_metadata shape ({}, {}), got {}.".format(
                    batch_size, self.meta_dim, tuple(neighbor_metadata.shape)
                )
            )

        actual_num_patch = get_num_patches(
            height, width, self.patch_size
        )
        if actual_num_patch != self.num_patch:
            raise ValueError(
                "Feature size ({}, {}) and patch_size {} produce {} patches; "
                "configured num_patch is {}.".format(
                    height, width, self.patch_size, actual_num_patch,
                    self.num_patch
                )
            )

        # [B,C,H,W] -> [B,C,H,W]
        refined_feature = ego_feature + self.ego_encoder(ego_feature)

        # Reuse Oracle patch indexing: [B,C,H,W] -> [B,C,P,ph,pw].
        patches = patchify_map(refined_feature, self.patch_size)
        # [B,C,P,ph,pw] -> [B,P,C*ph*pw] -> [B,P,D]
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().flatten(2)
        patch_tokens = self.patch_embedding(patches)

        # [B,meta_dim] -> [B,meta_embed_dim] -> [B,P,meta_embed_dim]
        metadata_token = self.metadata_encoder(neighbor_metadata)
        metadata_tokens = metadata_token.unsqueeze(1).expand(
            -1, self.num_patch, -1
        )

        # [B,P,D+meta_embed_dim] -> [B,P,D]
        fused_tokens = self.feature_fusion(
            torch.cat((patch_tokens, metadata_tokens), dim=-1)
        )
        fused_tokens = fused_tokens + self.position_embedding

        # Coordinates [B,P,2] -> coordinate embedding [B,P,D].
        coordinate_embedding = self.build_coordinate_embedding(
            batch_size, height, width, fused_tokens
        )
        if coordinate_embedding is not None:
            fused_tokens = fused_tokens + coordinate_embedding

        # [B,P,D] -> [B,P,D]
        modeled_tokens = self.transformer(fused_tokens)

        # [B,P,D] -> LayerNorm [B,P,D] -> [B,P,1] -> [B,P]
        normalized_tokens = self.output_norm(modeled_tokens)
        value_map = self.value_head(normalized_tokens).squeeze(-1)
        if return_dict:
            return {
                "value_map": value_map,
                "patch_tokens": modeled_tokens,
            }
        return value_map
