"""Inference-visible ego patch coverage, with the SAME label-alignment rules."""

import torch

from opencood.models.comm_modules.patch_utils import get_num_patches
from opencood.tools.align_value_predictor_oracle_labels import align_patch_utility


def geometry_valid_mask(metadata, feature_size=(50, 126), patch_size=(2, 3),
                        min_valid_fraction=0.5):
    """Compute valid ego patches from pose/FOV alone, without any utility labels.

The zero utility array only supplies the shape to the existing alignment
function. Its valid_mask is computed by warping an all-ones source map.
"""
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0,1]")
    height, width = feature_size
    count = get_num_patches(height, width, patch_size)
    _, mask, _ = align_patch_utility(
        torch.zeros(count), metadata, height, width, patch_size,
        downsample_rate=8, discrete_ratio=0.2,
        min_valid_fraction=min_valid_fraction,
    )
    return mask
