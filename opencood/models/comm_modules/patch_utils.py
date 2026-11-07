import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


PatchSize = Tuple[int, int]


def get_patch_grid(height: int, width: int, patch_size: PatchSize) -> Tuple[int, int]:
    patch_h, patch_w = patch_size
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(
            "Feature size ({}, {}) must be divisible by patch size {}.".format(
                height, width, patch_size
            )
        )
    return height // patch_h, width // patch_w


def get_num_patches(height: int, width: int, patch_size: PatchSize) -> int:
    grid_h, grid_w = get_patch_grid(height, width, patch_size)
    return grid_h * grid_w


def patch_id_to_slice(
    patch_id: int, height: int, width: int, patch_size: PatchSize
) -> Tuple[slice, slice]:
    patch_h, patch_w = patch_size
    grid_h, grid_w = get_patch_grid(height, width, patch_size)
    num_patches = grid_h * grid_w
    if patch_id < 0 or patch_id >= num_patches:
        raise ValueError("patch_id {} is out of range [0, {}).".format(patch_id, num_patches))
    row = patch_id // grid_w
    col = patch_id % grid_w
    return slice(row * patch_h, (row + 1) * patch_h), slice(col * patch_w, (col + 1) * patch_w)


def patchify_map(x: torch.Tensor, patch_size: PatchSize) -> torch.Tensor:
    """Convert [..., H, W] to [..., num_patches, patch_h, patch_w]."""
    patch_h, patch_w = patch_size
    height, width = x.shape[-2:]
    grid_h, grid_w = get_patch_grid(height, width, patch_size)
    leading_shape = x.shape[:-2]
    x = x.reshape(*leading_shape, grid_h, patch_h, grid_w, patch_w)
    # [...,grid_h,patch_h,grid_w,patch_w]
    #     -> [...,grid_h,grid_w,patch_h,patch_w].
    # This keeps patch ids row-major and consistent with patch_id_to_slice.
    x = x.transpose(-3, -2).reshape(
        *leading_shape, grid_h * grid_w, patch_h, patch_w
    )
    return x


def patch_confidence_stats(
    confidence_map: torch.Tensor,
    patch_size: PatchSize = (3, 2),
    activation_threshold: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute patch-level statistics from confidence map shaped [..., 1, H, W] or [..., H, W]."""
    if confidence_map.dim() >= 3 and confidence_map.shape[-3] == 1:
        confidence_map = confidence_map.squeeze(-3)
    patches = patchify_map(confidence_map, patch_size)
    flat = patches.flatten(-2)
    return {
        "mean": flat.mean(dim=-1),
        "max": flat.max(dim=-1).values,
        "activation_ratio": (flat > activation_threshold).float().mean(dim=-1),
    }


def dense_mask_from_patch_values(
    patch_values: torch.Tensor,
    height: int,
    width: int,
    patch_size: PatchSize = (3, 2),
) -> torch.Tensor:
    """Expand patch values [..., P] to dense values [..., 1, H, W]."""
    patch_h, patch_w = patch_size
    grid_h, grid_w = get_patch_grid(height, width, patch_size)
    if patch_values.shape[-1] != grid_h * grid_w:
        raise ValueError(
            "Expected {} patch values, got {}.".format(grid_h * grid_w, patch_values.shape[-1])
        )
    leading_shape = patch_values.shape[:-1]
    dense = patch_values.reshape(*leading_shape, grid_h, grid_w)
    dense = dense.repeat_interleave(patch_h, dim=-2).repeat_interleave(patch_w, dim=-1)
    return dense.unsqueeze(-3)


def apply_patch_state(
    dense_mask: torch.Tensor,
    patch_id: int,
    value: float,
    patch_size: PatchSize = (3, 2),
    agent_index: int = 1,
) -> torch.Tensor:
    """Return a cloned mask with one agent patch forced to ON/OFF."""
    mask = dense_mask.clone()
    height, width = mask.shape[-2:]
    h_slice, w_slice = patch_id_to_slice(patch_id, height, width, patch_size)
    if mask.dim() == 5:
        mask[:, agent_index, :, h_slice, w_slice] = value
    elif mask.dim() == 4:
        mask[agent_index, :, h_slice, w_slice] = value
    else:
        raise ValueError("dense_mask must be [B,L,1,H,W] or [L,1,H,W], got {}.".format(mask.shape))
    return mask


def flatten_agent_mask(mask: torch.Tensor, record_len: torch.Tensor) -> torch.Tensor:
    """Convert [B,L,1,H,W] mask to [sum(record_len),1,H,W]."""
    if mask.dim() == 4:
        return mask
    if mask.dim() != 5:
        raise ValueError("mask must be 4D or 5D, got {}.".format(mask.shape))
    flattened = []
    for batch_idx in range(mask.shape[0]):
        cav_num = int(record_len[batch_idx])
        flattened.append(mask[batch_idx, :cav_num])
    return torch.cat(flattened, dim=0)


def unflatten_agent_mask(mask: torch.Tensor, record_len: torch.Tensor, max_cav: Optional[int] = None) -> torch.Tensor:
    """Convert [sum(record_len),1,H,W] mask to padded [B,L,1,H,W]."""
    if mask.dim() == 5:
        return mask
    if mask.dim() != 4:
        raise ValueError("mask must be 4D or 5D, got {}.".format(mask.shape))
    batch_size = int(record_len.shape[0])
    max_cav = int(max_cav or record_len.max().item())
    padded_shape = (batch_size, max_cav) + tuple(mask.shape[1:])
    padded = mask.new_zeros(padded_shape)
    offset = 0
    for batch_idx in range(batch_size):
        cav_num = int(record_len[batch_idx])
        padded[batch_idx, :cav_num] = mask[offset : offset + cav_num]
        offset += cav_num
    return padded


def communication_rate_from_mask(
    mask: torch.Tensor,
    record_len: torch.Tensor,
    ego_index: int = 0,
) -> torch.Tensor:
    """Compute non-ego communicated cell ratio per batch, averaged over batch."""
    padded = unflatten_agent_mask(mask, record_len)
    _, _, _, height, width = padded.shape
    rates = []
    for batch_idx in range(padded.shape[0]):
        cav_num = int(record_len[batch_idx])
        if cav_num <= 1:
            rates.append(padded.new_tensor(0.0))
            continue
        non_ego = [idx for idx in range(cav_num) if idx != ego_index]
        rates.append(padded[batch_idx, non_ego].sum() / (len(non_ego) * height * width))
    return torch.stack(rates).mean()


def stratified_sample_patch_ids(
    patch_scores: torch.Tensor,
    bins: Sequence[int] = (2, 3, 4, 7),
    generator: Optional[random.Random] = None,
) -> List[int]:
    """Sample patch ids from score quartiles using the requested per-quartile counts."""
    if patch_scores.dim() != 1:
        patch_scores = patch_scores.flatten()
    generator = generator or random
    sorted_ids = torch.argsort(patch_scores, descending=False).cpu().tolist()
    num_patches = len(sorted_ids)
    selected: List[int] = []
    for quartile_idx, count in enumerate(bins):
        start = math.floor(num_patches * quartile_idx / 4)
        end = math.floor(num_patches * (quartile_idx + 1) / 4)
        bucket = sorted_ids[start:end]
        if len(bucket) <= count:
            selected.extend(bucket)
        else:
            selected.extend(generator.sample(bucket, count))
    return selected


def topk_patch_ids(
    patch_scores: torch.Tensor,
    k: int,
    threshold: Optional[float] = None,
    largest: bool = True,
) -> List[int]:
    if patch_scores.dim() != 1:
        patch_scores = patch_scores.flatten()
    candidate_ids = torch.arange(patch_scores.numel(), device=patch_scores.device)
    candidate_scores = patch_scores
    if threshold is not None:
        keep = candidate_scores > threshold if largest else candidate_scores < threshold
        candidate_scores = candidate_scores[keep]
        candidate_ids = candidate_ids[keep]
    if candidate_scores.numel() == 0 or k <= 0:
        return []
    k = min(k, candidate_scores.numel())
    ids = candidate_ids[torch.topk(candidate_scores, k=k, largest=largest).indices]
    return ids.cpu().tolist()


def resize_mask_nearest(mask: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    original_shape = mask.shape
    if mask.dim() == 5:
        batch_size, cav_num = mask.shape[:2]
        mask = mask.reshape(batch_size * cav_num, *mask.shape[2:])
        mask = F.interpolate(mask.float(), size=target_size, mode="nearest")
        return mask.reshape(batch_size, cav_num, *mask.shape[1:])
    if mask.dim() == 4:
        return F.interpolate(mask.float(), size=target_size, mode="nearest")
    raise ValueError("Cannot resize mask with shape {}.".format(original_shape))

