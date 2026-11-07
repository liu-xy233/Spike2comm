import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from opencood.models.comm_modules.patch_utils import (
    dense_mask_from_patch_values,
    patchify_map,
)


def build_theta(metadata, height, width,
                downsample_rate=8, discrete_ratio=0.2):
    """
    Reconstruct the 2D affine matrix from the saved metadata:
    [dx, dy, yaw, distance].

    Apply the same aspect-ratio and translation normalization
    as Where2comm's fusion module.
    """
    dx, dy, yaw = [
        float(v) for v in metadata[:3]
    ]

    c = math.cos(yaw)
    s = math.sin(yaw)

    theta = torch.tensor(
        [[c, -s, dx],
         [s,  c, dy]],
        dtype=torch.float32,
    )

    # Match where2comm_attn.py's matrix normalization.
    theta[0, 1] *= height / width
    theta[1, 0] *= width / height

    theta[0, 2] = (
        dx / (downsample_rate * discrete_ratio * width) * 2
    )
    theta[1, 2] = (
        dy / (downsample_rate * discrete_ratio * height) * 2
    )

    return theta


def align_patch_utility(
    utility,
    metadata,
    height,
    width,
    patch_size=(2, 3),
    downsample_rate=8,
    discrete_ratio=0.2,
    min_valid_fraction=0.5,
):
    """
    Map the neighbor-coordinate utility map into the ego grid.

    Returns:
        aligned_utility: [num_patches]
        valid_mask:      [num_patches], bool
        valid_fraction:  [num_patches]
    """
    utility = torch.as_tensor(
        utility, dtype=torch.float32
    ).flatten()

    patch_h, patch_w = patch_size
    expected = (
        height // patch_h
    ) * (
        width // patch_w
    )

    if utility.numel() != expected:
        raise ValueError(
            "Expected {} labels, got {}".format(
                expected, utility.numel()
            )
        )

    if height % patch_h or width % patch_w:
        raise ValueError("Feature size is not divisible by patch size.")

    # [P] -> [1, H, W] -> [1, 1, H, W]
    dense = dense_mask_from_patch_values(
        utility, height, width, patch_size
    ).unsqueeze(0)

    theta = build_theta(
        metadata,
        height,
        width,
        downsample_rate,
        discrete_ratio,
    ).unsqueeze(0)

    grid = F.affine_grid(
        theta,
        dense.shape,
        align_corners=False,
    )

    # Nearest-neighbor sampling preserves piecewise-constant
    # patch utility values rather than interpolating their magnitudes.
    aligned_dense = F.grid_sample(
        dense,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )

    # Warp an all-ones map to identify valid source coverage.
    source_valid = torch.ones_like(dense)
    valid_dense = F.grid_sample(
        source_valid,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0] > 0.5

    # Re-aggregate the aligned dense map into ego-coordinate patches.
    value_patches = patchify_map(
        aligned_dense[0, 0], patch_size
    )
    valid_patches = patchify_map(
        valid_dense.float(), patch_size
    )

    valid_count = valid_patches.sum(dim=(-1, -2))
    patch_area = patch_h * patch_w

    valid_fraction = valid_count / patch_area

    # Ignore patches with less than the configured valid coverage.
    min_valid_cells = math.ceil(
        patch_area * min_valid_fraction
    )
    valid_mask = valid_count >= min_valid_cells

    value_sum = (
        value_patches * valid_patches
    ).sum(dim=(-1, -2))

    aligned_utility = (
        value_sum / valid_count.clamp_min(1)
    )

    # Invalid labels are stored as zero but MUST be excluded
    # from the training loss using valid_mask.
    aligned_utility = torch.where(
        valid_mask,
        aligned_utility,
        torch.zeros_like(aligned_utility),
    )

    return aligned_utility, valid_mask, valid_fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_path", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--patch_h", type=int, default=2)
    parser.add_argument("--patch_w", type=int, default=3)
    parser.add_argument("--downsample_rate", type=int, default=8)
    parser.add_argument("--discrete_ratio", type=float, default=0.2)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_valid_fraction", type=float, default=0.5)
    args = parser.parse_args()

    oracle_path = Path(args.oracle_path)
    feature_dir = Path(args.feature_dir)
    output_path = Path(args.output_path)

    assert oracle_path.is_file(), (
        "Oracle file not found: {}".format(oracle_path)
    )
    assert feature_dir.is_dir(), (
        "Feature directory not found: {}".format(feature_dir)
    )

    records = torch.load(
        str(oracle_path), map_location="cpu"
    )
    assert isinstance(records, list) and records, (
        "Oracle records must be a non-empty list."
    )

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive.")
        records = records[:args.max_samples]

    patch_size = (args.patch_h, args.patch_w)
    aligned_records = []
    valid_counts = []

    seen = set()

    for record in records:
        sid = int(record["sample_id"])
        assert sid not in seen, "Duplicate sample_id: {}".format(sid)
        seen.add(sid)

        feature_path = feature_dir / "sample_{:04d}.pt".format(sid)
        assert feature_path.is_file(), (
            "Missing feature file: {}".format(feature_path)
        )

        feature = torch.load(
            str(feature_path), map_location="cpu"
        )

        metadata = torch.as_tensor(
            feature["neighbor_metadata"],
            dtype=torch.float32,
        ).flatten()

        assert metadata.shape == (4,), (
            "Invalid metadata for sample {}".format(sid)
        )

        height, width = [
            int(v) for v in record["feature_size"]
        ]

        assert tuple(feature["ego_feature"].shape[-2:]) == (
            height, width
        ), "Feature/Oracle spatial size mismatch at sample {}".format(sid)

        assert tuple(record["patch_size"]) == patch_size, (
            "Patch-size mismatch at sample {}".format(sid)
        )

        original_utility = torch.as_tensor(
            record["utility"], dtype=torch.float32
        ).flatten()

        aligned, valid, fraction = align_patch_utility(
            original_utility,
            metadata,
            height,
            width,
            patch_size=patch_size,
            downsample_rate=args.downsample_rate,
            discrete_ratio=args.discrete_ratio,
            min_valid_fraction=args.min_valid_fraction,
        )

        output_record = dict(record)
        output_record["neighbor_utility"] = original_utility.tolist()
        output_record["utility"] = aligned.tolist()
        output_record["valid_mask"] = valid.tolist()
        output_record["valid_fraction"] = fraction.tolist()
        output_record["label_coordinate_system"] = "ego"

        if "oracle_patch_ids" in output_record:
            output_record["neighbor_oracle_patch_ids"] = (
                output_record.pop("oracle_patch_ids")
            )

        valid_count = int(valid.sum().item())
        valid_counts.append(valid_count)
        aligned_records.append(output_record)

        if len(aligned_records) <= 3:
            print(
                "sample={} valid_patches={}/{}".format(
                    sid, valid_count, aligned.numel()
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(aligned_records, str(output_path))

    print("\n===== Alignment summary =====")
    print("Records:", len(aligned_records))
    print("Output:", output_path)
    print("Patch size:", patch_size)
    print("Grid size: {} x {}".format(
        height // args.patch_h,
        width // args.patch_w,
    ))
    print("Valid patches/sample: mean={:.1f}, min={}, max={}".format(
        sum(valid_counts) / len(valid_counts),
        min(valid_counts),
        max(valid_counts),
    ))
    print("[OK] Aligned labels saved; original Oracle unchanged.")


if __name__ == "__main__":
    main()
