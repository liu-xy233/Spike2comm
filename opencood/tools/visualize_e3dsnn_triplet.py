import os
import random
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize raw lidar / BEV-adapter feature / pre-communication feature for E3DSNN + Where2comm."
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to trained model directory (contains config.yaml and checkpoints).")
    parser.add_argument("--eval_epoch", type=int, required=True,
                        help="Checkpoint epoch to load.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Dataset split to visualize.")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="How many random samples to visualize.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampling.")
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="Optional comma-separated indices, e.g. 0,10,25 . If set, overrides random sampling.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save visualizations.")
    parser.add_argument("--save_tensors", action="store_true",
                        help="Also save raw tensors (.pt) for each sample.")
    return parser.parse_args()


def normalize_map(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.log1p(np.maximum(x, 0.0))
    x = x - x.min()
    maxv = x.max()
    if maxv > 1e-8:
        x = x / maxv
    return x


def feature_to_heatmap(feat_chw):
    """
    feat_chw: torch.Tensor [C, H, W]
    """
    feat = feat_chw.detach().float().abs().mean(dim=0).cpu().numpy()
    return normalize_map(feat)


def filter_points_in_range(points, lidar_range):
    """
    points: [N, >=3]
    lidar_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    x_min, y_min, z_min, x_max, y_max, z_max = lidar_range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return points[mask]


def draw_lidar_bev(ax, points, lidar_range, title):
    pts = filter_points_in_range(points, lidar_range)
    if pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(lidar_range[0], lidar_range[3])
    ax.set_ylim(lidar_range[1], lidar_range[4])
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def draw_heatmap(ax, heatmap, title):
    ax.imshow(heatmap, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("W")
    ax.set_ylabel("H")


def load_hypes_and_dataset(model_dir, split):
    cfg_path = os.path.join(model_dir, "config.yaml")
    hypes = yaml_utils.load_yaml(cfg_path, None)

    if split == "train":
        visualize_train = True
        train_flag = True
    else:
        visualize_train = True
        train_flag = False

    dataset = build_dataset(hypes, visualize=visualize_train, train=train_flag)
    return hypes, dataset


def build_single_batch(dataset, index):
    sample = dataset[index]
    batch = dataset.collate_batch_test([sample])
    return batch


def pick_indices(dataset_len, num_samples, seed, sample_indices=None):
    if sample_indices is not None and len(sample_indices.strip()) > 0:
        return [int(x.strip()) for x in sample_indices.split(",") if x.strip()]
    rng = random.Random(seed)
    num_samples = min(num_samples, dataset_len)
    return rng.sample(list(range(dataset_len)), num_samples)


def main():
    opt = parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)

    print("Loading config and dataset...")
    hypes, dataset = load_hypes_and_dataset(opt.model_dir, opt.split)

    print("Creating model...")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Loading checkpoint...")
    epoch_id, model = train_utils.load_saved_model(opt.model_dir, model, opt.eval_epoch)
    model.eval()
    print("Loaded epoch =", epoch_id)

    if not hasattr(model, "_extract_cached_context"):
        raise RuntimeError("This script requires a model with _extract_cached_context API.")

    lidar_range = hypes["preprocess"]["cav_lidar_range"]

    selected_indices = pick_indices(
        len(dataset),
        opt.num_samples,
        opt.seed,
        opt.sample_indices
    )
    print("Selected indices:", selected_indices)

    summary = []

    for vis_id, data_idx in enumerate(selected_indices):
        print("\n[{} / {}] visualizing sample index {}".format(
            vis_id + 1, len(selected_indices), data_idx
        ))

        # base_data: raw lidar for each agent
        if not hasattr(dataset, "retrieve_base_data"):
            raise RuntimeError("Dataset does not provide retrieve_base_data().")
        base_data = dataset.retrieve_base_data(data_idx)

        # model input batch
        batch_data = build_single_batch(dataset, data_idx)
        batch_data = train_utils.to_device(batch_data, device)
        ego_data = batch_data["ego"]

        with torch.no_grad():
            cached = model._extract_cached_context(ego_data)

            record_len = int(cached["record_len"][0].item())
            if record_len < 2:
                print("Skip sample {} because record_len < 2".format(data_idx))
                continue

            # 1) BEV adapter output
            spatial_features = cached["spatial_features"][:record_len]  # [N, C, H, W]

            # 2) first communication scale input = after first 2D block, before mask
            pre_comm_features = model.backbone.blocks[0](spatial_features)  # [N, C, H, W]

        cav_keys = list(base_data.keys())[:record_len]

        # Usually DAIR has 2 agents; we only visualize first 2 for clarity
        num_agents_to_show = min(2, record_len)

        fig, axes = plt.subplots(num_agents_to_show, 3, figsize=(15, 5 * num_agents_to_show), squeeze=False)
        fig.suptitle(
            "Sample {} | epoch {} | split {}\ncol1=raw lidar, col2=BEV-adapter output, col3=pre-communication feature".format(
                data_idx, epoch_id, opt.split
            ),
            fontsize=12
        )

        meta = {
            "dataset_index": int(data_idx),
            "epoch": int(epoch_id),
            "split": opt.split,
            "record_len": int(record_len),
            "agents": []
        }

        for agent_idx in range(num_agents_to_show):
            cav_id = cav_keys[agent_idx]
            cav_content = base_data[cav_id]
            lidar_np = cav_content["lidar_np"]

            # raw lidar
            draw_lidar_bev(
                axes[agent_idx, 0],
                lidar_np,
                lidar_range,
                title="Agent {} raw lidar\n{} points".format(agent_idx, lidar_np.shape[0])
            )

            # BEV adapter output
            bev_heatmap = feature_to_heatmap(spatial_features[agent_idx])
            draw_heatmap(
                axes[agent_idx, 1],
                bev_heatmap,
                title="Agent {} spatial_features\nshape={}".format(
                    agent_idx, tuple(spatial_features[agent_idx].shape)
                )
            )

            # pre-communication feature
            pre_comm_heatmap = feature_to_heatmap(pre_comm_features[agent_idx])
            draw_heatmap(
                axes[agent_idx, 2],
                pre_comm_heatmap,
                title="Agent {} pre_comm feature\nshape={}".format(
                    agent_idx, tuple(pre_comm_features[agent_idx].shape)
                )
            )

            meta["agents"].append({
                "agent_idx": int(agent_idx),
                "cav_id": str(cav_id),
                "raw_lidar_points": int(lidar_np.shape[0]),
                "spatial_features_shape": list(spatial_features[agent_idx].shape),
                "pre_comm_features_shape": list(pre_comm_features[agent_idx].shape),
            })

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        png_path = os.path.join(opt.output_dir, "sample_{:06d}.png".format(data_idx))
        plt.savefig(png_path, dpi=200)
        plt.close(fig)

        json_path = os.path.join(opt.output_dir, "sample_{:06d}.json".format(data_idx))
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        if opt.save_tensors:
            tensor_path = os.path.join(opt.output_dir, "sample_{:06d}.pt".format(data_idx))
            torch.save({
                "dataset_index": data_idx,
                "spatial_features": spatial_features.cpu(),
                "pre_comm_features": pre_comm_features.cpu(),
            }, tensor_path)

        summary.append(meta)
        print("Saved:", png_path)

    summary_path = os.path.join(opt.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()