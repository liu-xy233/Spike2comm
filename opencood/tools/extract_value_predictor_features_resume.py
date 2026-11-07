"""从冻结的 Where2comm + E3D-SNN 模型抽取 ValueNet 输入。"""

import argparse
import os

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ego BEV features for Value Predictor training."
    )
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def build_neighbor_metadata(pairwise_t_matrix: torch.Tensor) -> torch.Tensor:
    """由原始 pairwise transform 生成 [dx,dy,yaw,distance]。"""
    assert pairwise_t_matrix.dim() in (4, 5), (
        "pairwise_t_matrix must be [L,L,4,4] or [B,L,L,4,4]."
    )
    if pairwise_t_matrix.dim() == 5:
        assert pairwise_t_matrix.shape[0] == 1, (
            "Extraction requires batch size 1."
        )
        pairwise_t_matrix = pairwise_t_matrix[0]
    assert pairwise_t_matrix.shape[0] >= 2, "A neighbor is required."
    # T[0,1] 与 fusion 中 t_matrix[0,1] 相同：neighbor 1 -> ego 0。
    transform = pairwise_t_matrix[0, 1]
    assert transform.shape == (4, 4), "Neighbor transform must be [4,4]."
    dx = transform[0, 3]
    dy = transform[1, 3]
    yaw = torch.atan2(transform[1, 0], transform[0, 0])
    distance = torch.sqrt(dx.square() + dy.square())
    metadata = torch.stack((dx, dy, yaw, distance)).float()
    assert metadata.shape == (4,), "neighbor_metadata must be [4]."
    assert torch.isfinite(metadata).all(), "neighbor_metadata contains NaN/Inf."
    return metadata


def build_loader(hypes, split, num_workers):
    from opencood.data_utils.datasets import build_dataset

    dataset = build_dataset(hypes, visualize=False, train=split == "train")
    # batch_test 保留 sample_idx；batch size 1 同样可用于 train split 抽取。
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    return loader


def main():
    args = parse_args()

    from tqdm import tqdm

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils

    config = yaml_utils.load_yaml(args.config)
    extraction = config["extraction"]
    detector_hypes = yaml_utils.load_yaml(extraction["detector_hypes"])
    split = str(extraction.get("split", "train"))
    if split not in ("train", "val"):
        raise ValueError("extraction.split must be train or val.")

    output_dir = os.path.abspath(os.path.expanduser(extraction["output_dir"]))
    os.makedirs(output_dir, exist_ok=True)
    loader = build_loader(
        detector_hypes, split, int(extraction.get("num_workers", 0))
    )

    device_name = extraction.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    model = train_utils.create_model(detector_hypes).to(device)
    _, model = train_utils.load_saved_model(
        extraction["checkpoint_dir"],
        model,
        extraction.get("eval_epoch"),
    )
    if not hasattr(model, "_extract_cached_context"):
        raise RuntimeError("Detector has no _extract_cached_context API.")
    model.eval()

    saved = 0
    for batch_index, batch_data in tqdm(
            enumerate(loader), total=len(loader)):
        if args.max_samples is not None and saved >= args.max_samples:
            break
        if batch_data is None:
            continue
        batch_data = train_utils.to_device(batch_data, device)
        ego_data = batch_data["ego"]
        sample_id = int(ego_data.get("sample_idx", batch_index))
        output_path = os.path.join(output_dir, "sample_{:04d}.pt".format(sample_id))
        if os.path.isfile(output_path):
            continue

        with torch.no_grad():
            cached_context = model._extract_cached_context(ego_data)
            assert int(cached_context["record_len"][0]) >= 2, (
                "sample_id {} has no neighbor.".format(sample_id)
            )
            # 通信前 ego BEV：[sum_agents,128,50,126] -> [128,50,126]。
            ego_feature = cached_context["spatial_features"][0].detach().cpu()
            metadata = build_neighbor_metadata(
                ego_data["pairwise_t_matrix"]
            ).detach().cpu()

        assert ego_feature.shape == (128, 50, 126), (
            "sample_id {} ego_feature must be [128,50,126], got {}.".format(
                sample_id, tuple(ego_feature.shape)
            )
        )
        assert torch.isfinite(ego_feature).all(), (
            "sample_id {} ego_feature contains NaN/Inf.".format(sample_id)
        )
        assert metadata.shape == (4,), "neighbor_metadata must be [4]."
        output = {
            "sample_id": sample_id,
            "ego_feature": ego_feature.float(),
            "neighbor_metadata": metadata.float(),
        }
        output_path = os.path.join(
            output_dir, "sample_{:04d}.pt".format(sample_id)
        )
        torch.save(output, output_path)
        saved += 1

    print("Saved {} feature samples to {}.".format(saved, output_dir))


if __name__ == "__main__":
    main()
