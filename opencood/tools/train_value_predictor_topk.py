"""使用离线 Oracle utility 训练 Counterfactual Value Predictor。"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.value_predictor_dataset import ValuePredictorDataset
from opencood.models.comm_modules.value_prediction_loss import (
    ValuePredictionLoss,
)
from opencood.models.comm_modules.value_predictor_factory import (
    build_value_predictor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Value Predictor.")
    parser.add_argument("--config", "-c", required=True)
    return parser.parse_args()


def save_checkpoint(path, model, optimizer, epoch, config):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "config": config,
        },
        path,
    )


def main():
    args = parse_args()

    import opencood.hypes_yaml.yaml_utils as yaml_utils

    config = yaml_utils.load_yaml(args.config)
    training = config["training"]
    dataset_config = config["dataset"]
    seed = int(training.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = ValuePredictorDataset(
        dataset_config["feature_dir"],
        dataset_config["utility_path"],
        in_channels=int(config["model"].get("in_channels", 256)),
        num_patch=int(config["model"].get("num_patch", 1050)),
        return_valid_mask=True,
        require_valid_mask=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=bool(training.get("pin_memory", True)),
        drop_last=False,
    )

    device_name = training.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    model = build_value_predictor(config["model"]).to(device)
    from opencood.models.comm_modules.value_prediction_loss_topk_masked import (
        TopKHardValuePredictionLoss
    )
    criterion = TopKHardValuePredictionLoss(config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )

    output_dir = os.path.abspath(
        os.path.expanduser(training["checkpoint_dir"])
    )
    os.makedirs(output_dir, exist_ok=True)
    epochs = int(training["epochs"])
    assert epochs > 0, "training.epochs must be positive."

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"total_loss": 0.0, "reg_loss": 0.0, "rank_loss": 0.0}
        for batch in loader:
            # DataLoader 增加 batch 维，得到训练输入及 dense label。
            ego_feature = batch["ego_feature"].to(device)
            metadata = batch["neighbor_metadata"].to(device)
            utility = batch["utility"].to(device)
            valid_mask = batch["valid_mask"].to(
                device=device, dtype=torch.bool
            )
            assert ego_feature.dim() == 4 and ego_feature.shape[1:] == (
                128, 50, 126
            )
            assert metadata.dim() == 2 and metadata.shape[1] == 4
            assert utility.dim() == 2 and utility.shape[1] == 1050
            assert valid_mask.shape == utility.shape
            assert bool(valid_mask.any(dim=1).all()), "Every sample needs at least one valid patch."

            optimizer.zero_grad()
            value_map = model(ego_feature, metadata)
            assert value_map.shape == utility.shape
            losses = criterion(
                value_map, utility, valid_mask=valid_mask
            )
            losses["total_loss"].backward()
            grad_clip_norm = float(
                training.get("grad_clip_norm", 0.0)
            )
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
            optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach().item())

        num_batches = len(loader)
        if num_batches == 0:
            raise RuntimeError("Value Predictor training loader is empty.")
        print(
            "Epoch {}/{} total={:.6f} reg={:.6f} rank={:.6f}".format(
                epoch,
                epochs,
                totals["total_loss"] / num_batches,
                totals["reg_loss"] / num_batches,
                totals["rank_loss"] / num_batches,
            )
        )
        save_every = int(training.get("save_every", 1))
        if save_every < 1:
            raise ValueError("save_every must be >= 1")
        if epoch % save_every != 0 and epoch != epochs:
            continue

        epoch_path = os.path.join(
            output_dir, "value_predictor_epoch_{:03d}.pt".format(epoch)
        )
        save_checkpoint(epoch_path, model, optimizer, epoch, config)
        save_checkpoint(
            os.path.join(output_dir, "value_predictor_latest.pt"),
            model,
            optimizer,
            epoch,
            config,
        )


if __name__ == "__main__":
    main()
