import argparse
import csv
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from opencood.data_utils.value_predictor_dataset import (
    ValuePredictorDataset,
)
from opencood.models.comm_modules.value_predictor_factory import (
    build_value_predictor,
    validate_value_predictor_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="outputs/value_predictor_formal_ep26/"
                "value_predictor_aligned_masked_v1.yaml",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    eval_dataset = config.get("evaluation")
    if (eval_dataset and eval_dataset.get("feature_dir")
            and eval_dataset.get("utility_path")):
        feature_dir = Path(eval_dataset["feature_dir"])
        label_path = Path(eval_dataset["utility_path"])
    else:
        base = Path("outputs/value_predictor_formal_ep26")
        feature_dir = base / "features_val_full"
        label_path = base / "oracle_val_ego_aligned_full.pt"

    assert feature_dir.is_dir(), "Validation feature directory missing"
    assert label_path.is_file(), "Aligned validation labels missing"
    assert args.topk > 0, "--topk must be positive"

    device_name = config["training"].get("device", "cuda")
    device = torch.device(
        device_name if (
            not str(device_name).startswith("cuda")
            or torch.cuda.is_available()
        ) else "cpu"
    )

    dataset = ValuePredictorDataset(
        str(feature_dir),
        str(label_path),
        in_channels=int(config["model"].get("in_channels", 256)),
        num_patch=int(config["model"].get("num_patch", 1050)),
        return_valid_mask=True,
        require_valid_mask=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 2)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = build_value_predictor(config["model"]).to(device)

    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint, map_location=device
        )
        if isinstance(checkpoint, dict):
            state = checkpoint.get(
                "model_state_dict",
                checkpoint.get("state_dict"),
            )
        else:
            state = None

        if state is None:
            raise RuntimeError(
                "Checkpoint has no model_state_dict/state_dict"
            )

        validate_value_predictor_checkpoint(model, checkpoint)
        model.load_state_dict(state, strict=True)
        print("[INFO] Loaded checkpoint:", args.checkpoint)
    else:
        print(
            "[SMOKE TEST] Randomly initialized model; "
            "results are not model-performance measurements."
        )

    model.eval()

    recalls = []
    captures = []
    prediction_stds = []
    random_captures = []
    detail = []
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            ego = batch["ego_feature"].to(device).float()
            metadata = batch["neighbor_metadata"].to(device).float()
            utility = batch["utility"].to(device).float()
            valid_mask = batch["valid_mask"].to(device).bool()

            prediction = model(ego, metadata)
            if isinstance(prediction, dict):
                prediction = prediction["value_map"]

            assert prediction.shape == utility.shape
            assert prediction.shape == valid_mask.shape
            assert torch.isfinite(prediction).all()
            assert torch.isfinite(utility).all()

            for i in range(prediction.shape[0]):
                valid_ids = torch.where(valid_mask[i])[0]
                n_valid = valid_ids.numel()

                if n_valid < 2:
                    raise RuntimeError(
                        "Insufficient valid patches in sample."
                    )

                k = min(args.topk, n_valid)

                pred_ids = valid_ids[
                    torch.topk(
                        prediction[i, valid_ids], k
                    ).indices
                ]
                oracle_ids = valid_ids[
                    torch.topk(
                        utility[i, valid_ids], k
                    ).indices
                ]

                overlap = torch.isin(
                    pred_ids, oracle_ids
                ).sum().item()

                recall = overlap / float(k)

                # Positive utility mass captured by predicted Top-K.
                positive_mass = utility[i, valid_ids].clamp_min(0).sum()
                selected_mass = utility[i, pred_ids].clamp_min(0).sum()

                if positive_mass.item() > 1e-12:
                    capture = (
                        selected_mass / positive_mass
                    ).item()
                    captures.append(capture)

                pred_std = prediction[i, valid_ids].std(
                    unbiased=False
                ).item()

                # Exact expected capture for uniform random Top-K:
                # each valid patch has inclusion probability k / n_valid.
                random_capture = k / float(n_valid)

                recalls.append(recall)
                prediction_stds.append(pred_std)
                random_captures.append(random_capture)

                if "sample_id" in batch:
                    sample_id = batch["sample_id"][i]
                    if torch.is_tensor(sample_id):
                        sample_id = sample_id.item()
                else:
                    # Dataset 未返回 sample_id。
                    # DataLoader 使用 shuffle=False，使用顺序索引记录诊断结果。
                    sample_id = total_samples

                detail.append({
                    "sample_id": int(sample_id),
                    "valid_patches": int(n_valid),
                    "topk": int(k),
                    "recall_at_k_pct": 100 * recall,
                    "positive_utility_capture_pct": (
                        100 * capture
                        if positive_mass.item() > 1e-12
                        else ""
                    ),
                    "random_expected_capture_pct": (
                        100 * random_capture
                    ),
                    "prediction_std": pred_std,
                })

                total_samples += 1

                if (
                    args.max_samples > 0
                    and total_samples >= args.max_samples
                ):
                    break

            if (
                args.max_samples > 0
                and total_samples >= args.max_samples
            ):
                break

    if not recalls:
        raise RuntimeError("No validation samples were evaluated.")

    summary = {
        "checkpoint": args.checkpoint or "random_init_smoke_test",
        "samples": total_samples,
        "topk": args.topk,
        "mean_valid_patches": sum(
            x["valid_patches"] for x in detail
        ) / len(detail),
        "recall_at_k_pct": 100 * sum(recalls) / len(recalls),
        "positive_utility_capture_at_k_pct": (
            100 * sum(captures) / len(captures)
            if captures else float("nan")
        ),
        "capture_samples": len(captures),
        "random_expected_capture_pct": (
            100 * sum(random_captures) / len(random_captures)
        ),
        "mean_prediction_std": (
            sum(prediction_stds) / len(prediction_stds)
        ),
    }

    print("\n========== VALIDATION SUMMARY ==========")
    for key, value in summary.items():
        print("{}: {}".format(key, value))

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(summary.keys())
            )
            writer.writeheader()
            writer.writerow(summary)
        print("[OK] Summary saved:", out)

    print("[OK] Validation evaluator completed.")


if __name__ == "__main__":
    main()
