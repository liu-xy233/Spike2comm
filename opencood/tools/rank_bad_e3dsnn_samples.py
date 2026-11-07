import os
import csv
import argparse

import torch
from tqdm import tqdm

from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools import inference_utils
from opencood.utils import eval_utils_global as eval_utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--eval_epoch",
        type=int,
        default=28
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--min_gt",
        type=int,
        default=1
    )

    return parser.parse_args()


def new_result_stat():
    return {
        0.3: {
            "tp": [],
            "fp": [],
            "score": [],
            "gt": 0
        },
        0.5: {
            "tp": [],
            "fp": [],
            "score": [],
            "gt": 0
        },
        0.7: {
            "tp": [],
            "fp": [],
            "score": [],
            "gt": 0
        },
    }


def safe_sum(x):
    if len(x) == 0:
        return 0
    return int(sum(x))


def get_metric(stat, threshold):
    x = stat[threshold]

    tp = safe_sum(x["tp"])
    fp = safe_sum(x["fp"])
    gt = int(x["gt"])

    fn = max(gt - tp, 0)

    recall = float(tp) / gt if gt > 0 else 0.0

    precision = (
        float(tp) / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    if precision + recall > 0:
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt": gt,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def main():
    opt = parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)

    config_path = os.path.join(
        opt.model_dir,
        "config.yaml"
    )

    hypes = yaml_utils.load_yaml(
        config_path,
        None
    )

    dataset = build_dataset(
        hypes,
        visualize=False,
        train=False
    )

    model = train_utils.create_model(hypes)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    epoch_id, model = train_utils.load_saved_model(
        opt.model_dir,
        model,
        opt.eval_epoch
    )

    model.eval()

    print("Loaded epoch:", epoch_id)
    print("Dataset length:", len(dataset))

    num_samples = len(dataset)

    if opt.max_samples > 0:
        num_samples = min(
            num_samples,
            opt.max_samples
        )

    rows = []

    for idx in tqdm(range(num_samples)):

        sample = dataset[idx]

        batch_data = dataset.collate_batch_test(
            [sample]
        )

        if batch_data is None:
            continue

        batch_data = train_utils.to_device(
            batch_data,
            device
        )

        with torch.no_grad():

            pred_box_tensor, \
            pred_score, \
            gt_box_tensor = \
                inference_utils.inference_intermediate_fusion(
                    batch_data,
                    model,
                    dataset
                )

        stat = new_result_stat()

        for threshold in [0.3, 0.5, 0.7]:

            eval_utils.caluclate_tp_fp(
                pred_box_tensor,
                pred_score,
                gt_box_tensor,
                stat,
                threshold
            )

        m30 = get_metric(stat, 0.3)
        m50 = get_metric(stat, 0.5)
        m70 = get_metric(stat, 0.7)

        gt_num = m70["gt"]

        if gt_num < opt.min_gt:
            continue

        localization_drop = (
            m50["tp"] - m70["tp"]
        )

        localization_drop_rate = (
            float(localization_drop) / gt_num
            if gt_num > 0
            else 0.0
        )

        row = {
            "dataset_index": idx,

            "gt": gt_num,

            "tp30": m30["tp"],
            "fp30": m30["fp"],
            "recall30": m30["recall"],

            "tp50": m50["tp"],
            "fp50": m50["fp"],
            "fn50": m50["fn"],
            "precision50": m50["precision"],
            "recall50": m50["recall"],
            "f1_50": m50["f1"],

            "tp70": m70["tp"],
            "fp70": m70["fp"],
            "fn70": m70["fn"],
            "precision70": m70["precision"],
            "recall70": m70["recall"],
            "f1_70": m70["f1"],

            "localization_drop": localization_drop,
            "localization_drop_rate":
                localization_drop_rate,
        }

        rows.append(row)

    # -----------------------------------
    # Overall bad:
    # low IoU=0.7 F1 first
    # -----------------------------------
    overall_bad = sorted(
        rows,
        key=lambda x: (
            x["f1_70"],
            x["recall70"],
            -x["fn70"]
        )
    )

    # -----------------------------------
    # Localization-hard:
    # many TP50 disappear at TP70
    # -----------------------------------
    localization_bad = sorted(
        rows,
        key=lambda x: (
            -x["localization_drop_rate"],
            -x["localization_drop"],
            -x["recall50"]
        )
    )

    fieldnames = list(rows[0].keys())

    all_path = os.path.join(
        opt.output_dir,
        "all_samples.csv"
    )

    overall_path = os.path.join(
        opt.output_dir,
        "worst_overall.csv"
    )

    localization_path = os.path.join(
        opt.output_dir,
        "worst_localization.csv"
    )

    for path, data in [
        (all_path, rows),
        (overall_path, overall_bad),
        (localization_path, localization_bad),
    ]:

        with open(
            path,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(data)

    print("\n================================")
    print("Worst overall samples")
    print("================================")

    for x in overall_bad[:20]:

        print(
            "idx={:<5d} GT={:<3d} "
            "R50={:.3f} R70={:.3f} "
            "F1_70={:.3f} "
            "drop={}".format(
                x["dataset_index"],
                x["gt"],
                x["recall50"],
                x["recall70"],
                x["f1_70"],
                x["localization_drop"],
            )
        )

    print("\n================================")
    print("Worst localization samples")
    print("================================")

    for x in localization_bad[:20]:

        print(
            "idx={:<5d} GT={:<3d} "
            "TP50={:<3d} TP70={:<3d} "
            "R50={:.3f} R70={:.3f} "
            "drop={:<3d} drop_rate={:.3f}".format(
                x["dataset_index"],
                x["gt"],
                x["tp50"],
                x["tp70"],
                x["recall50"],
                x["recall70"],
                x["localization_drop"],
                x["localization_drop_rate"],
            )
        )

    print("\nSaved to:", opt.output_dir)


if __name__ == "__main__":
    main()
