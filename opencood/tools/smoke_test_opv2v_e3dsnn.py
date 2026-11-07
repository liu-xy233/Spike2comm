import yaml
import torch
import numpy as np

import os
import sys
from pathlib import Path

# Force this repository to take priority over other OpenCOOD projects.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT.resolve())

if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)

sys.path.insert(0, PROJECT_ROOT_STR)

import opencood
from opencood.data_utils import datasets as dataset_registry

print("[IMPORT] Project root:", PROJECT_ROOT)
print("[IMPORT] OpenCOOD:", Path(opencood.__file__).resolve())
print("[IMPORT] Dataset registry:",
      Path(dataset_registry.__file__).resolve())

actual_root = os.path.commonpath([
    str(Path(opencood.__file__).resolve()),
    PROJECT_ROOT_STR
])

assert actual_root == PROJECT_ROOT_STR, (
    "Wrong OpenCOOD package imported: "
    + str(opencood.__file__)
)

assert (
    "IntermediateFusionDatasetOPV2VE3DSNN"
    in dataset_registry.__all__
), "OPV2V E3DSNN Dataset missing from active registry"

print("[IMPORT] OPV2V Dataset registration: PASSED")

from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    return obj


def reset_snn(model):
    try:
        from spikingjelly.activation_based import functional
        functional.reset_net(model)
        return
    except (ImportError, AttributeError):
        pass

    try:
        from spikingjelly.clock_driven import functional
        functional.reset_net(model)
    except (ImportError, AttributeError):
        print("[WARN] SpikingJelly reset_net unavailable")


def inspect_output(output, expected_agents, batch_size, name):

    print("\n========== {} ==========".format(name))
    print("Output keys:", list(output.keys()))

    required = [
        "psm", "rm", "iou",
        "psm_single", "rm_single", "iou_single"
    ]

    for key in required:
        if key not in output:
            raise KeyError("Missing model output: " + key)

        value = output[key]

        print("{}: {}".format(key, tuple(value.shape)))

        assert torch.isfinite(value).all(), key

    for key in ["psm", "rm", "iou"]:
        assert output[key].shape[0] == batch_size

    for key in ["psm_single", "rm_single", "iou_single"]:
        assert output[key].shape[0] == expected_agents, (
            key, output[key].shape, expected_agents
        )

    assert output["psm"].shape[-2:] == (48, 176)
    assert output["rm"].shape[-2:] == (48, 176)
    assert output["iou"].shape[-2:] == (48, 176)

    assert output["psm_single"].shape[-2:] == (48, 176)
    assert output["rm_single"].shape[-2:] == (48, 176)
    assert output["iou_single"].shape[-2:] == (48, 176)

    print("{} PASSED".format(name))


def main():

    torch.set_num_threads(1)

    assert torch.cuda.is_available()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    cfg_path = (
        "opencood/hypes_yaml/opv2v/"
        "opv2v_e3dsnn_where2comm_phase2b.yaml"
    )

    with open(cfg_path, "r") as f:
        hypes = yaml.safe_load(f)

    print("=" * 70)
    print("OPV2V E3DSNN SMOKE TEST")
    print("=" * 70)

    # 1. Dataset
    print("\n[1] BUILD DATASET")

    dataset = build_dataset(
        hypes,
        visualize=False,
        train=True
    )

    print("Dataset length:", len(dataset))

    # Previous Phase 2A verified these two indices.
    sample3 = dataset[0]
    sample4 = dataset[579]

    n3 = int(sample3["ego"]["cav_num"])
    n4 = int(sample4["ego"]["cav_num"])

    print("Sample 0 agents:", n3)
    print("Sample 579 agents:", n4)

    assert n3 == 3
    assert n4 == 4

    # 2. Collate
    print("\n[2] COLLATE")

    batch4 = dataset.collate_batch_train([sample4])
    batch34 = dataset.collate_batch_train([sample3, sample4])

    assert batch4 is not None
    assert batch34 is not None

    print("Single record_len:",
          batch4["ego"]["record_len"].tolist())

    print("Mixed record_len:",
          batch34["ego"]["record_len"].tolist())

    # 3. Model and loss
    print("\n[3] BUILD MODEL")

    model = train_utils.create_model(hypes)
    criterion = train_utils.create_loss(hypes)

    model = model.to(device)

    print("Model:", model.__class__.__name__)

    print("Trainable parameters:",
          sum(p.numel() for p in model.parameters()
              if p.requires_grad))

    # 4. Single 4-agent forward
    print("\n[4] FOUR-AGENT FORWARD")

    model.eval()

    batch4_gpu = to_device(batch4, device)

    reset_snn(model)

    with torch.no_grad():
        output4 = model(batch4_gpu["ego"])

    inspect_output(
        output4,
        expected_agents=4,
        batch_size=1,
        name="FOUR_AGENT"
    )

    del output4
    torch.cuda.empty_cache()

    # 5. Mixed [3,4] forward
    print("\n[5] MIXED BATCH FORWARD")

    batch34_gpu = to_device(batch34, device)

    reset_snn(model)

    with torch.no_grad():
        output34 = model(batch34_gpu["ego"])

    inspect_output(
        output34,
        expected_agents=7,
        batch_size=2,
        name="MIXED_3_4"
    )

    del output34
    del batch34_gpu
    torch.cuda.empty_cache()

    # 6. Backward
    print("\n[6] BACKWARD")

    model.train()
    model.zero_grad(set_to_none=True)

    reset_snn(model)

    output = model(batch4_gpu["ego"])

    loss = criterion(
        output,
        batch4_gpu["ego"]["label_dict"]
    )

    assert torch.is_tensor(loss)
    assert loss.numel() == 1
    assert torch.isfinite(loss).all()

    print("Detection loss:", loss.item())

    loss.backward()

    total_grad = 0.0
    backbone_grad = 0.0
    fusion_grad = 0.0
    head_grad = 0.0
    nonzero = 0

    for name, p in model.named_parameters():

        if p.grad is None:
            continue

        g = p.grad.detach()

        assert torch.isfinite(g).all(), name

        value = g.abs().sum().item()

        total_grad += value

        if value > 0:
            nonzero += 1

        lname = name.lower()

        if "backbone" in lname:
            backbone_grad += value

        if "fusion" in lname or "fuse" in lname:
            fusion_grad += value

        if "head" in lname:
            head_grad += value

    print("Total gradient:", total_grad)
    print("Nonzero parameters:", nonzero)
    print("Backbone gradient:", backbone_grad)
    print("Fusion gradient:", fusion_grad)
    print("Head gradient:", head_grad)

    assert total_grad > 0
    assert nonzero > 0
    assert backbone_grad > 0
    assert head_grad > 0

    torch.cuda.synchronize()

    print("Peak GPU memory:",
          round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
          "GB")

    print("\n" + "=" * 70)
    print("PHASE 2B PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
