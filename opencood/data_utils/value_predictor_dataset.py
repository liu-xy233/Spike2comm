"""Counterfactual Value Predictor 的离线训练数据集。"""

import os
from glob import glob
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class ValuePredictorDataset(Dataset):
    """按 sample_id 对齐离线 ego feature 和 Oracle utility。"""

    def __init__(self, feature_dir: str, dense_patch_utility_path: str,
                 num_patch: int = 1050, in_channels: int = 256,
                 return_valid_mask: bool = False,
                 require_valid_mask: bool = False):
        self.feature_dir = os.path.abspath(os.path.expanduser(feature_dir))
        self.utility_path = os.path.abspath(
            os.path.expanduser(dense_patch_utility_path)
        )
        self.num_patch = int(num_patch)
        self.in_channels = int(in_channels)
        self.return_valid_mask = bool(return_valid_mask or require_valid_mask)
        self.require_valid_mask = bool(require_valid_mask)
        self.valid_mask_by_sample = {}

        if not os.path.isdir(self.feature_dir):
            raise FileNotFoundError(
                "Feature directory does not exist: {}".format(self.feature_dir)
            )
        if not os.path.isfile(self.utility_path):
            raise FileNotFoundError(
                "Utility file does not exist: {}".format(self.utility_path)
            )

        label_records = torch.load(self.utility_path, map_location="cpu")
        if not isinstance(label_records, list):
            raise TypeError("dense_patch_utility.pt must contain list[dict].")
        self.utility_by_sample = self._index_utilities(label_records)
        self.samples = self._index_features()
        if not self.samples:
            raise RuntimeError(
                "No sample_*.pt feature files found in {}.".format(
                    self.feature_dir
                )
            )

    def _index_utilities(self, records: List[Dict]) -> Dict[int, torch.Tensor]:
        utilities = {}
        for record in records:
            if not isinstance(record, dict) or "sample_id" not in record:
                raise ValueError("Every utility record must contain sample_id.")
            sample_id = int(record["sample_id"])
            if sample_id in utilities:
                raise ValueError(
                    "Duplicate utility sample_id {}.".format(sample_id)
                )
            if "patch_size" in record:
                assert tuple(record["patch_size"]) == (2, 3), (
                    "sample_id {} patch_size must be [2,3].".format(sample_id)
                )
            if "feature_size" in record:
                assert tuple(record["feature_size"]) == (50, 126), (
                    "sample_id {} feature_size must be [50,126].".format(
                        sample_id
                    )
                )
            utility = torch.as_tensor(record.get("utility"), dtype=torch.float32)
            utility = utility.flatten()
            assert utility.shape == (self.num_patch,), (
                "sample_id {} utility must be [{}], got {}.".format(
                    sample_id, self.num_patch, tuple(utility.shape)
                )
            )
            utilities[sample_id] = utility
            if not torch.isfinite(utility).all():
                raise ValueError("Non-finite utility for sample_id {}".format(sample_id))
            if self.return_valid_mask:
                if "valid_mask" not in record:
                    if self.require_valid_mask:
                        raise ValueError("Missing aligned valid_mask for sample_id {}".format(sample_id))
                    mask = torch.ones(self.num_patch, dtype=torch.bool)
                else:
                    mask = torch.as_tensor(record["valid_mask"]).flatten()
                    if mask.shape != utility.shape or not torch.all((mask == 0) | (mask == 1)):
                        raise ValueError("valid_mask must contain one boolean per patch")
                    mask = mask.bool()
                if not mask.any():
                    raise ValueError("Every sample needs at least one valid patch")
                self.valid_mask_by_sample[sample_id] = mask
        return utilities

    def _index_features(self) -> List[Tuple[int, str]]:
        samples = []
        seen_ids = set()
        pattern = os.path.join(self.feature_dir, "sample_*.pt")
        for path in sorted(glob(pattern)):
            record = torch.load(path, map_location="cpu")
            if not isinstance(record, dict) or "sample_id" not in record:
                raise ValueError(
                    "Feature file {} must contain sample_id.".format(path)
                )
            sample_id = int(record["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(
                    "Duplicate feature sample_id {}.".format(sample_id)
                )
            if sample_id not in self.utility_by_sample:
                raise KeyError(
                    "Feature sample_id {} has no Oracle utility label.".format(
                        sample_id
                    )
                )
            seen_ids.add(sample_id)
            samples.append((sample_id, path))
        return sorted(samples, key=lambda item: item[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_id, path = self.samples[index]
        record = torch.load(path, map_location="cpu")
        assert int(record["sample_id"]) == sample_id, (
            "Feature sample_id changed after dataset initialization."
        )

        # 离线特征去除了 batch 维：[C,50,126]，C 由模型配置指定。
        ego_feature = torch.as_tensor(record.get("ego_feature")).float()
        neighbor_metadata = torch.as_tensor(
            record.get("neighbor_metadata")
        ).float()
        utility = self.utility_by_sample[sample_id].clone()

        assert ego_feature.shape == (self.in_channels, 50, 126), (
            "ego_feature must be [{},50,126], got {}.".format(
                self.in_channels, tuple(ego_feature.shape)
            )
        )
        assert neighbor_metadata.shape == (4,), (
            "neighbor_metadata must be [4], got {}.".format(
                tuple(neighbor_metadata.shape)
            )
        )
        assert utility.shape == (self.num_patch,), (
            "utility must be [{}], got {}.".format(
                self.num_patch, tuple(utility.shape)
            )
        )
        if not torch.isfinite(ego_feature).all() or not torch.isfinite(neighbor_metadata).all():
            raise ValueError("Features and metadata must be finite")
        result = {
            "ego_feature": ego_feature,
            "neighbor_metadata": neighbor_metadata,
            "utility": utility,
        }
        if self.return_valid_mask:
            result["valid_mask"] = self.valid_mask_by_sample[sample_id].clone()
            result["sample_id"] = sample_id
        return result
