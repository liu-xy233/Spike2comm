# -*- coding: utf-8 -*-
"""
V2X-Sim 2.0 raw PKL/LiDAR reader.

Responsibilities:
    1. Load CoSDH-compatible train/val/test info PKLs.
    2. Resolve legacy absolute LiDAR paths to the current raw dataset root.
    3. Read per-agent 4x4 lidar poses and global 10D GT boxes.
    4. Read raw float32 N x 5 LiDAR and expose the first four features.
    5. Keep raw agent IDs unchanged (Agent 1 is the fixed Ego).

This class deliberately does not perform:
    - communication-range filtering;
    - coordinate projection;
    - GT merging/de-duplication;
    - voxelization;
    - model-specific collation.

Those operations belong to the intermediate-fusion dataset.
"""

import os
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class V2XSimBaseDataset(Dataset):
    """Read one V2X-Sim 2.0 split from CoSDH info PKL files."""

    VALID_SPLITS = ("train", "val", "test")

    DEFAULT_RAW_ROOT = (
        "/data/v2xsim/V2X-Sim-2.0"
    )
    DEFAULT_INFO_ROOT = (
        "/data/v2xsim/v2xsim2_info"
    )
    DEFAULT_LEGACY_ROOT = (
        "/GPFS/rhome/yifanlu/workspace/dataset/v2xsim2-complete"
    )

    RAW_POINT_FEATURES = 5
    MODEL_POINT_FEATURES = 4

    def __init__(self, params=None, split=None, train=None):
        """
        Args:
            params: V2X-Sim configuration dictionary. It accepts either
                    a complete OpenCOOD config containing a 'v2xsim'
                    section or the section itself.
            split: Explicit split: train / val / test.
            train: Compatibility fallback. True -> train;
                   False -> configured eval_split (default val).
        """
        if params is None:
            params = {}

        self.params = params
        self.cfg = params.get("v2xsim", params)

        if split is None:
            if train is None:
                split = self.cfg.get("split", "train")
            elif train:
                split = "train"
            else:
                split = self.cfg.get("eval_split", "val")

        split = str(split).lower()
        if split not in self.VALID_SPLITS:
            raise ValueError(
                "Invalid V2X-Sim split '{}'; expected one of {}"
                .format(split, self.VALID_SPLITS)
            )

        self.split = split

        self.raw_root = Path(
            self.cfg.get("raw_root", self.DEFAULT_RAW_ROOT)
        ).expanduser()

        self.info_root = Path(
            self.cfg.get("info_root", self.DEFAULT_INFO_ROOT)
        ).expanduser()

        self.legacy_root = Path(
            self.cfg.get("legacy_root", self.DEFAULT_LEGACY_ROOT)
        ).expanduser()

        self.max_cav = int(self.cfg.get("max_cav", 5))

        self.raw_point_features = int(
            self.cfg.get(
                "raw_point_features",
                self.RAW_POINT_FEATURES
            )
        )

        self.num_point_features = int(
            self.cfg.get(
                "num_point_features",
                self.MODEL_POINT_FEATURES
            )
        )

        self.check_finite_points = bool(
            self.cfg.get("check_finite_points", True)
        )

        if self.raw_point_features != 5:
            raise ValueError(
                "This V2X-Sim PKL protocol expects 5 raw point features."
            )

        if self.num_point_features != 4:
            raise ValueError(
                "The first implementation expects 4 model point features."
            )

        pkl_name = "v2xsim_infos_{}.pkl".format(self.split)
        self.info_path = Path(
            self.cfg.get("info_files", {}).get(
                self.split,
                str(self.info_root / pkl_name)
            )
        )

        if not self.info_path.is_file():
            raise FileNotFoundError(
                "V2X-Sim info PKL not found: {}".format(self.info_path)
            )

        with open(str(self.info_path), "rb") as f:
            self.records = pickle.load(f, encoding="latin1")

        if not isinstance(self.records, (list, tuple)):
            raise TypeError(
                "Expected PKL content to be a list/tuple, got {}"
                .format(type(self.records).__name__)
            )

        self.records = list(self.records)

        if not self.records:
            raise RuntimeError(
                "V2X-Sim split '{}' is empty.".format(self.split)
            )

        self._validate_record_schema()

        print(
            "[V2XSimBaseDataset] split={} samples={} "
            "max_cav={} raw_root={}".format(
                self.split,
                len(self.records),
                self.max_cav,
                self.raw_root
            )
        )

    def __len__(self):
        return len(self.records)

    def _validate_record_schema(self):
        """Validate metadata keys without loading LiDAR files."""
        required_common = (
            "agent_num",
            "sample_token",
            "timestamp",
        )

        for index, record in enumerate(self.records):
            if not isinstance(record, dict):
                raise TypeError(
                    "Record {} is not a dict.".format(index)
                )

            for key in required_common:
                if key not in record:
                    raise KeyError(
                        "Record {} missing key '{}'".format(index, key)
                    )

            agent_num = int(record["agent_num"])

            if agent_num < 1 or agent_num > self.max_cav:
                raise ValueError(
                    "Record {} has agent_num={}, max_cav={}"
                    .format(index, agent_num, self.max_cav)
                )

            for agent_id in range(1, agent_num + 1):
                required = (
                    "lidar_path_{}".format(agent_id),
                    "lidar_pose_{}".format(agent_id),
                    "labels_{}".format(agent_id),
                )

                for key in required:
                    if key not in record:
                        raise KeyError(
                            "Record {} agent {} missing '{}'"
                            .format(index, agent_id, key)
                        )

    def _resolve_lidar_path(self, original_path):
        """
        Map the CoSDH legacy absolute path to the current raw data root.

        Supported:
            - already valid absolute path;
            - legacy absolute path;
            - path relative to raw_root;
            - path beginning at the 'sweeps/' directory.
        """
        original = Path(str(original_path)).expanduser()

        # Already points to an existing file.
        if original.is_absolute() and original.is_file():
            return original.resolve()

        if original.is_absolute():
            try:
                relative = original.relative_to(self.legacy_root)
            except ValueError:
                # Explicit fallback: preserve the path below "sweeps".
                parts = original.parts

                if "sweeps" not in parts:
                    raise ValueError(
                        "Cannot map legacy LiDAR path '{}' to raw_root '{}'. "
                        "It is outside legacy_root and has no sweeps/ anchor."
                        .format(original, self.raw_root)
                    )

                sweep_index = parts.index("sweeps")
                relative = Path(*parts[sweep_index:])
        else:
            relative = original

        resolved = (self.raw_root / relative).resolve()

        if not resolved.is_file():
            raise FileNotFoundError(
                "Mapped LiDAR file does not exist.\n"
                "Original: {}\n"
                "Mapped:   {}".format(original, resolved)
            )

        return resolved

    @staticmethod
    def _read_pose(record, agent_id):
        key = "lidar_pose_{}".format(agent_id)
        pose = np.asarray(record[key], dtype=np.float64)

        if pose.shape != (4, 4):
            raise ValueError(
                "{} must have shape (4,4), got {}"
                .format(key, pose.shape)
            )

        if not np.isfinite(pose).all():
            raise ValueError(
                "{} contains NaN/Inf.".format(key)
            )

        return pose.copy()

    @staticmethod
    def _read_global_gt(record, agent_id):
        key = "labels_{}".format(agent_id)
        label = record[key]

        if not isinstance(label, dict):
            raise TypeError(
                "{} must be a dict, got {}"
                .format(key, type(label).__name__)
            )

        if "gt_boxes_global" not in label:
            raise KeyError(
                "{} missing gt_boxes_global".format(key)
            )

        if "gt_object_ids" not in label:
            raise KeyError(
                "{} missing gt_object_ids".format(key)
            )

        boxes = np.asarray(
            label["gt_boxes_global"],
            dtype=np.float32
        )

        if boxes.size == 0:
            boxes = np.zeros((0, 10), dtype=np.float32)

        if boxes.ndim != 2 or boxes.shape[1] != 10:
            raise ValueError(
                "{} gt_boxes_global must be (N,10), got {}"
                .format(key, boxes.shape)
            )

        if not np.isfinite(boxes).all():
            raise ValueError(
                "{} gt_boxes_global contains NaN/Inf".format(key)
            )

        object_ids = np.asarray(
            label["gt_object_ids"]
        ).reshape(-1).tolist()

        if len(object_ids) != len(boxes):
            raise ValueError(
                "{} has {} boxes but {} object IDs."
                .format(key, len(boxes), len(object_ids))
            )

        gt_names = label.get("gt_names", None)

        if gt_names is not None:
            gt_names = np.asarray(gt_names).reshape(-1).tolist()

            if len(gt_names) != len(boxes):
                raise ValueError(
                    "{} has {} boxes but {} gt_names."
                    .format(key, len(boxes), len(gt_names))
                )

        return {
            "gt_boxes_global": boxes.copy(),
            "gt_object_ids": list(object_ids),
            "gt_names": (
                None if gt_names is None else list(gt_names)
            ),
        }

    def get_record_metadata(self, index):
        """
        Return one sample's agent metadata without reading point clouds.

        This method is intended for the intermediate-fusion dataset:
        it can filter agents by communication range before loading LiDAR.
        """
        if index < 0:
            index += len(self.records)

        if index < 0 or index >= len(self.records):
            raise IndexError(index)

        record = self.records[index]
        agent_num = int(record["agent_num"])

        agents = OrderedDict()

        for agent_id in range(1, agent_num + 1):
            path_key = "lidar_path_{}".format(agent_id)

            lidar_path = self._resolve_lidar_path(
                record[path_key]
            )

            pose = self._read_pose(record, agent_id)
            gt = self._read_global_gt(record, agent_id)

            agents[agent_id] = {
                "agent_id": agent_id,
                "is_ego": agent_id == 1,
                "lidar_path": str(lidar_path),
                "lidar_pose": pose,
                "gt_boxes_global": gt["gt_boxes_global"],
                "gt_object_ids": gt["gt_object_ids"],
                "gt_names": gt["gt_names"],
            }

        return {
            "split": self.split,
            "sample_index": int(index),
            "sample_token": record["sample_token"],
            "token": record.get("token"),
            "timestamp": record["timestamp"],
            "agent_num": agent_num,
            "ego_agent_id": 1,
            "agents": agents,
        }

    def load_lidar_raw(self, agent_record):
        """Read one raw LiDAR file as float32 (N,5)."""
        path = Path(agent_record["lidar_path"])

        if not path.is_file():
            raise FileNotFoundError(str(path))

        raw = np.fromfile(str(path), dtype=np.float32)

        if raw.size % self.raw_point_features != 0:
            raise ValueError(
                "LiDAR float count {} is not divisible by {}: {}"
                .format(
                    raw.size,
                    self.raw_point_features,
                    path
                )
            )

        points = raw.reshape(
            -1, self.raw_point_features
        )

        if self.check_finite_points and not np.isfinite(points).all():
            raise ValueError(
                "LiDAR contains NaN/Inf: {}".format(path)
            )

        return points

    def load_lidar(self, agent_record):
        """
        Return model input points as float32 (N,4).

        CoSDH's raw V2X-Sim files contain five float32 features.
        The E3DSNN pipeline uses XYZ + intensity.
        """
        points = self.load_lidar_raw(agent_record)

        return np.ascontiguousarray(
            points[:, :self.num_point_features],
            dtype=np.float32
        )

    def __getitem__(self, index):
        """
        Load all agents for standalone inspection.

        The intermediate-fusion dataset should preferably call
        get_record_metadata(), filter by range, then load_lidar()
        only for selected agents.
        """
        sample = self.get_record_metadata(index)

        for agent_id, agent_record in sample["agents"].items():
            agent_record["lidar_np"] = self.load_lidar(
                agent_record
            )

        return sample
