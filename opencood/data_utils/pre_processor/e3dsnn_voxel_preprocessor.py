# -*- coding: utf-8 -*-
"""E-3DSNN-style sparse voxel preprocessing for DAIR/Where2comm inputs.

This preprocessor keeps the OpenCOOD dataset contract intact while producing
per-voxel sparse features suitable for a spconv 3D backbone.  Compared with the
PointPillar path, voxel points are reduced before collation instead of being
passed to PillarVFE.
"""

import numpy as np
import torch

from opencood.data_utils.pre_processor.sp_voxel_preprocessor import (
    SpVoxelPreprocessor,
)


class E3DSNNVoxelPreprocessor(SpVoxelPreprocessor):
    """Generate sparse voxel coordinates and mean/SVC-style voxel features."""

    def __init__(self, preprocess_params, train):
        super(E3DSNNVoxelPreprocessor, self).__init__(preprocess_params, train)
        args = self.params.get("args", {})
        self.encoding = args.get("encoding", "mean").lower()
        self.num_point_features = int(args.get("num_point_features", 4))
        self.num_time_bins = int(args.get("num_time_bins", 1))
        self.clip_feature_range = args.get("clip_feature_range", None)

        if self.encoding not in ["mean", "svc"]:
            raise ValueError(
                "E3DSNNVoxelPreprocessor encoding must be 'mean' or 'svc', "
                "got '{}'.".format(self.encoding)
            )

    def _reduce_voxels(self, voxels, num_points):
        if voxels.shape[0] == 0:
            channels = self.num_point_features * self.num_time_bins
            return np.zeros((0, channels), dtype=np.float32)

        voxels = voxels[:, :, : self.num_point_features].astype(np.float32, copy=False)
        counts = np.maximum(num_points.reshape(-1, 1), 1).astype(np.float32)
        summed = voxels.sum(axis=1)
        features = summed / counts

        if self.encoding == "svc":
            features = self._encode_spike_voxels(features)

        if self.clip_feature_range is not None:
            lo, hi = self.clip_feature_range
            features = np.clip(features, lo, hi)

        return features.astype(np.float32, copy=False)

    def _encode_spike_voxels(self, features):
        """Lightweight spike voxel coding compatible with static LiDAR frames.

        E-3DSNN represents voxel input as spike-like discrete activations.  DAIR
        samples here are single frames, so we quantize normalized voxel features
        into repeated temporal bins and flatten them back to a sparse feature
        vector.  The sparse backbone can then consume them as normal channels.
        """

        if self.num_time_bins <= 1:
            return np.floor(np.clip(features, 0.0, 4.0) + 0.5)

        denom = np.maximum(np.abs(features).max(axis=0, keepdims=True), 1e-6)
        normalized = np.clip((features / denom + 1.0) * 0.5, 0.0, 1.0)
        thresholds = (
            np.arange(1, self.num_time_bins + 1, dtype=np.float32)
            / float(self.num_time_bins)
        )
        spikes = normalized[:, None, :] >= thresholds[None, :, None]
        return spikes.astype(np.float32).reshape(features.shape[0], -1)

    def preprocess(self, pcd_np):
        voxel_output = self.voxel_generator.generate(pcd_np)
        if isinstance(voxel_output, dict):
            voxels = voxel_output["voxels"]
            coordinates = voxel_output["coordinates"]
            num_points = voxel_output["num_points_per_voxel"]
        else:
            voxels, coordinates, num_points = voxel_output

        return {
            "voxel_features": self._reduce_voxels(voxels, num_points),
            "voxel_coords": coordinates,
            "voxel_num_points": num_points,
        }

    @staticmethod
    def collate_batch_list(batch):
        voxel_features = []
        voxel_num_points = []
        voxel_coords = []

        for batch_idx, item in enumerate(batch):
            voxel_features.append(item["voxel_features"])
            voxel_num_points.append(item["voxel_num_points"])
            coords = item["voxel_coords"]
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)), mode="constant",
                       constant_values=batch_idx)
            )

        return {
            "voxel_features": torch.from_numpy(np.concatenate(voxel_features)).float(),
            "voxel_coords": torch.from_numpy(np.concatenate(voxel_coords)).int(),
            "voxel_num_points": torch.from_numpy(np.concatenate(voxel_num_points)).int(),
        }

    @staticmethod
    def collate_batch_dict(batch):
        voxel_features = torch.from_numpy(
            np.concatenate(batch["voxel_features"])
        ).float()
        voxel_num_points = torch.from_numpy(
            np.concatenate(batch["voxel_num_points"])
        ).int()

        voxel_coords = []
        for batch_idx, coords in enumerate(batch["voxel_coords"]):
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)), mode="constant",
                       constant_values=batch_idx)
            )

        return {
            "voxel_features": voxel_features,
            "voxel_coords": torch.from_numpy(np.concatenate(voxel_coords)).int(),
            "voxel_num_points": voxel_num_points,
        }
