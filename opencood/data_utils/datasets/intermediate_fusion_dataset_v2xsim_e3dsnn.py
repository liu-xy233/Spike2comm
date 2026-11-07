# -*- coding: utf-8 -*-
"""V2X-Sim 2.0 intermediate fusion dataset for E3DSNN.

V1 protocol:
    - Agent 1 is always Ego.
    - Original Agent IDs and order are preserved.
    - Neighbor communication range is 50 m.
    - proj_first=False: each agent voxelizes its local LiDAR.
    - Pairwise transforms are computed from global 4x4 poses.
    - Global GT from all selected agents is projected to Ego coordinates.
    - GT is stably de-duplicated by global object ID.
    - Only cooperative/fused detection labels are generated.

The output contract follows the existing OPV2V E3DSNN dataset.
"""

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

import opencood.data_utils.post_processor as post_processor

from opencood.data_utils.datasets.v2xsim.basedataset import (
    V2XSimBaseDataset,
)
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.transformation_utils import tfm_to_pose


class IntermediateFusionDatasetV2XSimE3DSNN(Dataset):
    """Intermediate-fusion V2X-Sim 2.0 dataset."""

    def __init__(self, params, visualize=False, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.v2xsim_cfg = params.get("v2xsim", {})

        self.max_cav = int(
            params.get("train_params", {}).get(
                "max_cav",
                self.v2xsim_cfg.get("max_cav", 5)
            )
        )

        if self.max_cav < 1 or self.max_cav > 5:
            raise ValueError(
                "V2X-Sim V1 max_cav must be in [1,5], got {}"
                .format(self.max_cav)
            )

        self.comm_range = float(
            params.get(
                "comm_range",
                params.get("fusion", {}).get(
                    "args", {}
                ).get("comm_range", 50.0)
            )
        )

        if self.comm_range <= 0:
            raise ValueError("comm_range must be positive.")

        self.proj_first = bool(
            params.get("fusion", {}).get(
                "args", {}
            ).get("proj_first", False)
        )

        if self.proj_first:
            raise NotImplementedError(
                "V2X-Sim V1 requires proj_first=False. "
                "Feature-level alignment is handled by Where2comm."
            )

        self.lidar_range = np.asarray(
            params["preprocess"]["cav_lidar_range"],
            dtype=np.float32
        )

        self.gt_filter_range = list(
            params["postprocess"]["anchor_args"]["cav_lidar_range"]
        )

        self.max_num = int(
            params["postprocess"]["max_num"]
        )

        self.box_order = params["postprocess"].get(
            "order", "hwl"
        )

        if self.box_order != "hwl":
            raise ValueError(
                "V2X-Sim V1 with VoxelPostprocessor requires "
                "postprocess.order='hwl'."
            )

        self.pre_processor = build_preprocessor(
            params["preprocess"], train
        )

        self.post_processor = post_processor.build_postprocessor(
            params["postprocess"], train=train
        )

        if train:
            split = "train"
        else:
            split = self.v2xsim_cfg.get(
                "eval_split",
                self.v2xsim_cfg.get("split", "val")
            )

        self.split = str(split).lower()

        self.base_dataset = V2XSimBaseDataset(
            params=params,
            split=self.split
        )

        if self.base_dataset.max_cav < self.max_cav:
            raise ValueError(
                "Base reader max_cav={} is smaller than dataset max_cav={}"
                .format(self.base_dataset.max_cav, self.max_cav)
            )

        # Anchor boxes are independent of the sample.
        self.anchor_box = self.post_processor.generate_anchor_box()

        if self.anchor_box is None:
            raise RuntimeError("Postprocessor returned no anchor boxes.")

        print(
            "[V2XSimE3DSNN] split={} samples={} "
            "max_cav={} comm_range={} proj_first={}".format(
                self.split,
                len(self.base_dataset),
                self.max_cav,
                self.comm_range,
                self.proj_first
            )
        )

    def __len__(self):
        return len(self.base_dataset)

    def _filter_agents(self, sample):
        """Keep Agent 1 and neighbors within the configured XY range."""
        agents = sample["agents"]

        if 1 not in agents:
            raise RuntimeError(
                "Sample {} does not contain Agent 1."
                .format(sample["sample_token"])
            )

        ego_pose = agents[1]["lidar_pose"]
        ego_xy = ego_pose[:2, 3]

        active = OrderedDict()

        for agent_id, agent in agents.items():
            pose = agent["lidar_pose"]
            distance = np.linalg.norm(
                pose[:2, 3] - ego_xy
            )

            if agent_id == 1 or distance <= self.comm_range:
                active[agent_id] = agent

        if not active or next(iter(active.keys())) != 1:
            raise RuntimeError(
                "Agent 1 must remain the first active agent."
            )

        if len(active) > self.max_cav:
            raise RuntimeError(
                "Active Agent count {} exceeds max_cav={}"
                .format(len(active), self.max_cav)
            )

        return active

    def get_pairwise_transformation(self, active_agents):
        """Return T_agent_i_to_agent_j with shape (max_cav,max_cav,4,4)."""
        matrices = np.zeros(
            (self.max_cav, self.max_cav, 4, 4),
            dtype=np.float32
        )

        agent_records = list(active_agents.values())
        n = len(agent_records)

        for i in range(n):
            T_world_i = agent_records[i]["lidar_pose"]

            for j in range(n):
                if i == j:
                    matrices[i, j] = np.eye(
                        4, dtype=np.float32
                    )
                    continue

                T_world_j = agent_records[j]["lidar_pose"]

                # T_j_i = inv(T_world_j) @ T_world_i
                matrices[i, j] = np.linalg.solve(
                    T_world_j, T_world_i
                ).astype(np.float32)

        return matrices

    def _empty_voxel_features(self):
        """Return a valid empty sparse-voxel dictionary."""
        channels = int(
            getattr(self.pre_processor, "num_point_features", 4)
        )

        if getattr(self.pre_processor, "encoding", "mean") == "svc":
            channels *= int(
                getattr(self.pre_processor, "num_time_bins", 1)
            )

        return {
            "voxel_features": np.zeros(
                (0, channels), dtype=np.float32
            ),
            "voxel_coords": np.zeros(
                (0, 3), dtype=np.int32
            ),
            "voxel_num_points": np.zeros(
                (0,), dtype=np.int32
            ),
        }

    def _process_agent_lidar(self, agent):
        """Local LiDAR filtering and E3DSNN voxel preprocessing."""
        lidar_np = self.base_dataset.load_lidar(agent)

        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np = mask_points_by_range(
            lidar_np, self.lidar_range
        )

        if lidar_np.shape[0] == 0:
            processed = self._empty_voxel_features()
        else:
            processed = self.pre_processor.preprocess(lidar_np)

        return processed, lidar_np

    def _collect_global_gt(self, active_agents, ego_pose):
        """Project all selected agents' global GT to the fixed Ego frame.

        This uses the repository's project_world_objects_v2x utility.
        Its lidar_np argument is retained for API compatibility; the
        current utility implementation performs range filtering but
        does not perform point-in-box hit filtering.
        """
        ego_pose6 = tfm_to_pose(ego_pose)

        unique_boxes = OrderedDict()

        empty_lidar = np.zeros(
            (0, 4), dtype=np.float32
        )

        for agent_id, agent in active_agents.items():
            gt_boxes = agent["gt_boxes_global"]
            object_ids = agent["gt_object_ids"]

            if len(gt_boxes) != len(object_ids):
                raise ValueError(
                    "Agent {} has GT/ID count mismatch."
                    .format(agent_id)
                )

            object_dict = {
                "gt_boxes": gt_boxes,
                "object_ids": object_ids,
            }

            projected = OrderedDict()

            box_utils.project_world_objects_v2x(
                object_dict=object_dict,
                output_dict=projected,
                reference_lidar_pose=ego_pose6,
                lidar_range=self.gt_filter_range,
                order=self.box_order,
                lidar_np=empty_lidar,
            )

            for object_id, box in projected.items():
                if hasattr(object_id, "item"):
                    object_id = object_id.item()

                box = np.asarray(
                    box, dtype=np.float32
                ).reshape(-1, 7)

                if box.shape[0] != 1:
                    raise ValueError(
                        "Expected one projected box for object {}, "
                        "got shape {}.".format(object_id, box.shape)
                    )

                row = box[0]

                if not np.isfinite(row).all():
                    raise ValueError(
                        "Non-finite projected GT for object {}."
                        .format(object_id)
                    )

                if np.any(row[3:6] <= 0):
                    raise ValueError(
                        "Non-positive projected GT dimensions for object {}."
                        .format(object_id)
                    )

                # Stable first-seen de-duplication:
                # Agent 1 first, then neighbors in original ID order.
                if object_id not in unique_boxes:
                    unique_boxes[object_id] = row.copy()

        if len(unique_boxes) > self.max_num:
            raise RuntimeError(
                "Unique cooperative GT count {} exceeds postprocess.max_num={}. "
                "Increase max_num rather than silently truncating labels."
                .format(len(unique_boxes), self.max_num)
            )

        object_bbx_center = np.zeros(
            (self.max_num, 7), dtype=np.float32
        )
        object_bbx_mask = np.zeros(
            (self.max_num,), dtype=np.float32
        )

        object_ids = list(unique_boxes.keys())

        for index, box in enumerate(unique_boxes.values()):
            object_bbx_center[index] = box
            object_bbx_mask[index] = 1.0

        return object_bbx_center, object_bbx_mask, object_ids

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """Merge per-agent or per-scene sparse feature dictionaries."""
        merged = OrderedDict()

        for feature_dict in processed_feature_list:
            for name, feature in feature_dict.items():
                if name not in merged:
                    merged[name] = []

                if isinstance(feature, list):
                    merged[name].extend(feature)
                else:
                    merged[name].append(feature)

        return merged

    def __getitem__(self, idx):
        sample = self.base_dataset.get_record_metadata(idx)

        active_agents = self._filter_agents(sample)

        agent_ids = list(active_agents.keys())
        cav_num = len(agent_ids)

        ego_pose = active_agents[1]["lidar_pose"]

        pairwise_t_matrix = self.get_pairwise_transformation(
            active_agents
        )

        processed_features = []
        origin_lidar_list = []

        lidar_poses = []
        ego_flags = []
        transformation_matrices = []

        for agent_id, agent in active_agents.items():
            processed, lidar_np = self._process_agent_lidar(agent)

            processed_features.append(processed)

            if self.visualize:
                origin_lidar_list.append(lidar_np)

            lidar_pose = agent["lidar_pose"]

            lidar_poses.append(
                tfm_to_pose(lidar_pose)
            )

            ego_flags.append(agent_id == 1)

            transformation_matrices.append(
                np.linalg.solve(
                    ego_pose, lidar_pose
                ).astype(np.float32)
            )

        object_bbx_center, object_bbx_mask, object_ids = (
            self._collect_global_gt(active_agents, ego_pose)
        )

        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center,
            anchors=self.anchor_box,
            mask=object_bbx_mask,
        )

        merged_feature_dict = self.merge_features_to_dict(
            processed_features
        )

        # V2X-Sim is vehicle-to-vehicle in this PKL protocol.
        # Prior encodings are zero-filled for interface compatibility.
        velocity = np.zeros(
            (self.max_cav,), dtype=np.float32
        )
        time_delay = np.zeros(
            (self.max_cav,), dtype=np.float32
        )
        infra = np.zeros(
            (self.max_cav,), dtype=np.float32
        )

        spatial_correction_matrix = np.tile(
            np.eye(4, dtype=np.float32)[None, :, :],
            (self.max_cav, 1, 1)
        )

        output_ego = {
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": object_bbx_mask,
            "object_ids": object_ids,
            "anchor_box": self.anchor_box,
            "processed_lidar": merged_feature_dict,
            "label_dict": label_dict,
            "cav_num": cav_num,
            "record_len": cav_num,
            "agent_ids": agent_ids,
            "sample_token": sample["sample_token"],
            "sample_index": int(idx),
            "velocity": velocity,
            "time_delay": time_delay,
            "infra": infra,
            "spatial_correction_matrix": spatial_correction_matrix,
            "pairwise_t_matrix": pairwise_t_matrix,
            "lidar_pose": lidar_poses,
            "ego_flag": ego_flags,
            "transformation_matrix": transformation_matrices,
        }

        if self.visualize:
            if origin_lidar_list:
                output_ego["origin_lidar"] = np.vstack(
                    origin_lidar_list
                )
            else:
                output_ego["origin_lidar"] = np.zeros(
                    (0, 4), dtype=np.float32
                )

        return {"ego": output_ego}

    def collate_batch_train(self, batch):
        output_dict = {"ego": {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []

        processed_lidar_list = []
        record_len = []
        label_dict_list = []
        pairwise_t_matrix_list = []
        spatial_correction_matrix_list = []

        velocity = []
        time_delay = []
        infra = []

        lidar_poses = []
        ego_flags = []
        agent_ids = []
        sample_tokens = []
        sample_indices = []

        if self.visualize:
            origin_lidar = []

        for sample in batch:
            ego_dict = sample["ego"]

            object_bbx_center.append(
                ego_dict["object_bbx_center"]
            )
            object_bbx_mask.append(
                ego_dict["object_bbx_mask"]
            )
            object_ids.append(
                ego_dict["object_ids"]
            )

            processed_lidar_list.append(
                ego_dict["processed_lidar"]
            )

            record_len.append(
                ego_dict["cav_num"]
            )

            label_dict_list.append(
                ego_dict["label_dict"]
            )

            pairwise_t_matrix_list.append(
                ego_dict["pairwise_t_matrix"]
            )

            spatial_correction_matrix_list.append(
                ego_dict["spatial_correction_matrix"]
            )

            velocity.append(ego_dict["velocity"])
            time_delay.append(ego_dict["time_delay"])
            infra.append(ego_dict["infra"])

            lidar_poses.append(ego_dict["lidar_pose"])
            ego_flags.append(ego_dict["ego_flag"])
            agent_ids.append(ego_dict["agent_ids"])

            sample_tokens.append(ego_dict["sample_token"])
            sample_indices.append(ego_dict["sample_index"])

            if self.visualize:
                origin_lidar.append(
                    ego_dict["origin_lidar"]
                )

        object_bbx_center = torch.from_numpy(
            np.asarray(object_bbx_center, dtype=np.float32)
        )

        object_bbx_mask = torch.from_numpy(
            np.asarray(object_bbx_mask, dtype=np.float32)
        )

        merged_feature_dict = self.merge_features_to_dict(
            processed_lidar_list
        )

        processed_lidar_torch_dict = (
            self.pre_processor.collate_batch(
                merged_feature_dict
            )
        )

        record_len = torch.as_tensor(
            record_len, dtype=torch.long
        )

        label_torch_dict = self.post_processor.collate_batch(
            label_dict_list
        )

        velocity = torch.from_numpy(
            np.asarray(velocity, dtype=np.float32)
        )
        time_delay = torch.from_numpy(
            np.asarray(time_delay, dtype=np.float32)
        )
        infra = torch.from_numpy(
            np.asarray(infra, dtype=np.float32)
        )

        prior_encoding = torch.stack(
            [velocity, time_delay, infra], dim=-1
        ).float()

        pairwise_t_matrix = torch.from_numpy(
            np.asarray(
                pairwise_t_matrix_list,
                dtype=np.float32
            )
        )

        spatial_correction_matrix = torch.from_numpy(
            np.asarray(
                spatial_correction_matrix_list,
                dtype=np.float32
            )
        )

        output_dict["ego"].update({
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": object_bbx_mask,
            "processed_lidar": processed_lidar_torch_dict,
            "record_len": record_len,
            "label_dict": label_torch_dict,

            # Legacy inference interface: first sample's IDs.
            "object_ids": object_ids[0],
            "object_ids_batch": object_ids,

            "prior_encoding": prior_encoding,
            "spatial_correction_matrix": spatial_correction_matrix,
            "pairwise_t_matrix": pairwise_t_matrix,

            "lidar_pose": lidar_poses,
            "ego_flag": ego_flags,
            "agent_ids": agent_ids,
            "sample_token": sample_tokens,
            "sample_index": sample_indices,
        })

        if self.visualize:
            origin_lidar = np.asarray(
                downsample_lidar_minimum(
                    pcd_np_list=origin_lidar
                )
            )
            output_dict["ego"]["origin_lidar"] = (
                torch.from_numpy(origin_lidar)
            )

        return output_dict

    def collate_batch_test(self, batch):
        if len(batch) != 1:
            raise AssertionError(
                "V2X-Sim test batch size must be 1."
            )

        output_dict = self.collate_batch_train(batch)

        anchor_box = batch[0]["ego"]["anchor_box"]

        output_dict["ego"]["anchor_box"] = torch.from_numpy(
            np.asarray(anchor_box, dtype=np.float32)
        )

        output_dict["ego"]["transformation_matrix"] = (
            torch.from_numpy(np.eye(4, dtype=np.float32))
        )

        return output_dict

    def post_process(self, data_dict, output_dict):
        pred_box_tensor, pred_score = (
            self.post_processor.post_process(
                data_dict, output_dict
            )
        )

        gt_box_tensor = self.post_processor.generate_gt_bbx(
            data_dict
        )

        return pred_box_tensor, pred_score, gt_box_tensor
