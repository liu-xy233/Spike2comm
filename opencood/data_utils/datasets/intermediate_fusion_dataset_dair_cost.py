import copy

import numpy as np
import torch

from opencood.data_utils.datasets.intermediate_fusion_dataset_dair import (
    IntermediateFusionDatasetDAIR,
)
from opencood.utils.transformation_utils import pose_to_tfm


class IntermediateFusionDatasetDAIRCoST(IntermediateFusionDatasetDAIR):
    """DAIR intermediate-fusion dataset wrapper with optional CoST history frames."""

    def __init__(self, params, visualize, train=True):
        super().__init__(params, visualize, train)
        cost_temporal = params.get("cost_temporal", {})
        self.prev_frames = int(cost_temporal.get("prev_frames", 1))
        self.interval = int(cost_temporal.get("interval", 1))

    def __getitem__(self, idx):
        current = super().__getitem__(idx)
        if self.prev_frames <= 0:
            current["ego"]["history"] = []
            current["ego"]["prev_exist"] = False
            return current

        history = []
        for frame_offset in range(self.prev_frames, 0, -1):
            prev_idx = idx - frame_offset * self.interval
            if prev_idx < 0:
                continue
            prev = super().__getitem__(prev_idx)
            history.append(prev["ego"])

        current["ego"]["history"] = history
        current["ego"]["prev_exist"] = len(history) > 0
        return current

    def collate_batch_train(self, batch):
        if batch is None:
            return None
        base_batch, histories, prev_exist = self._split_cost_history(batch)
        output_dict = super().collate_batch_train(base_batch)
        if output_dict is None:
            return None
        self._attach_cost_meta(output_dict["ego"])

        max_history_len = max((len(history) for history in histories), default=0)
        collated_history = []
        for history_idx in range(max_history_len):
            history_batch = []
            for sample_idx, history in enumerate(histories):
                if len(history) == max_history_len:
                    history_ego = history[history_idx]
                elif len(history) > history_idx:
                    history_ego = history[history_idx]
                else:
                    history_ego = base_batch[sample_idx]["ego"]
                history_batch.append({"ego": copy.copy(history_ego)})

            history_output = super().collate_batch_train(history_batch)
            if history_output is None:
                return None
            self._attach_cost_meta(history_output["ego"])
            collated_history.append(history_output["ego"])

        output_dict["ego"]["history"] = collated_history
        output_dict["ego"]["prev_exist"] = torch.from_numpy(np.array(prev_exist, dtype=bool))
        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None

        if batch[0]["ego"]["anchor_box"] is not None:
            output_dict["ego"].update({"anchor_box": torch.from_numpy(np.array(batch[0]["ego"]["anchor_box"]))})

        transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
        transformation_matrix_clean_torch = torch.from_numpy(np.identity(4)).float()
        output_dict["ego"].update(
            {
                "transformation_matrix": transformation_matrix_torch,
                "transformation_matrix_clean": transformation_matrix_clean_torch,
                "sample_idx": batch[0]["ego"]["sample_idx"],
                "cav_id_list": batch[0]["ego"]["cav_id_list"],
            }
        )
        return output_dict

    @staticmethod
    def _split_cost_history(batch):
        base_batch = []
        histories = []
        prev_exist = []
        for item in batch:
            ego_dict = item["ego"]
            histories.append(ego_dict.get("history", []))
            prev_exist.append(bool(ego_dict.get("prev_exist", False)))
            base_ego = {
                key: value
                for key, value in ego_dict.items()
                if key not in ["history", "prev_exist"]
            }
            base_batch.append({"ego": base_ego})
        return base_batch, histories, prev_exist

    def _attach_cost_meta(self, ego_dict):
        record_len = ego_dict["record_len"]
        lidar_pose = ego_dict["lidar_pose"].float()
        split_pose = self._regroup_pose(lidar_pose, record_len)
        ego_pose = torch.stack([pose_to_tfm(pose[:1]).squeeze(0) for pose in split_pose], dim=0)

        batch_size = len(record_len)
        prior_encoding = torch.zeros(batch_size, self.max_cav, 3, dtype=torch.float32)
        for batch_idx, cav_num in enumerate(record_len):
            if int(cav_num) > 1:
                prior_encoding[batch_idx, 1 : int(cav_num), 2] = 1

        ego_dict["ego_pose"] = ego_pose
        ego_dict["prior_encoding"] = prior_encoding

    @staticmethod
    def _regroup_pose(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        return torch.tensor_split(x, cum_sum_len[:-1].cpu())

