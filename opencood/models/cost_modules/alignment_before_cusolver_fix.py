import torch

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


def get_pairwise_transformation_torch(lidar_poses_list, max_cav=3):
    """Build pairwise transforms following CoST's temporal transform convention."""
    batch_size = lidar_poses_list.shape[0]
    pairwise_t_matrix = torch.eye(4, device=lidar_poses_list.device).view(
        1, 1, 1, 4, 4
    ).repeat(batch_size, max_cav, max_cav, 1, 1)

    for batch_idx in range(batch_size):
        t_list = lidar_poses_list[batch_idx]
        for i in range(len(t_list)):
            for j in range(len(t_list)):
                if i != j:
                    pairwise_t_matrix[batch_idx][i, j] = torch.linalg.solve(
                        t_list[j], t_list[i]
                    )
    return pairwise_t_matrix


def transforms_to_bev_affine(pairwise_t_matrix, height, width, align_ratio):
    pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]].clone()
    pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * height / width
    pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * width / height
    pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (align_ratio * width) * 2
    pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (align_ratio * height) * 2
    return pairwise_t_matrix


def align_features(features, ego_poses, align_ratio=1.6, target_pos=-1):
    """CoST temporal feature alignment.

    Args:
        features: [B, T, C, H, W]
        ego_poses: [B, T, 4, 4], each pose is T_world_ego
        align_ratio: voxel size times BEV feature stride
        target_pos: -1 aligns all frames to current frame, matching CoST default
    """
    if target_pos not in [-1, 0]:
        raise ValueError("target_pos must be -1 or 0.")

    batch_size, time_len, _, height, width = features.shape
    pairwise_t_matrix = get_pairwise_transformation_torch(ego_poses, max_cav=time_len)
    pairwise_t_matrix = transforms_to_bev_affine(
        pairwise_t_matrix, height, width, align_ratio
    )

    aligned_features = []
    for batch_idx in range(batch_size):
        t_matrix = pairwise_t_matrix[batch_idx][:time_len, :time_len, :, :]
        aligned = warp_affine_simple(
            features[batch_idx], t_matrix[target_pos, :, :, :], (height, width)
        )
        aligned_features.append(aligned)
    return torch.stack(aligned_features)


def align_agents_to_ego(features, pairwise_t_matrix, record_len, align_ratio):
    """Align current-frame agent features to the ego BEV frame."""
    batch_size, max_cav, _, height, width = features.shape
    pairwise_t_matrix = transforms_to_bev_affine(
        pairwise_t_matrix, height, width, align_ratio
    )

    aligned_features = features.clone()
    for batch_idx in range(batch_size):
        cav_num = int(record_len[batch_idx])
        if cav_num <= 0:
            continue
        t_matrix = pairwise_t_matrix[batch_idx][:cav_num, :cav_num, :, :]
        aligned_features[batch_idx, :cav_num] = warp_affine_simple(
            features[batch_idx, :cav_num], t_matrix[0, :, :, :], (height, width)
        )
    return aligned_features

