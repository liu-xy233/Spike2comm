"""Counterfactual utility 的回归与排序联合损失。"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValuePredictionLoss(nn.Module):
    """SmoothL1 回归损失加 pairwise margin ranking 损失。"""

    def __init__(self, config: Dict):
        super(ValuePredictionLoss, self).__init__()
        self.lambda_rank = float(config.get("lambda_rank", 1.0))
        self.margin = float(config.get("margin", 0.1))
        self.num_pairs = int(config.get("num_pairs", 256))
        self.pair_sampling = str(config.get("pair_sampling", "random"))
        self.topk = int(config.get("topk", 32))
        if self.lambda_rank < 0 or self.margin < 0:
            raise ValueError("lambda_rank and margin must be non-negative.")
        if self.num_pairs <= 0 or self.topk <= 0:
            raise ValueError("num_pairs and topk must be positive.")
        if self.pair_sampling not in ("random", "topk"):
            raise ValueError("pair_sampling must be 'random' or 'topk'.")
        self.regression_loss = nn.SmoothL1Loss()

    def _random_pairs(self, utility: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor]:
        batch_size, num_patch = utility.shape
        if num_patch < 2:
            raise ValueError("Ranking loss requires at least two patches.")
        first = torch.randint(
            num_patch,
            (batch_size, self.num_pairs),
            device=utility.device,
        )
        offset = torch.randint(
            1,
            num_patch,
            (batch_size, self.num_pairs),
            device=utility.device,
        )
        second = (first + offset) % num_patch
        return first, second

    def _topk_pairs(self, utility: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor]:
        _, num_patch = utility.shape
        k = min(self.topk, num_patch // 2)
        high = torch.topk(utility, k=k, dim=1, largest=True).indices
        low = torch.topk(utility, k=k, dim=1, largest=False).indices
        # [B,k,k] 展开为高 utility patch 与低 utility patch 的候选 pair。
        first = high.unsqueeze(2).expand(-1, -1, k).reshape(
            utility.shape[0], -1
        )
        second = low.unsqueeze(1).expand(-1, k, -1).reshape(
            utility.shape[0], -1
        )
        return first[:, :self.num_pairs], second[:, :self.num_pairs]

    def ranking_loss(self, pred_value: torch.Tensor,
                     utility: torch.Tensor) -> torch.Tensor:
        if self.pair_sampling == "topk":
            first, second = self._topk_pairs(utility)
        else:
            first, second = self._random_pairs(utility)

        pred_first = pred_value.gather(1, first)
        pred_second = pred_value.gather(1, second)
        utility_first = utility.gather(1, first)
        utility_second = utility.gather(1, second)

        # 始终将 utility 较大的 patch 放在 pair 前面。
        first_is_higher = utility_first > utility_second
        prediction_difference = torch.where(
            first_is_higher,
            pred_first - pred_second,
            pred_second - pred_first,
        )
        valid = utility_first != utility_second
        if not bool(valid.any()):
            return pred_value.sum() * 0.0
        return F.relu(self.margin - prediction_difference[valid]).mean()

    def forward(self, pred_value: torch.Tensor,
                utility: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert pred_value.dim() == 2, "pred_value must be [B,P]."
        assert utility.dim() == 2, "utility must be [B,P]."
        assert pred_value.shape == utility.shape, (
            "pred_value and utility shapes differ: {} vs {}.".format(
                tuple(pred_value.shape), tuple(utility.shape)
            )
        )
        assert torch.isfinite(pred_value).all(), "pred_value contains NaN/Inf."
        assert torch.isfinite(utility).all(), "utility contains NaN/Inf."

        regression = self.regression_loss(pred_value, utility)
        ranking = self.ranking_loss(pred_value, utility)
        total = regression + self.lambda_rank * ranking
        return {
            "total_loss": total,
            "reg_loss": regression,
            "rank_loss": ranking,
        }
