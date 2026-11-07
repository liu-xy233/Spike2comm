"""Top-K-aware hard-negative ranking loss."""

import torch
import torch.nn.functional as F

from opencood.models.comm_modules.value_prediction_loss import (
    ValuePredictionLoss
)


class TopKHardValuePredictionLoss(ValuePredictionLoss):

    def __init__(self, config):
        # 原类只支持random/topk，使用兼容配置初始化，
        # 然后覆盖ranking_loss。
        base_config = dict(config)
        base_config["pair_sampling"] = "random"
        super(TopKHardValuePredictionLoss, self).__init__(
            base_config
        )

        self.topk = int(config.get("topk", 10))
        self.hard_negative_k = int(
            config.get("hard_negative_k", 32)
        )

        if self.hard_negative_k <= 0:
            raise ValueError(
                "hard_negative_k must be positive"
            )

    def ranking_loss(self, pred_value, utility):
        batch_size, num_patch = utility.shape

        if num_patch < 2:
            raise ValueError(
                "Ranking requires at least 2 patches"
            )

        k = min(self.topk, num_patch - 1)
        neg_k = min(
            self.hard_negative_k,
            num_patch - k
        )

        # 真实Top-K正样本
        positive_idx = torch.topk(
            utility,
            k=k,
            dim=1,
            largest=True
        ).indices

        # 困难负样本：预测值高，但不属于真实Top-K
        with torch.no_grad():
            negative_scores = pred_value.detach().clone()
            negative_scores.scatter_(
                1,
                positive_idx,
                float("-inf")
            )

            negative_idx = torch.topk(
                negative_scores,
                k=neg_k,
                dim=1,
                largest=True
            ).indices

        # 真实Utility比较
        utility_pos = utility.gather(
            1, positive_idx
        ).unsqueeze(2)

        utility_neg = utility.gather(
            1, negative_idx
        ).unsqueeze(1)

        valid = utility_pos > utility_neg

        if not bool(valid.any()):
            return pred_value.sum() * 0.0

        # 预测值比较
        pred_pos = pred_value.gather(
            1, positive_idx
        ).unsqueeze(2)

        pred_neg = pred_value.gather(
            1, negative_idx
        ).unsqueeze(1)

        pair_losses = F.relu(
            self.margin - (pred_pos - pred_neg)
        )

        return pair_losses.expand_as(valid)[valid].mean()
