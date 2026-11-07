"""原始 ValuePredictionLoss 的有效区域掩码版本。

保留原始 SmoothL1 回归、margin、lambda_rank、num_pairs
及 pair_sampling 配置，只排除无效 Patch 的直接监督。
"""

import torch
import torch.nn.functional as F

from opencood.models.comm_modules.value_prediction_loss import (
    ValuePredictionLoss,
)


class MaskedValuePredictionLoss(ValuePredictionLoss):

    def ranking_loss(self, pred_value, utility, valid_mask):
        batch_size = utility.shape[0]
        all_losses = []

        for b in range(batch_size):
            valid_indices = torch.where(valid_mask[b])[0]
            n_valid = valid_indices.numel()

            if n_valid < 2:
                continue

            if self.pair_sampling == "topk":
                k = min(self.topk, n_valid // 2)
                values = utility[b, valid_indices]

                high = torch.topk(
                    values, k=k, largest=True
                ).indices
                low = torch.topk(
                    values, k=k, largest=False
                ).indices

                first = valid_indices[high].unsqueeze(1).expand(
                    -1, k
                ).reshape(-1)
                second = valid_indices[low].unsqueeze(0).expand(
                    k, -1
                ).reshape(-1)

                first = first[:self.num_pairs]
                second = second[:self.num_pairs]

            else:
                # 与原始 random 配对方式一致，
                # 但仅从有效 Patch 中采样。
                first_local = torch.randint(
                    n_valid,
                    (self.num_pairs,),
                    device=utility.device,
                )
                offset = torch.randint(
                    1,
                    n_valid,
                    (self.num_pairs,),
                    device=utility.device,
                )
                second_local = (first_local + offset) % n_valid

                first = valid_indices[first_local]
                second = valid_indices[second_local]

            utility_first = utility[b, first]
            utility_second = utility[b, second]

            prediction_first = pred_value[b, first]
            prediction_second = pred_value[b, second]

            unequal = utility_first != utility_second
            if not bool(unequal.any()):
                continue

            prediction_difference = torch.where(
                utility_first > utility_second,
                prediction_first - prediction_second,
                prediction_second - prediction_first,
            )

            pair_losses = F.relu(
                self.margin - prediction_difference[unequal]
            )
            all_losses.append(pair_losses)

        if not all_losses:
            return pred_value.sum() * 0.0

        return torch.cat(all_losses).mean()

    def forward(self, pred_value, utility, valid_mask=None):
        assert pred_value.dim() == 2
        assert utility.dim() == 2
        assert pred_value.shape == utility.shape

        if valid_mask is None:
            valid_mask = torch.ones_like(
                utility, dtype=torch.bool
            )
        else:
            valid_mask = valid_mask.to(
                device=utility.device, dtype=torch.bool
            )

        assert valid_mask.shape == utility.shape
        assert torch.isfinite(pred_value).all()

        if not bool(valid_mask.any()):
            zero = pred_value.sum() * 0.0
            return {
                "total_loss": zero,
                "reg_loss": zero,
                "rank_loss": zero,
            }

        assert torch.isfinite(utility[valid_mask]).all()

        # 仅在有效 Patch 上计算回归损失。
        regression = F.smooth_l1_loss(
            pred_value[valid_mask],
            utility[valid_mask],
        )

        # 仅使用有效 Patch 构造排序对。
        ranking = self.ranking_loss(
            pred_value, utility, valid_mask
        )

        total = regression + self.lambda_rank * ranking

        return {
            "total_loss": total,
            "reg_loss": regression,
            "rank_loss": ranking,
        }
