"""Top-K hard ranking loss restricted to valid ego-aligned patches."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKHardValuePredictionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_rank = float(config.get("lambda_rank", 1.0))
        self.margin = float(config.get("margin", 0.02))
        self.topk = int(config.get("topk", 10))
        self.hard_negative_k = int(config.get("hard_negative_k", 32))

        if self.lambda_rank < 0 or self.margin < 0:
            raise ValueError("lambda_rank and margin must be non-negative.")
        if self.topk <= 0 or self.hard_negative_k <= 0:
            raise ValueError("topk and hard_negative_k must be positive.")

    def ranking_loss(self, pred_value, utility, valid_mask):
        batch_losses = []

        for b in range(utility.shape[0]):
            valid_indices = torch.where(valid_mask[b])[0]
            n_valid = valid_indices.numel()

            if n_valid < 2:
                continue

            # Positive patches: highest ground-truth utility among valid patches.
            positive_count = min(self.topk, n_valid - 1)
            positive_local = torch.topk(
                utility[b, valid_indices], k=positive_count
            ).indices
            positive_indices = valid_indices[positive_local]

            # Hard negatives are also restricted to valid patches.
            candidate_mask = valid_mask[b].clone()
            candidate_mask[positive_indices] = False
            negative_candidates = torch.where(candidate_mask)[0]

            if negative_candidates.numel() == 0:
                continue

            negative_count = min(
                self.hard_negative_k, negative_candidates.numel()
            )
            negative_local = torch.topk(
                pred_value[b, negative_candidates].detach(),
                k=negative_count,
            ).indices
            negative_indices = negative_candidates[negative_local]

            utility_pos = utility[b, positive_indices].unsqueeze(1)
            utility_neg = utility[b, negative_indices].unsqueeze(0)
            valid_pairs = utility_pos > utility_neg

            if not bool(valid_pairs.any()):
                continue

            pred_pos = pred_value[b, positive_indices].unsqueeze(1)
            pred_neg = pred_value[b, negative_indices].unsqueeze(0)
            pair_losses = F.relu(self.margin - (pred_pos - pred_neg))
            batch_losses.append(pair_losses[valid_pairs])

        if not batch_losses:
            return pred_value.sum() * 0.0

        return torch.cat(batch_losses).mean()

    def forward(self, pred_value, utility, valid_mask=None):
        assert pred_value.dim() == 2 and utility.dim() == 2
        assert pred_value.shape == utility.shape
        assert torch.isfinite(pred_value).all()
        assert torch.isfinite(utility).all()

        if valid_mask is None:
            valid_mask = torch.ones_like(utility, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(
                device=utility.device, dtype=torch.bool
            )

        assert valid_mask.shape == utility.shape
        if not bool(valid_mask.any()):
            raise ValueError("Batch contains no valid patches.")

        # Regression ignores invalid patches.
        regression = F.smooth_l1_loss(
            pred_value[valid_mask], utility[valid_mask]
        )

        # Ranking ignores invalid positives and invalid hard negatives.
        ranking = self.ranking_loss(pred_value, utility, valid_mask)
        total = regression + self.lambda_rank * ranking

        return {
            "total_loss": total,
            "reg_loss": regression,
            "rank_loss": ranking,
        }
