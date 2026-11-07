# -*- coding: utf-8 -*-
"""Shared supervision-mode handling for single-GPU and DDP training.

Modes
-----
legacy_dair:
    Preserve the historical DAIR objective: fused detection loss, whichever
    legacy vehicle/infrastructure auxiliary branches have matching labels and
    predictions, and optional intermediate communication-round losses.
cooperative_only:
    Use only the final fused cooperative detection loss.
all_agents_aux:
    Use the fused loss plus an auxiliary loss over packed per-agent predictions
    and per-agent labels. The dataset must provide label_dict_all_agents.
"""

from __future__ import print_function


_VALID_MODES = (
    "legacy_dair",
    "cooperative_only",
    "all_agents_aux",
)


def get_loss_mode(hypes):
    """Resolve the configured supervision mode.

    New datasets should set loss.supervision_mode explicitly. The default is
    legacy_dair so existing DAIR configurations retain their previous
    supervision behavior without requiring YAML edits.
    """
    loss_cfg = hypes.get("loss", {})
    train_cfg = hypes.get("train_params", {})

    loss_mode = loss_cfg.get("supervision_mode")
    train_mode = train_cfg.get("loss_mode")

    if loss_mode is not None and train_mode is not None:
        if loss_mode != train_mode:
            raise ValueError(
                "Conflicting loss modes: loss.supervision_mode={!r}, "
                "train_params.loss_mode={!r}".format(loss_mode, train_mode)
            )

    mode = loss_mode if loss_mode is not None else train_mode
    if mode is None:
        mode = "legacy_dair"

    if mode not in _VALID_MODES:
        raise ValueError(
            "Unsupported supervision mode {!r}. Expected one of {}."
            .format(mode, _VALID_MODES)
        )
    return mode


def _requires_iou(criterion):
    return bool(getattr(criterion, "iou_quality_enabled", False))


def _check_prediction_branch(output_dict, prefix, require_iou):
    required = (
        "psm{}".format(prefix),
        "rm{}".format(prefix),
    )
    if require_iou:
        required += ("iou{}".format(prefix),)
    return all(key in output_dict for key in required)


def _add_legacy_dair_auxiliary_losses(
        total_loss, output_dict, target_dict, criterion):
    """Add legacy V/I losses only when both matching sides are available."""
    require_iou = _requires_iou(criterion)

    branches = (
        ("_single_v", "label_dict_single_v"),
        ("_single_i", "label_dict_single_i"),
    )

    for prefix, label_key in branches:
        has_target = label_key in target_dict
        has_prediction = _check_prediction_branch(
            output_dict, prefix, require_iou
        )

        if has_target and has_prediction:
            total_loss = total_loss + criterion(
                output_dict, target_dict[label_key], prefix=prefix
            )
        elif has_target != has_prediction:
            missing = (
                "prediction branch {}".format(prefix)
                if has_target else "target {}".format(label_key)
            )
            raise KeyError(
                "Incomplete legacy DAIR auxiliary supervision: missing {}."
                .format(missing)
            )

    return total_loss


def _add_round_losses(total_loss, output_dict, fused_target, hypes,
                      criterion):
    """Add optional intermediate-round fused detection losses."""
    model_args = hypes.get("model", {}).get("args", {})
    fusion_args = model_args.get("fusion_args", {})
    communication = fusion_args.get("communication", {})
    rounds = int(communication.get("round", 1))

    require_iou = _requires_iou(criterion)
    for round_id in range(1, rounds):
        prefix = "_v{}".format(round_id)
        if not _check_prediction_branch(output_dict, prefix, require_iou):
            raise KeyError(
                "communication round {} is configured but output keys for "
                "prefix {!r} are missing.".format(round_id, prefix)
            )
        total_loss = total_loss + criterion(
            output_dict, fused_target, prefix=prefix
        )
    return total_loss


def _add_all_agents_auxiliary_loss(total_loss, output_dict, target_dict,
                                   criterion, hypes):
    """Add local losses for every packed agent (V2 supervision mode)."""
    label_key = "label_dict_all_agents"
    if label_key not in target_dict:
        raise KeyError(
            "supervision_mode='all_agents_aux' requires "
            "batch_data['ego']['{}']; the Dataset must collate one local "
            "GT label per packed Agent in the same order as record_len."
            .format(label_key)
        )

    required = ("psm_single", "rm_single")
    missing = [key for key in required if key not in output_dict]
    if _requires_iou(criterion) and "iou_single" not in output_dict:
        missing.append("iou_single")
    if missing:
        raise KeyError(
            "all_agents_aux requires packed local prediction keys: {}."
            .format(", ".join(missing))
        )

    local_target = target_dict[label_key]
    local_output = {
        "psm": output_dict["psm_single"],
        "rm": output_dict["rm_single"],
    }
    if "iou_single" in output_dict:
        local_output["iou"] = output_dict["iou_single"]

    local_count = int(output_dict["psm_single"].shape[0])
    target_count = int(local_target["pos_equal_one"].shape[0])
    if local_count != target_count:
        raise ValueError(
            "Packed Agent prediction count ({}) does not match local GT "
            "count ({}). Check record_len ordering and Dataset collate."
            .format(local_count, target_count)
        )

    if "record_len" in target_dict:
        record_len = target_dict["record_len"]
        if hasattr(record_len, "sum"):
            expected_count = int(record_len.sum().item())
            if expected_count != local_count:
                raise ValueError(
                    "sum(record_len)={} but packed local predictions contain "
                    "{} Agents.".format(expected_count, local_count)
                )

    weight = float(
        hypes.get("loss", {}).get("agent_aux_weight", 1.0)
    )
    local_loss = criterion(local_output, local_target)
    return total_loss + weight * local_loss


def compute_training_loss(output_dict, batch_data, hypes, criterion,
                          loss_mode=None):
    """Compute the configured objective identically in train.py and DDP."""
    mode = loss_mode or get_loss_mode(hypes)
    target_dict = batch_data["ego"]
    fused_target = target_dict["label_dict"]

    total_loss = criterion(output_dict, fused_target)

    if mode == "cooperative_only":
        # V1: only the final fused cooperative detection objective.
        # Do not add local-agent or intermediate-round auxiliary losses.
        return total_loss

    if mode == "legacy_dair":
        total_loss = _add_legacy_dair_auxiliary_losses(
            total_loss, output_dict, target_dict, criterion
        )
    elif mode == "all_agents_aux":
        # V2: packed local predictions supervised by each Agent's own GT.
        total_loss = _add_all_agents_auxiliary_loss(
            total_loss, output_dict, target_dict, criterion, hypes
        )

    loss_cfg = hypes.get("loss", {})
    include_round_losses = loss_cfg.get(
        "include_round_losses", mode == "legacy_dair"
    )
    if include_round_losses:
        total_loss = _add_round_losses(
            total_loss, output_dict, fused_target, hypes, criterion
        )

    return total_loss
