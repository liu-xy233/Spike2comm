# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn


def _extract_state_dict(checkpoint):
    """从常见checkpoint封装中提取模型参数。"""

    state = checkpoint

    if isinstance(state, nn.Module):
        state = state.state_dict()

    if isinstance(state, dict):
        for field in (
            "model",
            "state_dict",
            "model_state_dict",
            "model_ema",
        ):
            value = state.get(field)

            if isinstance(value, nn.Module):
                state = value.state_dict()
                print("使用checkpoint字段:", field)
                break

            if isinstance(value, dict):
                state = value
                print("使用checkpoint字段:", field)
                break

    if not isinstance(state, dict):
        raise TypeError(
            "无法提取state_dict，当前对象类型为：{}".format(
                type(state)
            )
        )

    cleaned = {}

    for key, value in state.items():
        if not torch.is_tensor(value):
            continue

        # 去掉训练框架可能添加的前缀。
        changed = True

        while changed:
            changed = False

            for prefix in (
                "module.",
                "model.",
                "backbone.",
            ):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True

        cleaned[key] = value

    return cleaned


def _replace_prefix(key, source_prefix, target_prefix):
    if not key.startswith(source_prefix):
        return None

    return target_prefix + key[len(source_prefix):]


def _internal_name_variants(key):
    """
    本地移植代码可能将官方的Conv属性改成小写conv。
    同时保留未修改版本。
    """

    variants = [key]

    replacements = (
        (".Conv.", ".conv."),
        (".conv.", ".Conv."),
    )

    for old, new in replacements:
        if old in key:
            variants.append(key.replace(old, new))

    # 去重并保持顺序。
    output = []

    for item in variants:
        if item not in output:
            output.append(item)

    return output


def _candidate_target_keys(source_key):
    candidates = []

    # 下采样模块名称在两个网络中原则上相同。
    for prefix in (
        "downsample1_2.",
        "downsample2.",
        "downsample3.",
        "downsample4.",
    ):
        if source_key.startswith(prefix):
            candidates.append(source_key)

    prefix_mapping = (
        (
            "ConvBlock1_1.0.",
            (
                "stage1_block1.",
                "stage1_block1.0.",
            ),
        ),
        (
            "ConvBlock1_2.0.",
            (
                "stage1_block2.",
                "stage1_block2.0.",
            ),
        ),
        (
            "ConvBlock2_1.0.",
            (
                "stage2_blocks.0.",
                "stage2_block1.",
                "stage2_block1.0.",
            ),
        ),
        (
            "ConvBlock2_2.0.",
            (
                "stage2_blocks.1.",
                "stage2_block2.",
                "stage2_block2.0.",
            ),
        ),
        (
            "block3.",
            (
                "stage3_blocks.",
                "block3.",
            ),
        ),
        (
            "block4.",
            (
                "stage4_blocks.",
                "block4.",
            ),
        ),
    )

    for source_prefix, target_prefixes in prefix_mapping:
        if not source_key.startswith(source_prefix):
            continue

        for target_prefix in target_prefixes:
            candidate = _replace_prefix(
                source_key,
                source_prefix,
                target_prefix,
            )

            candidates.extend(
                _internal_name_variants(candidate)
            )

    # 若本地仍使用官方名称，也允许直接匹配。
    candidates.extend(
        _internal_name_variants(source_key)
    )

    output = []

    for key in candidates:
        if key not in output:
            output.append(key)

    return output


def load_sdtv3_imagenet_backbone(
    backbone,
    checkpoint_path,
    reset_bn_stats=True,
    enforce_validity=True,
):
    """
    将官方SDTv3 19M ImageNet参数部分加载到本地BEV backbone。

    明确跳过：
      1. downsample1_1：官方输入3通道，本地输入64通道；
      2. head：ImageNet 1000类分类头；
      3. BN运行统计：在LiDAR BEV域重新统计；
      4. QFPN、BEV适配层：官方分类模型中不存在。
    """

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "权重文件不存在：{}".format(checkpoint_path)
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    source_state = _extract_state_dict(checkpoint)
    target_state = backbone.state_dict()

    loadable = {}
    matched_source_keys = set()

    skipped_stem = []
    skipped_head = []
    skipped_bn_stats = []
    shape_mismatch = []
    unmatched = []

    bn_stat_suffixes = (
        "running_mean",
        "running_var",
        "num_batches_tracked",
    )

    for source_key, source_tensor in source_state.items():
        if source_key.startswith("downsample1_1."):
            skipped_stem.append(source_key)
            continue

        if source_key.startswith("head."):
            skipped_head.append(source_key)
            continue

        if source_key.endswith(bn_stat_suffixes):
            skipped_bn_stats.append(source_key)
            continue

        candidates = _candidate_target_keys(source_key)

        name_found = False
        loaded = False

        for target_key in candidates:
            if target_key not in target_state:
                continue

            name_found = True

            source_shape = tuple(source_tensor.shape)
            target_shape = tuple(target_state[target_key].shape)

            if source_shape != target_shape:
                shape_mismatch.append(
                    (
                        source_key,
                        target_key,
                        source_shape,
                        target_shape,
                    )
                )
                continue

            # 防止多个源参数意外映射到同一个目标参数。
            if target_key in loadable:
                continue

            loadable[target_key] = source_tensor
            matched_source_keys.add(source_key)
            loaded = True
            break

        if not loaded:
            unmatched.append(
                (
                    source_key,
                    tuple(source_tensor.shape),
                    candidates[:6],
                    name_found,
                )
            )

    load_result = backbone.load_state_dict(
        loadable,
        strict=False,
    )

    # 图像域和BEV域差异较大，重新估计BN运行统计。
    if reset_bn_stats:
        for module in backbone.modules():
            if isinstance(
                module,
                nn.modules.batchnorm._BatchNorm,
            ):
                module.reset_running_stats()

    loaded_parameter_count = sum(
        tensor.numel()
        for tensor in loadable.values()
    )

    target_parameter_count = sum(
        tensor.numel()
        for tensor in target_state.values()
        if torch.is_floating_point(tensor)
    )

    stage_counts = {
        "stage1_block1": sum(
            key.startswith("stage1_block1.")
            for key in loadable
        ),
        "stage1_block2": sum(
            key.startswith("stage1_block2.")
            for key in loadable
        ),
        "stage2": sum(
            key.startswith("stage2_blocks.")
            for key in loadable
        ),
        "stage3": sum(
            key.startswith("stage3_blocks.")
            for key in loadable
        ),
        "stage4": sum(
            key.startswith("stage4_blocks.")
            for key in loadable
        ),
        "downsample": sum(
            key.startswith("downsample")
            for key in loadable
        ),
    }

    coverage = (
        100.0
        * loaded_parameter_count
        / max(target_parameter_count, 1)
    )

    print()
    print("========== SDTv3 ImageNet加载报告 ==========")
    print("checkpoint:", checkpoint_path)
    print("源参数张量数:", len(source_state))
    print("成功加载张量数:", len(loadable))
    print("成功加载参数量:", loaded_parameter_count)
    print("目标backbone参数量:", target_parameter_count)
    print("参数覆盖率: {:.2f}%".format(coverage))
    print("跳过RGB输入层:", len(skipped_stem))
    print("跳过分类头:", len(skipped_head))
    print("跳过BN运行统计:", len(skipped_bn_stats))
    print("形状不匹配:", len(shape_mismatch))
    print("未匹配源张量:", len(unmatched))

    for name, count in stage_counts.items():
        print("{:<18}: {}".format(name, count))

    print(
        "load_state_dict missing:",
        len(load_result.missing_keys),
    )
    print(
        "load_state_dict unexpected:",
        len(load_result.unexpected_keys),
    )

    if shape_mismatch:
        print("\n前20个形状不匹配项：")

        for item in shape_mismatch[:20]:
            print(" ", item)

    if unmatched:
        print("\n前30个未匹配项：")

        for source_key, shape, candidates, name_found in unmatched[:30]:
            print(
                "  source={} shape={} name_found={} candidates={}".format(
                    source_key,
                    shape,
                    name_found,
                    candidates,
                )
            )

    print("===========================================")
    print()

    report = {
        "loaded_tensor_count": len(loadable),
        "loaded_parameter_count": loaded_parameter_count,
        "target_parameter_count": target_parameter_count,
        "coverage_percent": coverage,
        "stage_counts": stage_counts,
        "shape_mismatch": shape_mismatch,
        "unmatched": unmatched,
        "missing_keys": load_result.missing_keys,
    }

    if enforce_validity:
        errors = []

        if loaded_parameter_count < 5_000_000:
            errors.append(
                "成功加载参数量小于500万"
            )

        if stage_counts["stage3"] == 0:
            errors.append(
                "Stage 3没有加载任何参数"
            )

        if stage_counts["stage4"] == 0:
            errors.append(
                "Stage 4没有加载任何参数"
            )

        if errors:
            raise RuntimeError(
                "ImageNet参数映射检查失败：{}".format(
                    "；".join(errors)
                )
            )

    return report
