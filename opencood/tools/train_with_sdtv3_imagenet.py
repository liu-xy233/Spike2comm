# -*- coding: utf-8 -*-

import os

from opencood.tools import train as original_train
from opencood.tools import train_utils
from opencood.tools.sdtv3_imagenet_loader import (
    load_sdtv3_imagenet_backbone,
)


_ORIGINAL_CREATE_MODEL = train_utils.create_model


def create_model_with_imagenet_init(hypes):
    model = _ORIGINAL_CREATE_MODEL(hypes)

    checkpoint_path = os.environ.get(
        "SDTV3_IMAGENET_CKPT",
        "",
    )

    if not checkpoint_path:
        raise RuntimeError(
            "没有设置环境变量SDTV3_IMAGENET_CKPT"
        )

    if not hasattr(model, "backbone"):
        raise AttributeError(
            "模型没有backbone属性，实际模型类型为：{}".format(
                type(model)
            )
        )

    print(
        "正在加载SDTv3 ImageNet权重：{}".format(
            checkpoint_path
        ),
        flush=True,
    )

    report = load_sdtv3_imagenet_backbone(
        model.backbone,
        checkpoint_path,
        reset_bn_stats=True,
        enforce_validity=True,
    )

    model.sdtv3_imagenet_init_report = report

    return model


def main():
    # OpenCOOD train.py通常通过train_utils.create_model创建模型。
    train_utils.create_model = create_model_with_imagenet_init

    # 对不同版本的train.py再做两层兼容。
    if hasattr(original_train, "train_utils"):
        original_train.train_utils.create_model = (
            create_model_with_imagenet_init
        )

    if hasattr(original_train, "create_model"):
        original_train.create_model = (
            create_model_with_imagenet_init
        )

    original_train.main()


if __name__ == "__main__":
    main()
