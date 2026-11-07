# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.pre_processor.base_preprocessor import BasePreprocessor
from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
from opencood.data_utils.pre_processor.e3dsnn_voxel_preprocessor import E3DSNNVoxelPreprocessor

__all__ = {
    'BasePreprocessor': BasePreprocessor,
    'SpVoxelPreprocessor': SpVoxelPreprocessor,
    'E3DSNNVoxelPreprocessor': E3DSNNVoxelPreprocessor
}


def build_preprocessor(preprocess_cfg, train):
    process_method_name = preprocess_cfg['core_method']
    error_message = f"{process_method_name} is not found. " \
                     f"Please add your processor file's name in opencood/" \
                     f"data_utils/processor/init.py"
    assert process_method_name in ['BasePreprocessor', 'VoxelPreprocessor',
                                   'BevPreprocessor', 'SpVoxelPreprocessor',
                                   'E3DSNNVoxelPreprocessor'], \
        error_message

    processor = __all__[process_method_name](
        preprocess_params=preprocess_cfg,
        train=train
    )

    return processor
