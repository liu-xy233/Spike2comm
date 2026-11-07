# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.early_fusion_dataset_dair import EarlyFusionDatasetDAIR
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair import IntermediateFusionDatasetDAIR
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_cost import IntermediateFusionDatasetDAIRCoST
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_e3dsnn import IntermediateFusionDatasetDAIRE3DSNN
from opencood.data_utils.datasets.intermediate_fusion_dataset_opv2v_e3dsnn import IntermediateFusionDatasetOPV2VE3DSNN
from opencood.data_utils.datasets.intermediate_fusion_dataset_v2xsim_e3dsnn import IntermediateFusionDatasetV2XSimE3DSNN
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_e3dsnn_cost import IntermediateFusionDatasetDAIRE3DSNNCoST
from opencood.data_utils.datasets.late_fusion_dataset_dair import LateFusionDatasetDAIR

__all__ = {
    'EarlyFusionDatasetDAIR': EarlyFusionDatasetDAIR,
    'IntermediateFusionDatasetDAIR': IntermediateFusionDatasetDAIR,
    'IntermediateFusionDatasetDAIRCoST': IntermediateFusionDatasetDAIRCoST,
    'IntermediateFusionDatasetDAIRE3DSNN': IntermediateFusionDatasetDAIRE3DSNN,
    'IntermediateFusionDatasetOPV2VE3DSNN': IntermediateFusionDatasetOPV2VE3DSNN,
    'IntermediateFusionDatasetV2XSimE3DSNN': IntermediateFusionDatasetV2XSimE3DSNN,
    'IntermediateFusionDatasetDAIRE3DSNNCoST': IntermediateFusionDatasetDAIRE3DSNNCoST,
    'LateFusionDatasetDAIR': LateFusionDatasetDAIR
}

# the final range for evaluation
GT_RANGE_OPV2V = [-140, -40, -3, 140, 40, 1]
GT_RANGE_V2XSIM = [-32, -32, -3, 32, 32, 1]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
