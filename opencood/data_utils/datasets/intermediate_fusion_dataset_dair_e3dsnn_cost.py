# -*- coding: utf-8 -*-
"""DAIR-V2X E-3DSNN dataset entry with CoST temporal history packaging."""

from opencood.data_utils.datasets.intermediate_fusion_dataset_dair_cost import (
    IntermediateFusionDatasetDAIRCoST,
)


class IntermediateFusionDatasetDAIRE3DSNNCoST(IntermediateFusionDatasetDAIRCoST):
    """E-3DSNN CoST dataset hook.

    The actual voxelization remains controlled by ``E3DSNNVoxelPreprocessor`` in
    the yaml; this class only adds CoST history/prior metadata on top of the
    existing DAIR intermediate-fusion collation.
    """

    pass

