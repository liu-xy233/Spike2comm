# -*- coding: utf-8 -*-
"""DAIR-V2X intermediate-fusion dataset entry for the E-3DSNN local encoder.

The DAIR cooperative sample organization, labels, pairwise transforms, and
Where2comm collation remain identical to the existing DAIR intermediate-fusion
dataset.  This class exists as an explicit configuration hook for the E-3DSNN
voxel/preprocessor path, so experiments can be selected without touching the
legacy dataset entry.
"""

from opencood.data_utils.datasets.intermediate_fusion_dataset_dair import (
    IntermediateFusionDatasetDAIR,
)


class IntermediateFusionDatasetDAIRE3DSNN(IntermediateFusionDatasetDAIR):
    """DAIR-V2X reader compatible with E3DSNNVoxelPreprocessor."""

    pass
