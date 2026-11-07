import torch
import torch.nn.functional as F

from opencood.models.sub_modules.e3dsnn_bev_adapter import (
    SparseToBEVScatterAdapter,
)
from opencood.models.sub_modules.e3dsnn_multiscale_enhancement_neck import (
    MultiScaleEnhancementNeck,
)


class FakeSparseTensor:
    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size


def make_dense_sparse(batch_size, channels, depth, height, width):
    dense = torch.randn(batch_size, channels, depth, height, width)
    indices = []
    features = []
    for batch_index in range(batch_size):
        for z_index in range(depth):
            for y_index in range(height):
                for x_index in range(width):
                    indices.append([batch_index, z_index, y_index, x_index])
                    features.append(dense[batch_index, :, z_index, y_index, x_index])
    return dense, FakeSparseTensor(
        torch.stack(features),
        torch.tensor(indices, dtype=torch.int32),
        [depth, height, width],
        batch_size,
    )


def test_sparse_scatter_projection_matches_dense_height_collapse():
    dense, sparse = make_dense_sparse(2, 3, 4, 5, 7)
    adapter = SparseToBEVScatterAdapter(3, 4, 6).eval()

    actual = adapter(sparse)
    dense_collapsed = dense.view(2, 3 * 4, 5, 7)
    reference_weight = adapter.weight.view(6, 3 * 4, 1, 1)
    expected = F.conv2d(dense_collapsed, reference_weight)
    expected = adapter.relu(adapter.bn(expected))

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_multiscale_neck_returns_one_shape_compatible_bev_feature():
    _, stage2 = make_dense_sparse(2, 4, 3, 20, 28)
    _, stage3 = make_dense_sparse(2, 6, 2, 10, 14)
    _, stage4 = make_dense_sparse(2, 8, 1, 5, 7)
    original = torch.randn(2, 12, 5, 7)
    neck = MultiScaleEnhancementNeck(
        stage_channels=[4, 6, 8],
        stage_depths=[3, 2, 1],
        output_channels=12,
        neck_channels=5,
        alpha_init=0.1,
    )

    enhanced, debug = neck(
        stage2, stage3, stage4, original, return_debug=True
    )

    assert enhanced.shape == original.shape
    assert debug["stage2_bev"] == (2, 4, 20, 28)
    assert debug["stage3_bev"] == (2, 6, 10, 14)
    assert debug["stage4_bev"] == (2, 8, 5, 7)
    assert debug["p2"] == (2, 5, 20, 28)
    assert debug["p_down2"] == (2, 5, 5, 7)
    assert debug["f_ms"] == tuple(original.shape)
    assert debug["f_enhanced"] == tuple(original.shape)
