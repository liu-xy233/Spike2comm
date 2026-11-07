import unittest

import torch

from opencood.models.comm_modules.value_predictor import (
    CounterfactualValuePredictor,
    generate_patch_coordinates,
)
from opencood.models.comm_modules.patch_utils import (
    patch_id_to_slice,
    patchify_map,
)


class CounterfactualValuePredictorTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "enabled": True,
            "in_channels": 256,
            "embed_dim": 256,
            "patch_size": [2, 3],
            "num_patch": 1050,
            "transformer_layers": 2,
            "transformer_heads": 4,
            "meta_dim": 4,
            "coord_embed_dim": 64,
            "use_coord_embedding": True,
            "use_output_norm": True,
            "dropout": 0.0,
        }

    def test_forward_is_finite_and_backward_runs(self):
        torch.manual_seed(7)
        model = CounterfactualValuePredictor(self.config)
        ego_feature = torch.randn(1, 256, 50, 126, requires_grad=True)
        metadata = torch.randn(1, 4, requires_grad=True)

        value_map = model(ego_feature, metadata)

        self.assertEqual(value_map.shape, (1, 1050))
        self.assertTrue(bool(torch.isfinite(value_map).all()))

        value_map.square().mean().backward()
        self.assertIsNotNone(ego_feature.grad)
        self.assertIsNotNone(metadata.grad)
        self.assertTrue(bool(torch.isfinite(ego_feature.grad).all()))
        self.assertTrue(bool(torch.isfinite(metadata.grad).all()))
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.parameters()
        ))

        with torch.no_grad():
            output = model(
                ego_feature.detach(), metadata.detach(), return_dict=True
            )
        self.assertEqual(set(output), {"value_map", "patch_tokens"})
        self.assertEqual(output["value_map"].shape, (1, 1050))
        self.assertEqual(output["patch_tokens"].shape, (1, 1050, 256))

    def test_rejects_patch_count_mismatch(self):
        model = CounterfactualValuePredictor(self.config)
        ego_feature = torch.randn(1, 256, 48, 126)
        metadata = torch.randn(1, 4)

        with self.assertRaisesRegex(ValueError, "configured num_patch"):
            model(ego_feature, metadata)

    def test_patch_index_matches_oracle_row_major_order(self):
        torch.manual_seed(11)
        patch_id = int(torch.randint(0, 1050, (1,)).item())
        ego_feature = torch.zeros(1, 256, 50, 126)
        h_slice, w_slice = patch_id_to_slice(
            patch_id, 50, 126, (2, 3)
        )
        ego_feature[:, :, h_slice, w_slice] = 1.0

        # This is the exact patchify + rearrange path used by ValueNet.
        patches = patchify_map(ego_feature, (2, 3))
        patch_tokens = patches.permute(0, 2, 1, 3, 4).contiguous().flatten(2)
        active_patch_ids = torch.nonzero(
            patch_tokens.abs().sum(dim=-1)[0] > 0,
            as_tuple=False,
        ).flatten()

        self.assertTrue(torch.equal(active_patch_ids, torch.tensor([patch_id])))

        coordinates = generate_patch_coordinates(50, 126, (2, 3))
        self.assertEqual(coordinates.shape, (1050, 2))
        self.assertTrue(bool((coordinates >= -1.0).all()))
        self.assertTrue(bool((coordinates <= 1.0).all()))
        # In row-major ordering x changes first; y changes after 42 patches.
        self.assertGreater(float(coordinates[1, 0]), float(coordinates[0, 0]))
        self.assertEqual(float(coordinates[1, 1]), float(coordinates[0, 1]))
        self.assertEqual(float(coordinates[42, 0]), float(coordinates[0, 0]))
        self.assertGreater(float(coordinates[42, 1]), float(coordinates[0, 1]))


if __name__ == "__main__":
    unittest.main()
