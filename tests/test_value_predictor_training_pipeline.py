import os
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from opencood.data_utils.value_predictor_dataset import ValuePredictorDataset
from opencood.models.comm_modules.value_prediction_loss import (
    ValuePredictionLoss,
)
from opencood.models.comm_modules.value_predictor import (
    CounterfactualValuePredictor,
)
from opencood.tools.extract_value_predictor_features import (
    build_neighbor_metadata,
)


class ValuePredictorTrainingPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.feature_dir = os.path.join(self.temp_dir.name, "features")
        os.makedirs(self.feature_dir)
        self.utility_path = os.path.join(self.temp_dir.name, "utility.pt")

        labels = []
        for sample_id in (7, 9):
            torch.save(
                {
                    "sample_id": sample_id,
                    "ego_feature": torch.randn(256, 50, 126),
                    "neighbor_metadata": torch.randn(4),
                },
                os.path.join(
                    self.feature_dir, "sample_{:04d}.pt".format(sample_id)
                ),
            )
            labels.append(
                {
                    "sample_id": sample_id,
                    "utility": torch.linspace(-1.0, 1.0, 1050),
                    "patch_size": [2, 3],
                    "feature_size": [50, 126],
                }
            )
        torch.save(labels, self.utility_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_shapes(self):
        dataset = ValuePredictorDataset(
            self.feature_dir, self.utility_path
        )
        sample = dataset[0]
        self.assertEqual(len(dataset), 2)
        self.assertEqual(sample["ego_feature"].shape, (256, 50, 126))
        self.assertEqual(sample["neighbor_metadata"].shape, (4,))
        self.assertEqual(sample["utility"].shape, (1050,))

    def test_model_loss_forward_and_backward(self):
        dataset = ValuePredictorDataset(
            self.feature_dir, self.utility_path
        )
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        model = CounterfactualValuePredictor(
            {
                "in_channels": 256,
                "embed_dim": 32,
                "patch_size": [2, 3],
                "num_patch": 1050,
                "transformer_layers": 1,
                "transformer_heads": 4,
                "transformer_ffn_dim": 32,
                "meta_dim": 4,
                "coord_embed_dim": 16,
                "dropout": 0.0,
            }
        )
        criterion = ValuePredictionLoss(
            {
                "lambda_rank": 0.5,
                "margin": 0.1,
                "pair_sampling": "topk",
                "num_pairs": 64,
                "topk": 8,
            }
        )

        prediction = model(
            batch["ego_feature"], batch["neighbor_metadata"]
        )
        losses = criterion(prediction, batch["utility"])

        self.assertEqual(prediction.shape, (2, 1050))
        self.assertTrue(bool(torch.isfinite(losses["total_loss"])))
        losses["total_loss"].backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.parameters()
        ))

    def test_neighbor_metadata_from_pairwise_transform(self):
        transform = torch.eye(4).repeat(1, 2, 2, 1, 1)
        angle = torch.tensor(0.5)
        transform[0, 0, 1, 0, 0] = torch.cos(angle)
        transform[0, 0, 1, 0, 1] = -torch.sin(angle)
        transform[0, 0, 1, 1, 0] = torch.sin(angle)
        transform[0, 0, 1, 1, 1] = torch.cos(angle)
        transform[0, 0, 1, 0, 3] = 3.0
        transform[0, 0, 1, 1, 3] = 4.0

        metadata = build_neighbor_metadata(transform)

        self.assertEqual(metadata.shape, (4,))
        self.assertTrue(torch.allclose(
            metadata, torch.tensor([3.0, 4.0, 0.5, 5.0])
        ))


if __name__ == "__main__":
    unittest.main()
