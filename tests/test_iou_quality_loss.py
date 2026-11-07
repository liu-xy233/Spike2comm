import unittest
from unittest import mock

import torch

from opencood.loss.point_pillar_loss import PointPillarLoss


def make_loss(enabled=True):
    return PointPillarLoss(
        {
            "cls_weight": 1.0,
            "reg": 2.0,
            "box_order": "hwl",
            "iou_quality_head": {
                "enabled": enabled,
                "loss_weight": 1.0,
            },
            "anchor_args": {
                "W": 1,
                "H": 1,
                "feature_stride": 1,
                "cav_lidar_range": [-1.0, -1.0, -2.0, 1.0, 1.0, 2.0],
                "vw": 0.5,
                "vh": 0.5,
                "h": 1.5,
                "w": 2.0,
                "l": 4.0,
                "r": [0, 90],
            },
        }
    )


class IoUQualityLossTest(unittest.TestCase):
    def test_lightweight_shared_head_shapes_and_parameter_count(self):
        head = torch.nn.Conv2d(256, 2, kernel_size=1)
        local_feature = torch.randn(3, 256, 4, 5)
        fused_feature = torch.randn(3, 256, 4, 5)

        iou_single = head(local_feature)
        iou = head(fused_feature)

        self.assertEqual(iou_single.shape, (3, 2, 4, 5))
        self.assertEqual(iou.shape, (3, 2, 4, 5))
        self.assertEqual(sum(p.numel() for p in head.parameters()), 514)

    def test_target_scaling_endpoints(self):
        actual = PointPillarLoss.scale_iou_target(torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(actual, torch.tensor([-1.0, 1.0]))

    def test_target_uses_only_positives_and_is_detached(self):
        criterion = make_loss()
        box_deltas = torch.zeros((1, 2, 7), requires_grad=True)
        reg_targets = torch.zeros((1, 2, 7))
        positives = torch.tensor([[True, False]])

        def fake_iou(pred_boxes, gt_boxes):
            self.assertEqual(pred_boxes.shape, (1, 7))
            self.assertEqual(gt_boxes.shape, (1, 7))
            return torch.tensor([[0.75]], requires_grad=True)

        target = criterion.build_iou_targets(
            box_deltas, reg_targets, positives, iou_calculator=fake_iou
        )

        self.assertEqual(target.shape, (1,))
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target, torch.tensor([0.5]))

    def test_negative_anchor_does_not_contribute_to_iou_loss(self):
        criterion = make_loss()
        output = {
            "psm": torch.zeros((1, 2, 1, 1)),
            "rm": torch.zeros((1, 14, 1, 1)),
            "iou": torch.tensor([[[[0.0]], [[100.0]]]], requires_grad=True),
        }
        target = {
            "targets": torch.zeros((1, 1, 1, 14)),
            "pos_equal_one": torch.tensor([[[[1.0, 0.0]]]]),
        }

        with mock.patch.object(
            criterion, "build_iou_targets", return_value=torch.tensor([0.0])
        ):
            criterion(output, target)

        torch.testing.assert_close(
            criterion.loss_dict["iou_loss"], torch.tensor(0.0)
        )

    def test_disabled_loss_does_not_require_iou_output(self):
        criterion = make_loss(enabled=False)
        output = {
            "psm": torch.zeros((1, 2, 1, 1)),
            "rm": torch.zeros((1, 14, 1, 1)),
        }
        target = {
            "targets": torch.zeros((1, 1, 1, 14)),
            "pos_equal_one": torch.zeros((1, 1, 1, 2)),
        }
        criterion(output, target)
        torch.testing.assert_close(
            criterion.loss_dict["iou_loss"], torch.tensor(0.0)
        )


if __name__ == "__main__":
    unittest.main()
