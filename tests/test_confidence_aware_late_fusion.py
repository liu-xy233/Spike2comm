import unittest
from unittest import mock
import sys
import types

import numpy as np


# The lightweight CI environment used for these logic tests may not ship the
# project's Shapely runtime dependency.  Provide the small axis-aligned Polygon
# surface needed by box_utils.nms_rotated; production keeps using real Shapely.
try:
    from shapely.geometry import Polygon  # noqa: F401
except ImportError:
    class _RectanglePolygon(object):
        def __init__(self, points=None, area=None):
            if area is not None:
                self.area = float(area)
                self.bounds = None
                return
            points = np.asarray(points, dtype=np.float32)
            self.bounds = (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
            self.area = max(0.0, self.bounds[2] - self.bounds[0]) * max(
                0.0, self.bounds[3] - self.bounds[1]
            )

        def intersection(self, other):
            x_min = max(self.bounds[0], other.bounds[0])
            y_min = max(self.bounds[1], other.bounds[1])
            x_max = min(self.bounds[2], other.bounds[2])
            y_max = min(self.bounds[3], other.bounds[3])
            area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
            return _RectanglePolygon(area=area)

        def union(self, other):
            intersection = self.intersection(other).area
            return _RectanglePolygon(area=self.area + other.area - intersection)

    shapely_module = types.ModuleType("shapely")
    geometry_module = types.ModuleType("shapely.geometry")
    geometry_module.Polygon = _RectanglePolygon
    shapely_module.geometry = geometry_module
    sys.modules["shapely"] = shapely_module
    sys.modules["shapely.geometry"] = geometry_module

try:
    from icecream import ic  # noqa: F401
except ImportError:
    icecream_module = types.ModuleType("icecream")
    icecream_module.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
    sys.modules["icecream"] = icecream_module

try:
    from pyquaternion import Quaternion  # noqa: F401
except ImportError:
    pyquaternion_module = types.ModuleType("pyquaternion")
    pyquaternion_module.Quaternion = object
    sys.modules["pyquaternion"] = pyquaternion_module

try:
    from opencood.utils.box_overlaps import bbox_overlaps  # noqa: F401
except ImportError:
    box_overlaps_module = types.ModuleType("opencood.utils.box_overlaps")
    box_overlaps_module.bbox_overlaps = lambda *args, **kwargs: None
    sys.modules["opencood.utils.box_overlaps"] = box_overlaps_module

try:
    import cv2  # noqa: F401
except ImportError:
    vis_utils_module = types.ModuleType("opencood.visualization.vis_utils")
    vis_utils_module.visualize_single_sample_output_gt = lambda *args, **kwargs: None
    sys.modules["opencood.visualization.vis_utils"] = vis_utils_module

import torch

import opencood.data_utils.post_processor.confidence_aware_late_fusion as late_fusion_module
import opencood.data_utils.post_processor.voxel_postprocessor as voxel_postprocessor_module
from opencood.data_utils.post_processor.confidence_aware_late_fusion import (
    ConfidenceAwareLateFusion,
    rectify_iou_aware_scores,
    rotated_box_voting,
)
from opencood.data_utils.post_processor.voxel_postprocessor import (
    VoxelPostprocessor,
)
from opencood.tools.eval_iou_nms_diagnostic import (
    analyze_frame,
    best_gt_bev_matches,
    finalize_summary,
    new_totals,
    run_nms,
    update_totals,
)
from opencood.utils import box_utils


def make_box(x, y, length=4.0, width=2.0, z=0.0, yaw=0.0):
    centers = torch.tensor(
        [[x, y, z, 1.5, width, length, yaw]], dtype=torch.float32
    )
    return box_utils.boxes_to_corners_3d(centers, order="hwl")


class ConfidenceAwareLateFusionTest(unittest.TestCase):
    def make_fuser(self, enable_rbv=None, rbv_iou_threshold=None):
        return ConfidenceAwareLateFusion(
            score_threshold=0.4,
            collaborator_score_weight=0.7,
            base_score_threshold=0.2,
            nms_threshold=0.15,
            enable_rotated_box_voting=enable_rbv,
            rbv_iou_threshold=rbv_iou_threshold,
        )

    def make_ordinary_postprocessor(self, quality_enabled=False,
                                    inference_mode=None,
                                    quality_power=4.0):
        quality_cfg = {
            "enabled": quality_enabled,
            "quality_power": quality_power,
        }
        if inference_mode is not None:
            quality_cfg["inference_mode"] = inference_mode
        params = {
            "anchor_args": {"num": 2},
            "order": "hwl",
            "target_args": {"score_threshold": 0.2},
            "nms_thresh": 0.15,
            "gt_range": [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0],
            "confidence_aware_late_fusion": {"enabled": False},
            "iou_quality_head": quality_cfg,
        }
        return VoxelPostprocessor(params, train=False)

    @staticmethod
    def make_ordinary_inputs(logits=(2.1972246, 1.3862944),
                             iou_preds=(-1.0, 1.0),
                             centers_x=(0.0, 0.2)):
        anchor = torch.tensor(
            [[[[centers_x[0], 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
               [centers_x[1], 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]]],
            dtype=torch.float32,
        )
        data_dict = {
            "ego": {
                "transformation_matrix": torch.eye(4),
                "anchor_box": anchor,
            }
        }
        output_dict = {
            "ego": {
                "psm": torch.tensor(logits).view(1, 2, 1, 1),
                "rm": torch.zeros((1, 14, 1, 1)),
                "iou": torch.tensor(iou_preds).view(1, 2, 1, 1),
            }
        }
        return data_dict, output_dict

    @staticmethod
    def make_diagnostic_candidates(boxes, raw_scores, qualities,
                                   quality_power=1.0):
        return {
            "boxes": boxes,
            "raw_scores": raw_scores,
            "quality_scores": raw_scores * qualities.pow(quality_power),
            "predicted_quality": qualities,
            "candidate_indices": torch.arange(boxes.shape[0]),
        }

    def test_filter_then_weight(self):
        fuser = self.make_fuser()
        boxes = torch.cat([make_box(i * 10.0, 0.0) for i in range(4)])
        scores = torch.tensor([0.2, 0.4, 0.5, 0.8])

        kept_boxes, weighted = fuser.filter_and_weight_collaborator(
            boxes, scores
        )

        self.assertEqual(kept_boxes.shape[0], 3)
        self.assertTrue(
            torch.allclose(weighted, torch.tensor([0.28, 0.35, 0.56]))
        )

    def test_iou_quality_disabled_is_identity(self):
        cls_score = torch.tensor([0.8])
        actual = rectify_iou_aware_scores(
            cls_score, torch.tensor([-1.0]), enabled=False
        )
        self.assertTrue(torch.equal(actual, cls_score))

    def test_iou_quality_score_formula_and_monotonicity(self):
        cls_score = torch.tensor([0.8, 0.8, 0.8])
        # Raw predictions -1, 0, 1 map to qualities 0, 0.5, 1.
        actual = rectify_iou_aware_scores(
            cls_score, torch.tensor([-1.0, 0.0, 1.0]), quality_power=4.0
        )
        expected = torch.tensor([0.0, 0.8 * 0.5 ** 4, 0.8])
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(bool(torch.all(actual[1:] > actual[:-1])))

    def test_collaborator_raw_gate_precedes_quality_ranking(self):
        fuser = self.make_fuser()
        boxes = torch.cat([make_box(0.0, 0.0), make_box(10.0, 0.0)])
        raw_scores = torch.tensor([0.39, 0.5])
        ranking_scores = torch.tensor([0.99, 0.4])

        kept_boxes, weighted = fuser.filter_and_weight_collaborator(
            boxes, raw_scores, ranking_scores
        )

        self.assertEqual(kept_boxes.shape[0], 1)
        self.assertTrue(torch.allclose(weighted, torch.tensor([0.28])))

    def test_ego_priority_and_nonoverlapping_collaborator_survives(self):
        fuser = self.make_fuser()
        ego_boxes = make_box(0.0, 0.0)
        ego_scores = torch.tensor([0.8])
        collaborator_boxes = torch.cat(
            [make_box(0.0, 0.0), make_box(20.0, 0.0)]
        )
        collaborator_scores = torch.tensor([0.9, 0.8])

        boxes, scores = fuser(
            ego_boxes,
            ego_scores,
            [(collaborator_boxes, collaborator_scores)],
        )

        self.assertEqual(boxes.shape[0], 2)
        self.assertTrue(torch.any(torch.isclose(scores, torch.tensor(0.8))))
        self.assertTrue(torch.any(torch.isclose(scores, torch.tensor(0.56))))
        self.assertFalse(torch.any(torch.isclose(scores, torch.tensor(0.63))))

    def test_empty_and_single_agent_inputs(self):
        fuser = self.make_fuser()
        boxes, scores = fuser(None, None, [])
        self.assertIsNone(boxes)
        self.assertIsNone(scores)

        ego_boxes = make_box(0.0, 0.0)
        ego_scores = torch.tensor([0.8])
        boxes, scores = fuser(ego_boxes, ego_scores, [])
        self.assertEqual(boxes.shape[0], 1)
        self.assertTrue(torch.equal(scores, ego_scores))

    def test_multiple_collaborators_are_concatenated(self):
        fuser = self.make_fuser()
        ego_boxes = make_box(0.0, 0.0)
        ego_scores = torch.tensor([0.8])
        collaborators = [
            (make_box(20.0, 0.0), torch.tensor([0.8])),
            (make_box(40.0, 0.0), torch.tensor([0.7])),
        ]
        boxes, scores = fuser(ego_boxes, ego_scores, collaborators)
        self.assertEqual(boxes.shape[0], 3)
        self.assertEqual(scores.shape, (3,))

    def test_exactly_one_nms_call(self):
        fuser = self.make_fuser()
        ego_boxes = make_box(0.0, 0.0)
        ego_scores = torch.tensor([0.8])
        collaborator = (make_box(20.0, 0.0), torch.tensor([0.8]))

        original_nms = box_utils.nms_rotated
        with mock.patch.object(
            box_utils, "nms_rotated", wraps=original_nms
        ) as mocked_nms:
            fuser(ego_boxes, ego_scores, [collaborator])
        self.assertEqual(mocked_nms.call_count, 1)

    def test_rbv_default_off_matches_original_output(self):
        boxes = torch.cat([make_box(0.0, 0.0), make_box(0.4, 0.0)])
        scores = torch.tensor([0.9, 0.6])
        fuser = self.make_fuser()

        with mock.patch.object(
            late_fusion_module,
            "rotated_box_voting",
            side_effect=AssertionError("RBV must stay off by default"),
        ):
            actual_boxes, actual_scores = fuser(boxes, scores, [])

        keep = box_utils.nms_rotated(boxes, scores, 0.15)
        self.assertTrue(torch.equal(actual_boxes, boxes[keep]))
        self.assertTrue(torch.equal(actual_scores, scores[keep]))

    def test_rbv_on_refines_only_center(self):
        first = make_box(0.0, 0.0)
        second = make_box(0.4, 0.0, z=0.1)
        boxes = torch.cat([first, second])
        scores = torch.tensor([0.9, 0.6])
        fuser = self.make_fuser(enable_rbv=True, rbv_iou_threshold=0.1)

        voted_boxes, voted_scores = fuser(boxes, scores, [])

        self.assertEqual(voted_boxes.shape[0], 1)
        self.assertTrue(torch.equal(voted_scores, torch.tensor([0.9])))
        voted_center = voted_boxes[0].mean(dim=0)
        self.assertGreater(float(voted_center[0]), 0.0)
        self.assertGreater(float(voted_center[2]), 0.0)
        self.assertTrue(
            torch.allclose(
                voted_boxes[0] - voted_center,
                first[0] - first[0].mean(dim=0),
                atol=1e-6,
            )
        )

    def test_rbv_enabled_still_calls_nms_once(self):
        boxes = torch.cat([make_box(0.0, 0.0), make_box(0.4, 0.0)])
        scores = torch.tensor([0.9, 0.6])
        fuser = self.make_fuser(enable_rbv=True, rbv_iou_threshold=0.1)
        original_nms = box_utils.nms_rotated

        with mock.patch.object(
            box_utils, "nms_rotated", wraps=original_nms
        ) as mocked_nms:
            fuser(boxes, scores, [])

        self.assertEqual(mocked_nms.call_count, 1)

    def test_rbv_quality_off_and_full_use_active_ranking_scores(self):
        ego_box = make_box(0.0, 0.0)
        collaborator_box = make_box(0.4, 0.0)
        ego_score = torch.tensor([0.8])
        raw_collaborator_score = torch.tensor([0.9])
        full_quality_score = rectify_iou_aware_scores(
            raw_collaborator_score,
            torch.tensor([0.6]),
            quality_power=4.0,
        )

        off_fuser = self.make_fuser(enable_rbv=True, rbv_iou_threshold=0.1)
        full_fuser = self.make_fuser(enable_rbv=True, rbv_iou_threshold=0.1)
        off_boxes, off_scores = off_fuser(
            ego_box,
            ego_score,
            [(collaborator_box, raw_collaborator_score)],
        )
        full_boxes, full_scores = full_fuser(
            ego_box,
            ego_score,
            [(
                collaborator_box,
                raw_collaborator_score,
                full_quality_score,
            )],
        )

        off_center_x = off_boxes[0, :, 0].mean()
        full_center_x = full_boxes[0, :, 0].mean()
        self.assertGreater(float(off_center_x), float(full_center_x))
        self.assertGreater(float(full_center_x), 0.0)
        self.assertTrue(torch.equal(off_scores, ego_score))
        self.assertTrue(torch.equal(full_scores, ego_score))

    def test_rbv_assigns_each_candidate_to_only_best_nms_box(self):
        nms_boxes = torch.cat([make_box(0.0, 0.0), make_box(10.0, 0.0)])
        candidates = torch.cat(
            [make_box(0.0, 0.0), make_box(10.0, 0.0), make_box(5.0, 0.0)]
        )
        scores = torch.ones(3)
        fake_iou = torch.tensor(
            [[1.0, 0.0, 0.6], [0.0, 1.0, 0.7]], dtype=torch.float32
        )

        with mock.patch.object(
            late_fusion_module,
            "pairwise_rotated_bev_iou",
            return_value=fake_iou,
        ):
            voted = rotated_box_voting(
                nms_boxes,
                candidates,
                scores,
                iou_threshold=0.5,
                score_power=1.0,
                iou_power=1.0,
            )

        self.assertTrue(torch.isclose(voted[0, :, 0].mean(), torch.tensor(0.0)))
        self.assertLess(float(voted[1, :, 0].mean()), 10.0)

    def test_decode_projects_box_to_ego_coordinates(self):
        params = {
            "anchor_args": {"num": 1},
            "order": "hwl",
        }
        postprocessor = VoxelPostprocessor(params, train=False)
        psm = torch.tensor([[[[10.0]]]])
        rm = torch.zeros((1, 7, 1, 1))
        anchor = torch.tensor(
            [[[[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]]]
        )
        agent_to_ego = torch.eye(4)
        agent_to_ego[0, 3] = 5.0
        agent_to_ego[1, 3] = -2.0

        boxes, scores = postprocessor._decode_candidates(
            psm, rm, anchor, agent_to_ego, score_threshold=None
        )[0]

        self.assertEqual(boxes.shape, (1, 8, 3))
        self.assertTrue(torch.allclose(boxes[0, :, 0].mean(), torch.tensor(5.0)))
        self.assertTrue(torch.allclose(boxes[0, :, 1].mean(), torch.tensor(-2.0)))
        self.assertGreater(float(scores[0]), 0.99)

    def test_disabled_config_uses_legacy_path(self):
        params = {
            "anchor_args": {"num": 1},
            "order": "hwl",
            "target_args": {"score_threshold": 0.2},
            "nms_thresh": 0.15,
            "gt_range": [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0],
            "confidence_aware_late_fusion": {"enabled": False},
        }
        postprocessor = VoxelPostprocessor(params, train=False)
        data_dict = {
            "ego": {
                "transformation_matrix": torch.eye(4),
                "anchor_box": torch.tensor(
                    [[[[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]]]
                ),
            }
        }
        output_dict = {
            "ego": {
                "psm": torch.tensor([[[[10.0]]]]),
                "rm": torch.zeros((1, 7, 1, 1)),
            }
        }

        with mock.patch.object(
            postprocessor,
            "_post_process_confidence_aware",
            side_effect=AssertionError("new path must stay disabled"),
        ):
            boxes, scores = postprocessor.post_process(data_dict, output_dict)

        self.assertEqual(boxes.shape, (1, 8, 3))
        self.assertEqual(scores.shape, (1,))

    def test_enabled_postprocessor_uses_all_local_outputs(self):
        params = {
            "anchor_args": {"num": 1},
            "order": "hwl",
            "target_args": {"score_threshold": 0.2},
            "nms_thresh": 0.15,
            "gt_range": [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0],
            "confidence_aware_late_fusion": {
                "enabled": True,
                "score_threshold": 0.3,
                "collaborator_score_weight": 0.8,
            },
        }
        postprocessor = VoxelPostprocessor(params, train=False)
        anchor = torch.tensor(
            [[[[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]]]
        )
        pairwise = torch.eye(4).repeat(1, 2, 2, 1, 1)
        pairwise[0, 1, 0, 0, 3] = 20.0
        data_dict = {
            "ego": {
                "transformation_matrix": torch.eye(4),
                "pairwise_t_matrix": pairwise,
                "record_len": torch.tensor([2]),
                "anchor_box": anchor,
            }
        }
        output_dict = {
            "ego": {
                "psm": torch.tensor([[[[1.3862944]]]]),  # sigmoid = 0.8
                "rm": torch.zeros((1, 7, 1, 1)),
                "psm_single": torch.tensor(
                    [[[[0.0]]], [[[2.1972246]]]]  # collaborator sigmoid = 0.9
                ),
                "rm_single": torch.zeros((2, 7, 1, 1)),
            }
        }

        boxes, scores = postprocessor.post_process(data_dict, output_dict)

        self.assertEqual(boxes.shape, (2, 8, 3))
        self.assertTrue(torch.any(torch.isclose(scores, torch.tensor(0.8))))
        self.assertTrue(torch.any(torch.isclose(scores, torch.tensor(0.72))))
        box_centers_x = boxes[:, :, 0].mean(dim=1).sort().values
        self.assertTrue(
            torch.allclose(box_centers_x, torch.tensor([0.0, 20.0]))
        )

    def test_quality_enabled_keeps_one_global_nms_then_range_filter(self):
        params = {
            "anchor_args": {"num": 1},
            "order": "hwl",
            "target_args": {"score_threshold": 0.01},
            "nms_thresh": 0.15,
            "gt_range": [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0],
            "confidence_aware_late_fusion": {
                "enabled": True,
                "score_threshold": 0.3,
                "collaborator_score_weight": 0.8,
            },
            "iou_quality_head": {
                "enabled": True,
                "quality_power": 4.0,
                "debug": False,
            },
        }
        postprocessor = VoxelPostprocessor(params, train=False)
        anchor = torch.tensor(
            [[[[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]]]
        )
        pairwise = torch.eye(4).repeat(1, 2, 2, 1, 1)
        pairwise[0, 1, 0, 0, 3] = 20.0
        data_dict = {
            "ego": {
                "transformation_matrix": torch.eye(4),
                "pairwise_t_matrix": pairwise,
                "record_len": torch.tensor([2]),
                "anchor_box": anchor,
            }
        }
        output_dict = {
            "ego": {
                "psm": torch.tensor([[[[1.3862944]]]]),
                "rm": torch.zeros((1, 7, 1, 1)),
                "iou": torch.ones((1, 1, 1, 1)),
                "psm_single": torch.tensor([[[[0.0]]], [[[2.1972246]]]]),
                "rm_single": torch.zeros((2, 7, 1, 1)),
                "iou_single": torch.ones((2, 1, 1, 1)),
            }
        }
        call_order = []
        original_nms = box_utils.nms_rotated
        original_range = box_utils.get_mask_for_boxes_within_range_torch

        def tracked_nms(*args, **kwargs):
            call_order.append("nms")
            return original_nms(*args, **kwargs)

        def tracked_range(*args, **kwargs):
            call_order.append("post_nms_range_filter")
            return original_range(*args, **kwargs)

        with mock.patch.object(box_utils, "nms_rotated", side_effect=tracked_nms) as nms:
            with mock.patch.object(
                box_utils,
                "get_mask_for_boxes_within_range_torch",
                side_effect=tracked_range,
            ):
                boxes, scores = postprocessor.post_process(data_dict, output_dict)

        self.assertEqual(nms.call_count, 1)
        self.assertEqual(call_order, ["nms", "post_nms_range_filter"])
        self.assertEqual(boxes.shape[0], 2)
        self.assertTrue(torch.any(torch.isclose(scores, torch.tensor(0.8))))
        self.assertTrue(torch.any(torch.isclose(scores, torch.tensor(0.72))))

    def test_nms_only_power_zero_matches_quality_off(self):
        data_dict, output_dict = self.make_ordinary_inputs(
            centers_x=(0.0, 10.0)
        )
        off = self.make_ordinary_postprocessor(quality_enabled=False)
        nms_only = self.make_ordinary_postprocessor(
            quality_enabled=True,
            inference_mode="nms_only",
            quality_power=0.0,
        )

        off_boxes, off_scores = off.post_process(data_dict, output_dict)
        nms_boxes, nms_scores = nms_only.post_process(data_dict, output_dict)

        self.assertTrue(torch.equal(nms_boxes, off_boxes))
        self.assertTrue(torch.equal(nms_scores, off_scores))

    def test_nms_only_power_does_not_change_raw_candidate_set(self):
        data_dict, output_dict = self.make_ordinary_inputs(
            logits=(-1.3249254, 2.1972246),
            iou_preds=(-1.0, 1.0),
            centers_x=(0.0, 10.0),
        )
        candidate_centers = []

        def keep_all(boxes, scores, threshold):
            candidate_centers.append(boxes.mean(dim=1).clone())
            return torch.arange(boxes.shape[0], device=boxes.device)

        for power in (0.25, 0.5):
            postprocessor = self.make_ordinary_postprocessor(
                quality_enabled=True,
                inference_mode="nms_only",
                quality_power=power,
            )
            with mock.patch.object(
                box_utils, "nms_rotated", side_effect=keep_all
            ):
                boxes, scores = postprocessor.post_process(
                    data_dict, output_dict
                )
            self.assertEqual(boxes.shape[0], 2)
            self.assertTrue(torch.allclose(
                scores, torch.tensor([0.21, 0.9]), atol=1e-5
            ))

        self.assertEqual(len(candidate_centers), 2)
        self.assertTrue(torch.equal(candidate_centers[0], candidate_centers[1]))

    def test_nms_only_quality_changes_selection_but_returns_raw_score(self):
        data_dict, output_dict = self.make_ordinary_inputs()
        off = self.make_ordinary_postprocessor(quality_enabled=False)
        nms_only = self.make_ordinary_postprocessor(
            quality_enabled=True,
            inference_mode="nms_only",
            quality_power=0.5,
        )
        off_boxes, off_scores = off.post_process(data_dict, output_dict)
        original_nms = box_utils.nms_rotated

        with mock.patch.object(
            voxel_postprocessor_module,
            "ConfidenceAwareLateFusion",
            side_effect=AssertionError("late fusion must remain disabled"),
        ):
            with mock.patch.object(
                late_fusion_module,
                "rotated_box_voting",
                side_effect=AssertionError("RBV must remain disabled"),
            ):
                with mock.patch.object(
                    box_utils, "nms_rotated", wraps=original_nms
                ) as nms:
                    boxes, scores = nms_only.post_process(
                        data_dict, output_dict
                    )

        self.assertEqual(nms.call_count, 1)
        self.assertTrue(torch.allclose(
            off_boxes[0].mean(dim=0)[0], torch.tensor(0.0), atol=1e-5
        ))
        self.assertTrue(torch.allclose(off_scores, torch.tensor([0.9])))
        self.assertTrue(torch.allclose(
            boxes[0].mean(dim=0)[0], torch.tensor(0.2), atol=1e-5
        ))
        self.assertTrue(torch.allclose(scores, torch.tensor([0.8])))

    def test_full_default_and_explicit_modes_are_equivalent(self):
        data_dict, output_dict = self.make_ordinary_inputs(
            centers_x=(0.0, 10.0)
        )
        default_full = self.make_ordinary_postprocessor(
            quality_enabled=True,
            inference_mode=None,
            quality_power=0.5,
        )
        explicit_full = self.make_ordinary_postprocessor(
            quality_enabled=True,
            inference_mode="full",
            quality_power=0.5,
        )

        default_boxes, default_scores = default_full.post_process(
            data_dict, output_dict
        )
        explicit_boxes, explicit_scores = explicit_full.post_process(
            data_dict, output_dict
        )

        self.assertTrue(torch.equal(default_boxes, explicit_boxes))
        self.assertTrue(torch.equal(default_scores, explicit_scores))

    def test_nms_only_empty_candidates_skip_nms(self):
        data_dict, output_dict = self.make_ordinary_inputs(
            logits=(-10.0, -9.0),
            iou_preds=(1.0, -1.0),
        )
        postprocessor = self.make_ordinary_postprocessor(
            quality_enabled=True,
            inference_mode="nms_only",
            quality_power=0.5,
        )

        with mock.patch.object(box_utils, "nms_rotated") as nms:
            boxes, scores = postprocessor.post_process(data_dict, output_dict)

        self.assertIsNone(boxes)
        self.assertIsNone(scores)
        nms.assert_not_called()

    def test_diagnostic_candidate_export_reuses_filtered_path(self):
        data_dict, output_dict = self.make_ordinary_inputs()
        postprocessor = self.make_ordinary_postprocessor(
            quality_enabled=True,
            inference_mode="nms_only",
            quality_power=1.0,
        )

        with mock.patch.object(box_utils, "nms_rotated") as nms:
            candidates = postprocessor.post_process(
                data_dict,
                output_dict,
                return_nms_candidates=True,
            )

        nms.assert_not_called()
        self.assertEqual(candidates["boxes"].shape[0], 2)
        self.assertTrue(torch.equal(
            candidates["candidate_indices"], torch.tensor([0, 1])
        ))
        self.assertTrue(torch.allclose(
            candidates["raw_scores"], torch.tensor([0.9, 0.8])
        ))
        self.assertTrue(torch.allclose(
            candidates["predicted_quality"], torch.tensor([0.0, 1.0])
        ))

    def test_diagnostic_power_zero_and_unchanged_decisions(self):
        boxes = torch.cat([make_box(0.0, 0.0), make_box(0.4, 0.0)])
        raw_scores = torch.tensor([0.9, 0.8])
        qualities = torch.tensor([0.0, 1.0])
        candidates = self.make_diagnostic_candidates(
            boxes, raw_scores, qualities, quality_power=0.0
        )
        gt_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]

        raw_keep = run_nms(candidates, "raw_scores", 0.15, gt_range)
        quality_keep = run_nms(
            candidates, "quality_scores", 0.15, gt_range
        )
        frame = analyze_frame(
            0, candidates, make_box(0.0, 0.0), 0.15, gt_range
        )

        self.assertTrue(np.array_equal(raw_keep, quality_keep))
        self.assertFalse(frame["changed"])
        self.assertEqual(frame["raw_only_count"], 0)
        self.assertEqual(frame["quality_only_count"], 0)

    def test_diagnostic_positive_and_negative_replacements(self):
        boxes = torch.cat([make_box(0.0, 0.0), make_box(0.4, 0.0)])
        candidates = self.make_diagnostic_candidates(
            boxes,
            torch.tensor([0.9, 0.8]),
            torch.tensor([0.1, 1.0]),
        )
        gt_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]
        positive = analyze_frame(
            0, candidates, make_box(0.4, 0.0), 0.15, gt_range
        )
        negative = analyze_frame(
            1, candidates, make_box(0.0, 0.0), 0.15, gt_range
        )
        totals = new_totals()
        update_totals(totals, positive)
        update_totals(totals, negative)
        summary = finalize_summary(totals, {})

        self.assertEqual(len(positive["reliable_rows"]), 1)
        self.assertEqual(positive["reliable_rows"][0]["outcome"], "positive")
        self.assertEqual(len(negative["reliable_rows"]), 1)
        self.assertEqual(negative["reliable_rows"][0]["outcome"], "negative")
        self.assertEqual(summary["positive_replacements"], 1)
        self.assertEqual(summary["negative_replacements"], 1)
        self.assertGreater(
            summary["mean_positive_actual_gt_bev_iou_delta"], 0.0
        )
        self.assertLess(
            summary["mean_negative_actual_gt_bev_iou_delta"], 0.0
        )
        self.assertEqual(summary["quality_ordering_accuracy"], 0.5)

    def test_diagnostic_gt_coordinates_empty_and_unpaired_cases(self):
        exact_box = make_box(3.0, -2.0)
        boxes = torch.cat([exact_box, make_box(30.0, 30.0)])
        actual_iou, gt_index = best_gt_bev_matches(boxes, exact_box)
        self.assertTrue(np.isclose(actual_iou[0], 1.0))
        self.assertTrue(np.isclose(actual_iou[1], 0.0))
        self.assertEqual(gt_index.tolist(), [0, 0])

        gt_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]
        empty = analyze_frame(0, None, exact_box, 0.15, gt_range)
        no_gt_candidates = self.make_diagnostic_candidates(
            torch.cat([make_box(0.0, 0.0), make_box(0.4, 0.0)]),
            torch.tensor([0.9, 0.8]),
            torch.tensor([0.1, 1.0]),
        )
        no_gt = analyze_frame(
            1,
            no_gt_candidates,
            torch.empty((0, 8, 3)),
            0.15,
            gt_range,
        )

        self.assertEqual(empty["candidate_count"], 0)
        self.assertTrue(no_gt["changed"])
        self.assertEqual(len(no_gt["reliable_rows"]), 0)
        self.assertEqual(no_gt["background_exclusive_count"], 2)


if __name__ == "__main__":
    unittest.main()
