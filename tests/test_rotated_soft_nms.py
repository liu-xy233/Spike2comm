"""Real rotated geometry, pre-change golden outputs, and integration tests."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

# Reuse the existing optional-dependency bootstrap and synthetic input helpers.
# Do not import its TestCase into this module (unittest would collect it twice).
import test_confidence_aware_late_fusion as legacy
import numpy as np
import shapely
import torch

from opencood.data_utils.post_processor import confidence_aware_late_fusion as fusion
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.utils import box_utils


make_box = legacy.make_box


class RotatedSoftNMSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not getattr(shapely, '__version__', None):
            raise RuntimeError('Soft-NMS tests require real Shapely, not the rectangle stub')

    def make_fuser(self, **kwargs):
        params = dict(score_threshold=0.4, collaborator_score_weight=0.7,
                      base_score_threshold=0.2, nms_threshold=0.15)
        params.update(kwargs)
        return fusion.ConfidenceAwareLateFusion(**params)

    def assert_pair_equal(self, actual, expected):
        for left, right in zip(actual, expected):
            self.assertTrue(torch.equal(left, right))

    def test_default_hard_matches_pre_change_golden_outputs(self):
        path = Path(__file__).parent / 'fixtures/confidence_aware_hard_nms_baseline.json'
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(snapshot['source_sha256'],
                         '80DAA000E01C9D469DE8AC9C8B723A37C189F710C2F518B6956017B27B946F06')
        # Tied-score Hard NMS ordering belongs to the current NumPy runtime.
        # Cases 1 and 3 are checked against the actual old module below, not
        # against fixed coordinates captured in a different environment.
        for case_index in (0, 2):
            case = snapshot['cases'][case_index]
            with self.subTest(kwargs=case['kwargs'], scores=case['ego_scores']):
                args = [torch.tensor(case['ego_boxes']), torch.tensor(case['ego_scores']),
                        [(torch.tensor(case['collaborator_boxes']),
                          torch.tensor(case['raw_scores']),
                          torch.tensor(case['ranking_scores']))]]
                expected = (torch.tensor(case['expected_boxes']),
                            torch.tensor(case['expected_scores']))
                with mock.patch.object(fusion, 'rotated_soft_nms',
                                       side_effect=AssertionError('Hard must not call Soft-NMS')):
                    self.assert_pair_equal(fusion.ConfidenceAwareLateFusion(**case['kwargs'])(*args), expected)
                    explicit = fusion.ConfidenceAwareLateFusion(**case['kwargs'], nms_method='hard')
                    self.assert_pair_equal(explicit(*args), expected)

    def test_explicit_hard_matches_default_and_original_call(self):
        boxes = torch.cat([make_box(0, 0), make_box(0.4, 0), make_box(20, 0)])
        scores = torch.tensor([0.9, 0.8, 0.7])
        with mock.patch.object(box_utils, 'nms_rotated', wraps=box_utils.nms_rotated) as hard:
            default = self.make_fuser()(boxes, scores, [])
            explicit = self.make_fuser(nms_method='hard')(boxes, scores, [])
        self.assert_pair_equal(default, explicit)
        self.assertEqual(hard.call_count, 2)
        self.assertTrue(torch.equal(hard.call_args.args[0], boxes))
        self.assertTrue(torch.equal(hard.call_args.args[1], scores))
        self.assertEqual(hard.call_args.args[2], 0.15)

    def test_overlap_lowers_confidence(self):
        boxes = torch.cat([make_box(0, 0), make_box(0, 0)])
        keep, scores = fusion.rotated_soft_nms(boxes, torch.tensor([0.9, 0.8]), 0.0)
        self.assertEqual(keep.tolist(), [0, 1])
        self.assertAlmostEqual(scores[1].item(), 0.8 * math.exp(-2), places=6)

    def test_disjoint_and_touching_scores_unchanged(self):
        for distance in (4.0, 20.0):
            boxes = torch.cat([make_box(0, 0), make_box(distance, 0)])
            original = torch.tensor([0.8, 0.9])
            keep, scores = fusion.rotated_soft_nms(boxes, original, 0.2)
            self.assertTrue(torch.equal(scores, original[keep]))

    def test_iou_formula_and_monotonicity(self):
        decayed = []
        for distance in (0.0, 1.0, 2.0, 4.0):
            boxes = torch.cat([make_box(0, 0), make_box(distance, 0)])
            _, scores = fusion.rotated_soft_nms(boxes, torch.tensor([0.9, 0.8]), 0.0)
            iou = (4.0 - distance) / (4.0 + distance)
            self.assertAlmostEqual(scores[1].item(), 0.8 * math.exp(-iou**2 / 0.5), places=6)
            decayed.append(scores[1].item())
        self.assertTrue(all(a < b for a, b in zip(decayed, decayed[1:])))

    def test_sigma_formula_and_monotonicity(self):
        boxes = torch.cat([make_box(0, 0), make_box(1, 0)])
        decayed = []
        for sigma in (0.1, 0.5, 2.0):
            _, scores = fusion.rotated_soft_nms(boxes, torch.tensor([0.9, 0.8]), 0.0, sigma)
            self.assertAlmostEqual(scores[1].item(), 0.8 * math.exp(-0.6**2 / sigma), places=6)
            decayed.append(scores[1].item())
        self.assertTrue(all(a < b for a, b in zip(decayed, decayed[1:])))

    def test_true_rotated_iou_not_identical_axis_aligned_bounds(self):
        boxes = torch.cat([make_box(0, 0, yaw=math.pi / 4),
                           make_box(0, 0, yaw=-math.pi / 4)])
        # Perpendicular 4x2 rectangles: intersection=4, union=12, IoU=1/3.
        self.assertTrue(torch.allclose(boxes[0, :, :2].amin(0), boxes[1, :, :2].amin(0)))
        self.assertTrue(torch.allclose(boxes[0, :, :2].amax(0), boxes[1, :, :2].amax(0)))
        _, scores = fusion.rotated_soft_nms(boxes, torch.tensor([0.9, 0.8]), 0.0)
        self.assertAlmostEqual(scores[1].item(), 0.8 * math.exp(-(1/3)**2 / 0.5), places=6)
        self.assertGreater(scores[1].item(), 0.8 * math.exp(-2))

    def test_empty_input(self):
        keep, scores = fusion.rotated_soft_nms(torch.empty(0, 8, 3), torch.empty(0), 0.2)
        self.assertEqual(keep.shape, (0,))
        self.assertEqual(keep.dtype, torch.long)
        self.assertEqual(scores.shape, (0,))

    def test_single_box_and_strict_threshold_boundary(self):
        for score in (0.1, 0.2, 0.8):
            original = torch.tensor([score])
            keep, scores = fusion.rotated_soft_nms(make_box(0, 0), original, 0.2)
            self.assertEqual(keep.tolist(), [0] if score > 0.2 else [])
            self.assertTrue(torch.equal(scores, original[keep]))

    def test_decay_filters_below_threshold(self):
        boxes = torch.cat([make_box(0, 0), make_box(0, 0)])
        keep, scores = fusion.rotated_soft_nms(boxes, torch.tensor([0.9, 0.8]), 0.2)
        self.assertEqual(keep.tolist(), [0])
        self.assertTrue(torch.equal(scores, torch.tensor([0.9])))

    def test_reselection_cumulative_decay_and_original_index_alignment(self):
        boxes = torch.cat([make_box(1, 0), make_box(2, 0), make_box(0, 0), make_box(20, 0)])
        original = torch.tensor([0.8, 0.7, 0.9, 0.65])
        keep, scores = fusion.rotated_soft_nms(boxes, original, 0.1)
        self.assertEqual(keep.tolist(), [2, 3, 1, 0])
        self.assertTrue(torch.allclose(boxes[keep].mean(1)[:, 0], torch.tensor([0., 20., 2., 1.])))
        expected = torch.tensor([0.9, 0.65, 0.7 * math.exp(-(1/3)**2 / 0.5),
                                 0.8 * math.exp(-2 * 0.6**2 / 0.5)])
        self.assertTrue(torch.allclose(scores, expected))
        fused = self.make_fuser(nms_method='soft', base_score_threshold=0.1)(boxes, original, [])
        self.assert_pair_equal(fused, (boxes[keep], scores))

    def test_ties_are_repeatable_and_choose_lowest_original_index(self):
        boxes = torch.cat([make_box(0, 0), make_box(20, 0), make_box(0, 0)])
        scores = torch.full((3,), 0.8)
        expected = fusion.rotated_soft_nms(boxes, scores, 0.0)
        self.assertEqual(expected[0].tolist(), [0, 1, 2])
        for _ in range(5):
            self.assert_pair_equal(fusion.rotated_soft_nms(boxes, scores, 0.0), expected)

    def test_inputs_not_mutated_and_dtype_device_preserved(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float16, torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    boxes = torch.cat([make_box(0, 0), make_box(1, 0)]).to(device=device, dtype=dtype)
                    scores = torch.tensor([0.9, 0.8], device=device, dtype=dtype)
                    old_boxes, old_scores = boxes.clone(), scores.clone()
                    keep, decayed = fusion.rotated_soft_nms(boxes, scores, 0.0)
                    self.assert_pair_equal((boxes, scores), (old_boxes, old_scores))
                    self.assertEqual(keep.device, scores.device)
                    self.assertEqual(decayed.device, scores.device)
                    self.assertEqual(decayed.dtype, dtype)
                    self.assertEqual(keep.dtype, torch.long)

    def test_tiny_sigma_handles_zero_iou_without_nan(self):
        boxes = torch.cat([make_box(0, 0), make_box(0, 0), make_box(20, 0)])
        keep, scores = fusion.rotated_soft_nms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.0, 1e-300)
        self.assertEqual(keep.tolist(), [0, 2])
        self.assertTrue(torch.equal(scores, torch.tensor([0.9, 0.7])))

    def test_invalid_sigma_threshold_and_method(self):
        boxes, scores = make_box(0, 0), torch.tensor([0.9])
        for sigma in (0, -0.5, float('nan'), float('inf')):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                fusion.rotated_soft_nms(boxes, scores, 0.2, sigma)
            with self.assertRaises(ValueError):
                self.make_fuser(soft_nms_sigma=sigma)
        for threshold in (-0.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                fusion.rotated_soft_nms(boxes, scores, threshold)
        with self.assertRaises(ValueError):
            self.make_fuser(nms_method='unknown')

    def test_invalid_shapes_nonfinite_and_degenerate_geometry(self):
        boxes, scores = make_box(0, 0), torch.tensor([0.9])
        cases = [(boxes[0], scores), (boxes, scores[:, None]),
                 (boxes, torch.ones(2)), (boxes, torch.ones(1, dtype=torch.long)),
                 (boxes * float('nan'), scores), (boxes, scores * float('inf')),
                 (torch.zeros_like(boxes), scores)]
        for invalid_boxes, invalid_scores in cases:
            with self.subTest(shape=invalid_boxes.shape), self.assertRaises(ValueError):
                fusion.rotated_soft_nms(invalid_boxes, invalid_scores, 0.2)

    def test_soft_fuser_rbv_keeps_original_weights_and_decayed_output_scores(self):
        boxes = torch.cat([make_box(0, 0), make_box(0.4, 0)])
        scores = torch.tensor([0.9, 0.8])
        for sigma in (0.1, 0.5):
            keep, expected_scores = fusion.rotated_soft_nms(boxes, scores, 0.2, sigma)
            for enabled in (False, True):
                with self.subTest(sigma=sigma, rbv=enabled):
                    with mock.patch.object(fusion, 'rotated_box_voting', wraps=fusion.rotated_box_voting) as rbv:
                        actual_boxes, actual_scores = self.make_fuser(
                            nms_method='soft', soft_nms_sigma=sigma,
                            enable_rotated_box_voting=enabled)(boxes, scores, [])
                    self.assertTrue(torch.equal(actual_scores, expected_scores))
                    if enabled:
                        rbv.assert_called_once()
                        self.assertTrue(torch.equal(rbv.call_args.args[2], scores))
                        expected_boxes = fusion.rotated_box_voting(boxes[keep], boxes, scores)
                    else:
                        rbv.assert_not_called()
                        expected_boxes = boxes[keep]
                    self.assertTrue(torch.equal(actual_boxes, expected_boxes))
                    self.assertTrue(torch.allclose(
                        actual_boxes - actual_boxes.mean(1, keepdim=True),
                        boxes[keep] - boxes[keep].mean(1, keepdim=True), atol=1e-6))

    def test_soft_fuser_no_candidates_returns_none(self):
        fuser = self.make_fuser(nms_method='soft')
        self.assertEqual(fuser(None, None, []), (None, None))
        self.assertEqual(fuser(make_box(0, 0), torch.tensor([0.2]), []), (None, None))
        self.assertEqual(fuser(make_box(0, 0, z=20), torch.tensor([0.9]), []), (None, None))

    def test_soft_fuser_collaborator_filter_weight_and_quality_precede_decay(self):
        ego, ego_scores = make_box(0, 0), torch.tensor([0.9])
        candidates = torch.cat([make_box(1, 0), make_box(20, 0), make_box(30, 0)])
        raw, ranking = torch.tensor([0.8, 0.39, 0.8]), torch.tensor([0.7, 0.99, 0.1])
        boxes, scores = self.make_fuser(nms_method='soft')(
            ego, ego_scores, [(candidates, raw, ranking)])
        self.assertTrue(torch.equal(boxes, torch.cat([ego, candidates[:1]])))
        self.assertTrue(torch.allclose(scores, torch.tensor([0.9, 0.7 * 0.7 * math.exp(-0.6**2 / 0.5)])))

    @staticmethod
    def postprocessor_inputs(**config):
        cfg = dict(enabled=True, score_threshold=0.4, collaborator_score_weight=0.7)
        cfg.update(config)
        params = dict(anchor_args={'num': 1}, order='hwl',
                      target_args={'score_threshold': 0.2}, nms_thresh=0.15,
                      gt_range=[-50, -50, -5, 50, 50, 5],
                      confidence_aware_late_fusion=cfg)
        pairwise = torch.eye(4).repeat(1, 3, 3, 1, 1)
        pairwise[0, 1, 0, 0, 3] = 0.4
        pairwise[0, 2, 0, 0, 3] = 60.0
        data = {'ego': dict(transformation_matrix=torch.eye(4),
                            pairwise_t_matrix=pairwise, record_len=torch.tensor([3]),
                            anchor_box=torch.tensor([[[[0., 0., 0., 1.5, 2., 4., 0.]]]]))}
        output = {'ego': dict(psm=torch.tensor([[[[2.1972246]]]]), rm=torch.zeros(1, 7, 1, 1),
                              psm_single=torch.tensor([0., 1.3862944, 1.3862944]).view(3, 1, 1, 1),
                              rm_single=torch.zeros(3, 7, 1, 1))}
        return VoxelPostprocessor(params, train=False), data, output

    def test_postprocessor_passes_soft_parameters_then_range_filters(self):
        postprocessor, data, output = self.postprocessor_inputs(nms_method='soft', soft_nms_sigma=2.0)
        calls = []
        original_soft = fusion.rotated_soft_nms
        original_range = box_utils.get_mask_for_boxes_within_range_torch

        def tracked_soft(*args, **kwargs):
            calls.append('soft')
            return original_soft(*args, **kwargs)

        def tracked_range(*args, **kwargs):
            calls.append('range')
            return original_range(*args, **kwargs)

        with mock.patch.object(fusion, 'rotated_soft_nms', side_effect=tracked_soft) as soft:
            with mock.patch.object(box_utils, 'nms_rotated', side_effect=AssertionError('No hard NMS')):
                with mock.patch.object(box_utils, 'get_mask_for_boxes_within_range_torch', side_effect=tracked_range):
                    boxes, scores = postprocessor.post_process(data, output)
        self.assertEqual(calls, ['soft', 'range'])
        self.assertEqual(soft.call_args.kwargs, dict(score_threshold=0.2, sigma=2.0))
        self.assertTrue(torch.allclose(boxes.mean(1)[:, 0], torch.tensor([0., 0.4]), atol=1e-6))
        self.assertTrue(torch.allclose(scores, torch.tensor([0.9, 0.8 * 0.7 * math.exp(-(3.6/4.4)**2 / 2.0)])))

    def test_postprocessor_explicit_hard_matches_default(self):
        default, data, output = self.postprocessor_inputs()
        explicit, _, _ = self.postprocessor_inputs(nms_method='hard')
        self.assert_pair_equal(default.post_process(data, output), explicit.post_process(data, output))

    def test_disabled_late_fusion_keeps_ordinary_hard_path(self):
        postprocessor, data, output = self.postprocessor_inputs(enabled=False, nms_method='soft')
        with mock.patch.object(fusion, 'rotated_soft_nms', side_effect=AssertionError('Disabled path')):
            with mock.patch.object(box_utils, 'nms_rotated', wraps=box_utils.nms_rotated) as hard:
                postprocessor.post_process(data, output)
        hard.assert_called_once()


class SameEnvironmentHardRegressionTest(unittest.TestCase):
    """Execute both real source files in one process without rewriting either."""

    @classmethod
    def setUpClass(cls):
        if not getattr(shapely, '__version__', None):
            raise RuntimeError('Hard regression requires real Shapely')
        root = Path(__file__).resolve().parents[1]
        source = Path('opencood/data_utils/post_processor/confidence_aware_late_fusion.py')
        old_path = (root / 'codex_backups/gaussian_rotated_soft_nms_20261031_171804'
                    / 'original' / source)
        fixture_path = root / 'tests/fixtures/confidence_aware_hard_nms_baseline.json'
        snapshot = json.loads(fixture_path.read_text(encoding='utf-8'))
        if hashlib.sha256(old_path.read_bytes()).hexdigest().upper() != snapshot['source_sha256']:
            raise AssertionError('Original source SHA256 does not match fixture provenance')
        cls.cases = snapshot['cases']
        if len(cls.cases) != 4:
            raise AssertionError('Expected all four original regression cases')

        def load_source(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            # Register separate names while leaving the canonical module alone.
            if name in sys.modules:
                raise AssertionError('Regression module name already in use: ' + name)
            sys.modules[name] = module
            cls.addClassCleanup(sys.modules.pop, name, None)
            spec.loader.exec_module(module)
            return module

        cls.old = load_source('_hard_regression_before_soft_nms', old_path)
        cls.new = load_source('_hard_regression_after_soft_nms', root / source)

    @staticmethod
    def case_inputs(case):
        # One set of tensor objects is shared by all three calls. Each call
        # must leave every tensor identical to an independent pre-call clone.
        tensors = tuple(torch.tensor(case[key], dtype=torch.float32) for key in (
            'ego_boxes', 'ego_scores', 'collaborator_boxes',
            'raw_scores', 'ranking_scores'))
        return tensors, (tensors[0], tensors[1], [tensors[2:]])

    def test_loaded_versions_share_runtime_dependencies(self):
        self.assertIsNot(self.old, self.new)
        self.assertNotEqual(self.old.__name__, self.new.__name__)
        self.assertNotEqual(Path(self.old.__file__), Path(self.new.__file__))
        self.assertIs(sys.modules[fusion.__name__], fusion)
        for module in (self.old, self.new):
            self.assertIs(module.torch, torch)
            self.assertIs(module.box_utils, box_utils)
            self.assertIs(module.box_utils.nms_rotated, box_utils.nms_rotated)
            self.assertIs(module.common_utils, fusion.common_utils)
            self.assertIs(module.box_utils.np, np)
            self.assertIs(module.common_utils.np, np)
            self.assertIs(module.common_utils.torch, torch)
            self.assertIs(module.common_utils.Polygon, shapely.geometry.Polygon)
        self.assertIs(self.old.box_utils.nms_rotated.__globals__['np'], np)

    def compare_case(self, case_index):
        case = self.cases[case_index]
        self.assertEqual(case['kwargs']['enable_rotated_box_voting'], case_index >= 2)
        self.assertEqual(case['ego_scores'][0] == case['ego_scores'][1], case_index in (1, 3))
        tensors, args = self.case_inputs(case)
        original_tensors = tuple(tensor.clone() for tensor in tensors)
        original_hard = box_utils.nms_rotated
        nms_calls = []

        def record_hard(boxes, scores, threshold):
            # Observe the real function; do not sort, substitute indices, or
            # modify its arguments. This also exposes order before RBV.
            before_boxes, before_scores = boxes.clone(), scores.clone()
            keep = original_hard(boxes, scores, threshold)
            self.assertTrue(torch.equal(boxes, before_boxes))
            self.assertTrue(torch.equal(scores, before_scores))
            nms_calls.append((before_boxes, before_scores, threshold, keep.copy()))
            return keep

        outputs = []
        configurations = ((self.old, {}), (self.new, {}),
                          (self.new, {'nms_method': 'hard'}))
        with mock.patch.object(box_utils, 'nms_rotated', side_effect=record_hard):
            for module, extra in configurations:
                fuser = module.ConfidenceAwareLateFusion(**case['kwargs'], **extra)
                outputs.append(fuser(*args))
                for tensor, original in zip(tensors, original_tensors):
                    self.assertTrue(torch.equal(tensor, original),
                                    'Input tensor modified by ' + module.__name__)

        self.assertEqual(len(nms_calls), 3)
        old_boxes, old_scores = outputs[0]
        self.assertIsNotNone(old_boxes)
        self.assertIsNotNone(old_scores)
        rows = []
        for index, mode in ((1, 'default'), (2, 'explicit_hard')):
            new_boxes, new_scores = outputs[index]
            with self.subTest(case=case_index, mode=mode):
                self.assertEqual(new_boxes.shape, old_boxes.shape)
                self.assertEqual(new_scores.shape, old_scores.shape)
                self.assertEqual(new_boxes.dtype, old_boxes.dtype)
                self.assertEqual(new_scores.dtype, old_scores.dtype)
                self.assertEqual(new_boxes.device, old_boxes.device)
                self.assertEqual(new_scores.device, old_scores.device)
                # Exact, row-by-row equality: no tolerances or sorting before
                # comparison, including tied-score Cases 1 and 3.
                self.assertTrue(torch.equal(new_boxes, old_boxes))
                self.assertTrue(torch.equal(new_scores, old_scores))
                self.assertTrue(torch.equal(nms_calls[index][0], nms_calls[0][0]))
                self.assertTrue(torch.equal(nms_calls[index][1], nms_calls[0][1]))
                self.assertEqual(nms_calls[index][2], nms_calls[0][2])
                self.assertTrue(np.array_equal(nms_calls[index][3], nms_calls[0][3]))
                box_error = (new_boxes.double() - old_boxes.double()).abs().max().item()
                score_error = (new_scores.double() - old_scores.double()).abs().max().item()
                self.assertEqual(box_error, 0.0)
                self.assertEqual(score_error, 0.0)
                rows.append(dict(case=case_index, mode=mode,
                                 old_count=len(old_boxes), new_count=len(new_boxes),
                                 boxes_equal=True, scores_equal=True, order_equal=True,
                                 inputs_unchanged=True, max_box_error=box_error,
                                 max_score_error=score_error,
                                 old_keep=nms_calls[0][3].tolist(),
                                 new_keep=nms_calls[index][3].tolist()))
        return rows

    def test_case_0_rbv_off_matches_old_hard(self):
        self.compare_case(0)

    def test_case_1_rbv_off_tied_scores_matches_old_hard(self):
        self.compare_case(1)

    def test_case_2_rbv_on_matches_old_hard(self):
        self.compare_case(2)

    def test_case_3_rbv_on_tied_scores_matches_old_hard(self):
        self.compare_case(3)

    def test_hard_modes_never_call_soft_nms(self):
        # Cover both the independently loaded current module and the canonical
        # production import; fail immediately if either reaches Soft-NMS.
        with mock.patch.object(self.new, 'rotated_soft_nms',
                               side_effect=AssertionError('Hard called Soft-NMS')) as isolated_soft:
            with mock.patch.object(fusion, 'rotated_soft_nms',
                                   side_effect=AssertionError('Hard called Soft-NMS')) as canonical_soft:
                for case_index, case in enumerate(self.cases):
                    for module in (self.new, fusion):
                        for extra in ({}, {'nms_method': 'hard'}):
                            with self.subTest(case=case_index, module=module.__name__, extra=extra):
                                _, args = self.case_inputs(case)
                                module.ConfidenceAwareLateFusion(**case['kwargs'], **extra)(*args)
                isolated_soft.assert_not_called()
                canonical_soft.assert_not_called()


if __name__ == '__main__':
    unittest.main()
