"""SNN semantics, Oracle alignment, full-size gradients and pipeline tests.

Run: python -m unittest tests.test_spiking_value_predictor -v
Optional upstream parity: set SDTV3_SOURCE to Model_Base/models.py.
"""

import ast
import copy
import csv
import io
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from torch import nn
import yaml

from opencood.data_utils.value_predictor_dataset import ValuePredictorDataset
from opencood.models.comm_modules.patch_utils import patch_id_to_slice
from opencood.models.comm_modules.spiking_value_blocks import (
    MultiSpike, SignedInputCurrent, SingletonSafeBatchNorm1d,
    SpikeConvBlock, SpikePatchAttention, SpikeTransformerBlock,
)
from opencood.models.comm_modules.spiking_value_predictor import SpikingCounterfactualValuePredictor
from opencood.models.comm_modules.value_predictor import CounterfactualValuePredictor
from opencood.models.comm_modules.value_predictor_factory import (
    build_value_predictor, validate_value_predictor_checkpoint,
)
from opencood.models.comm_modules.value_prediction_loss_topk_masked import TopKHardValuePredictionLoss
from opencood.tools.align_value_predictor_oracle_labels import align_patch_utility
from opencood.tools.value_predictor_geometry import geometry_valid_mask


ROOT = Path(__file__).resolve().parents[1]


def small_config(**updates):
    # Only unit tests use reduced widths; full-size tests load the shipped YAML.
    config = dict(
        architecture="sdtv3_snn", in_channels=4, embed_dim=16,
        meta_hidden_dim=16, meta_embed_dim=8, coord_embed_dim=8,
        transformer_layers=2, transformer_heads=4, value_hidden_dim=8,
    )
    config.update(updates)
    return config


def metadata(device="cpu"):
    return torch.tensor([[40., -16., 0.8, 45.], [-20., 12., -1.2, 60.]], device=device)


class ScaleBranch(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return self.factor * x


class SpikingValuePredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.old_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(2026)

    def test_quantization_and_inclusive_surrogate_gradient(self):
        x = torch.tensor([-1., 0., .49, .51, 1.4, 2.6, 4., 5.], requires_grad=True)
        y = MultiSpike()(x)
        torch.testing.assert_close(y, torch.tensor([0., 0., 0., .25, .25, .75, 1., 1.]))
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([0., .25, .25, .25, .25, .25, .25, 0.]))

    def test_signed_metadata_and_coordinates_keep_polarity(self):
        encoder = SignedInputCurrent([10., 1.])
        x = torch.tensor([[[5.], [-.75]]])
        positive = MultiSpike()(encoder(x))
        negative = MultiSpike()(encoder(-x))
        torch.testing.assert_close(positive.flatten(), torch.tensor([.5, 0., 0., .75]))
        torch.testing.assert_close(negative, torch.cat((positive[:, 2:], positive[:, :2]), 1))

    def test_single_metadata_observation_preserves_running_statistics(self):
        bn = SingletonSafeBatchNorm1d(4).train()
        before = bn.running_var.clone()
        x = torch.randn(1, 4, 1, requires_grad=True)
        bn(x).square().sum().backward()
        torch.testing.assert_close(bn.running_var, before)
        self.assertEqual(int(bn.num_batches_tracked), 0)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(bn.weight.grad.abs().sum()), 0.)
        bn(torch.randn(2, 4, 1))
        self.assertEqual(int(bn.num_batches_tracked), 1)

    def test_residuals_are_sequential_and_preserve_unquantized_identity(self):
        x = torch.tensor([-1.3, .17, .74, 4.9]).reshape(1, 4, 1, 1)
        for block, names, expected_scale in (
            (SpikeConvBlock(4), ("sepconv", "channel_conv"), 6),
            (SpikeTransformerBlock(4), ("sepconv", "attention", "channel_mlp"), 24),
        ):
            for name in names:
                setattr(block, name, ScaleBranch(0))
            torch.testing.assert_close(block(x), x, rtol=0, atol=0)
            for factor, name in enumerate(names, 1):
                setattr(block, name, ScaleBranch(factor))
            torch.testing.assert_close(block(x), expected_scale * x)

    def test_patch_ids_and_within_patch_channels_match_oracle(self):
        model = SpikingCounterfactualValuePredictor(small_config())
        for patch_id in (0, 41, 42, 523, 1049):
            x = torch.zeros(1, 4, 50, 126)
            rows, cols = patch_id_to_slice(patch_id, 50, 126, (2, 3))
            values = torch.arange(1, 25.).reshape(1, 4, 2, 3)
            x[:, :, rows, cols] = values
            tokens = model.patchify_feature(x).flatten(2).transpose(1, 2)
            active = tokens.abs().sum(-1).nonzero()[:, 1]
            torch.testing.assert_close(active, torch.tensor([patch_id]))
            torch.testing.assert_close(tokens[0, patch_id], values.flatten())
            row, col = divmod(patch_id, 42)
            expected_xy = torch.tensor([2 * (col + .5) / 42 - 1, 2 * (row + .5) / 25 - 1])
            torch.testing.assert_close(model.patch_coordinates[patch_id], expected_xy)

    def test_no_extra_encoder_skip_and_all_weighted_inputs_are_spikes(self):
        model = SpikingCounterfactualValuePredictor(small_config()).train()
        model.ego_encoder = nn.Identity()
        x = torch.randn(2, 4, 50, 126)
        patch_input = []
        handles = [model.patch_embedding.register_forward_pre_hook(
            lambda _module, args: patch_input.append(args[0].detach())
        )]
        calls = []

        def inspect(_module, args):
            spikes = args[0].detach()
            self.assertTrue(bool(((spikes >= 0) & (spikes <= 1)).all()))
            torch.testing.assert_close(spikes * 4, (spikes * 4).round(), rtol=0, atol=0)
            calls.append(1)

        for layer in model.modules():
            self.assertNotIsInstance(layer, (nn.Linear, nn.ReLU, nn.GELU, nn.Softmax,
                                             nn.LayerNorm, nn.MultiheadAttention))
            if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
                handles.append(layer.register_forward_pre_hook(inspect))
        try:
            model(x, metadata())
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(patch_input[0], model.patchify_feature(x))
        self.assertGreater(len(calls), 20)

    def test_attention_reassociation_matches_forward_and_backward(self):
        direct = SpikePatchAttention(16, attention_impl="qk_first").train()
        efficient = copy.deepcopy(direct)
        efficient.attention_impl = "kv_first"
        a = torch.randn(2, 16, 5, 6, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        ya, yb = direct(a), efficient(b)
        torch.testing.assert_close(ya, yb, rtol=1e-5, atol=1e-6)
        probe = torch.randn_like(ya)
        (ya * probe).sum().backward()
        (yb * probe).sum().backward()
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-5)
        for pa, pb in zip(direct.parameters(), efficient.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-4, atol=1e-5)

    def test_complete_default_network_forward_backward_and_monitoring(self):
        config = yaml.safe_load((ROOT / "opencood/hypes_yaml/value_predictor_sdtv3.yaml").read_text())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Exercise actual C=128 cache and the original C=256 public interface.
        for channels in (128, 256):
            with self.subTest(channels=channels, device=device):
                model_config = dict(config["model"], in_channels=channels, collect_spike_stats=True)
                model = build_value_predictor(model_config).to(device).train()
                self.assertEqual(model.embed_dim, 256)
                self.assertEqual(len(model.transformer), 2)
                self.assertEqual(model.transformer[0].attention.heads, 4)
                self.assertEqual(model.transformer[0].attention.value_dim, 1024)
                self.assertEqual(model.transformer[0].channel_mlp.fc1.out_channels, 1024)
                ego = torch.randn(2, channels, 50, 126, device=device, requires_grad=True)
                meta = metadata(device).requires_grad_()
                result = model(ego, meta, return_dict=True)
                self.assertEqual(result["value_map"].shape, (2, 1050))
                self.assertEqual(result["patch_tokens"].shape, (2, 1050, 256))
                self.assertTrue(torch.isfinite(result["value_map"]).all())
                loss = TopKHardValuePredictionLoss(config["loss"])(
                    result["value_map"], torch.randn(2, 1050, device=device),
                    valid_mask=torch.ones(2, 1050, device=device, dtype=torch.bool),
                )["total_loss"]
                loss.backward()
                for group in (model.ego_encoder, model.patch_embedding, model.metadata_encoder,
                              model.feature_fusion, model.coord_encoder, model.value_head):
                    gradients = [p.grad for p in group.parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
                for block in model.transformer:
                    for group in (block.sepconv, block.attention.q_conv,
                                  block.attention.k_conv, block.attention.v_conv,
                                  block.channel_mlp):
                        self.assertGreater(sum(float(p.grad.abs().sum()) for p in group.parameters()), 0.)
                for grad in (ego.grad, meta.grad, model.position_embedding.grad):
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(float(grad.abs().sum()), 0.)
                for index in range(2):
                    stats = result["spike_stats"]["transformer.{}.attention.attn_spike".format(index)]
                    self.assertLess(stats["zero_fraction"], 1.)
                    self.assertLess(stats["saturation_fraction"], 1.)
                del result, loss, model, ego, meta

    def test_singleton_forward_stateless_eval_and_signed_readout(self):
        model = SpikingCounterfactualValuePredictor(small_config()).train()
        x = torch.randn(1, 4, 50, 126)
        y = model(x, metadata()[:1])
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(y).all())
        model.eval()
        with torch.no_grad():
            a = model(x, metadata()[:1])
            model(torch.randn_like(x), -metadata()[:1])
            b = model(x, metadata()[:1])
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            model.value_head[-1].weight.zero_()
            for sign in (-1., 1.):
                model.value_head[-1].bias.fill_(sign)
                torch.testing.assert_close(model(x, metadata()[:1]), torch.full((1, 1050), sign))
        self.assertEqual(set(model(x, metadata()[:1], return_dict=True)), {"value_map", "patch_tokens"})

    def test_metadata_changes_predictions_with_identical_ego_features(self):
        model = SpikingCounterfactualValuePredictor(small_config()).train()
        ego = torch.randn(1, 4, 50, 126).expand(2, -1, -1, -1)
        values = model(ego, metadata())
        self.assertGreater(float((values[0] - values[1]).detach().abs().max()), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA AMP verification requires a GPU")
    def test_attention_accumulates_in_fp32_under_amp(self):
        attention = SpikePatchAttention(256).cuda().eval()
        # Force dense spikes: the unscaled intermediate exceeds float16 range.
        for branch in (attention.q_conv, attention.k_conv, attention.v_conv):
            nn.init.zeros_(branch[0].weight)
            nn.init.constant_(branch[1].bias, 4.)
        observed = []
        handle = attention.attn_spike.register_forward_pre_hook(
            lambda _module, args: observed.append(args[0].detach())
        )
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                result = attention(torch.ones(1, 256, 25, 42, device="cuda"))
        finally:
            handle.remove()
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(observed[0].dtype, torch.float32)
        torch.testing.assert_close(observed[0], torch.full_like(observed[0], 16800.))

    def test_invalid_shape_and_configuration_fail_early(self):
        for updates in ({"num_patch": 1049}, {"embed_dim": 15},
                        {"metadata_scales": [0, 1, 1, 1]}, {"dropout": .1},
                        {"use_output_norm": True}, {"attention_impl": "softmax"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                SpikingCounterfactualValuePredictor(small_config(**updates))
        model = SpikingCounterfactualValuePredictor(small_config())
        with self.assertRaisesRegex(ValueError, "configured num_patch"):
            model(torch.randn(1, 4, 48, 126), metadata()[:1])
        with self.assertRaisesRegex(ValueError, "neighbor_metadata"):
            model(torch.randn(1, 4, 50, 126), torch.randn(1, 5))

    def test_checkpoint_semantics_and_ann_compatibility(self):
        config = small_config()
        model = build_value_predictor(config)
        checkpoint = {"config": {"model": config}, "model_state_dict": model.state_dict()}
        validate_value_predictor_checkpoint(model, checkpoint)
        clone = build_value_predictor(config)
        clone.load_state_dict(checkpoint["model_state_dict"], strict=True)
        clone.eval()
        model.eval()
        x = torch.randn(2, 4, 50, 126)
        with torch.no_grad():
            torch.testing.assert_close(model(x, metadata()), clone(x, metadata()), rtol=0, atol=0)
        for key, value in (("spike_norm", 2.), ("metadata_scales", [1., 1., 1., 1.]),
                           ("attention_impl", "qk_first")):
            mismatch = build_value_predictor(dict(config, **{key: value}))
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                validate_value_predictor_checkpoint(mismatch, checkpoint)
        with self.assertRaisesRegex(ValueError, "SNN checkpoint"):
            validate_value_predictor_checkpoint(model, {"model_state_dict": model.state_dict()})
        ann = build_value_predictor({"in_channels": 4, "embed_dim": 16})
        self.assertIsInstance(ann, CounterfactualValuePredictor)
        validate_value_predictor_checkpoint(ann, {"model_state_dict": ann.state_dict()})
        with self.assertRaises(ValueError):
            validate_value_predictor_checkpoint(ann, checkpoint)

    def test_geometry_coverage_is_label_independent(self):
        for pose in ([0., 0., 0., 0.], [16., -8., .2, 18.]):
            actual = geometry_valid_mask(pose)
            for labels in (torch.randn(1050), torch.ones(1050) * 99):
                _, expected, _ = align_patch_utility(labels, pose, 50, 126)
                torch.testing.assert_close(actual, expected)
        self.assertTrue(geometry_valid_mask([0., 0., 0., 0.]).all())

    def test_training_checkpoint_and_evaluation_entry_points(self):
        from opencood.tools.train_value_predictor_topk import main as train
        from opencood.tools.eval_value_predictor_aligned import main as evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = root / "features"
            features.mkdir()
            labels = []
            for index, sample_id in enumerate((7, 9)):
                torch.save({"sample_id": sample_id, "ego_feature": torch.randn(4, 50, 126),
                            "neighbor_metadata": metadata()[index]}, features / ("sample_%d.pt" % sample_id))
                labels.append({"sample_id": sample_id, "utility": torch.linspace(-1., 1., 1050),
                               "valid_mask": torch.arange(1050) % 3 != 0,
                               "feature_size": [50, 126], "patch_size": [2, 3]})
            label_path = root / "labels.pt"
            torch.save(labels, label_path)
            data = ValuePredictorDataset(str(features), str(label_path), in_channels=4,
                                         require_valid_mask=True)
            torch.testing.assert_close(data[0]["valid_mask"], labels[0]["valid_mask"])
            self.assertEqual(data[0]["sample_id"], 7)
            config = {
                "model": small_config(),
                "dataset": {"feature_dir": str(features), "utility_path": str(label_path),
                            "require_valid_mask": True},
                "evaluation": {"feature_dir": str(features), "utility_path": str(label_path)},
                "training": {"epochs": 2, "batch_size": 2, "lr": .001, "device": "cpu",
                             "num_workers": 0, "pin_memory": False, "grad_clip_norm": 1.,
                             "checkpoint_dir": str(root / "checkpoints")},
                "loss": {"topk": 10, "hard_negative_k": 32},
            }
            cfg = root / "config.yaml"
            cfg.write_text(yaml.safe_dump(config))
            with patch("sys.argv", ["train", "--config", str(cfg)]), redirect_stdout(io.StringIO()):
                train()
            ckpt = root / "checkpoints/value_predictor_latest.pt"
            saved = torch.load(ckpt, map_location="cpu")
            self.assertEqual(saved["epoch"], 2)
            self.assertTrue(saved["optimizer_state_dict"]["state"])
            earlier = torch.load(root / "checkpoints/value_predictor_epoch_001.pt", map_location="cpu")
            self.assertFalse(torch.equal(saved["model_state_dict"]["value_head.4.weight"],
                                         earlier["model_state_dict"]["value_head.4.weight"]))
            csv_path = root / "eval.csv"
            with patch("sys.argv", ["evaluate", "--config", str(cfg), "--checkpoint", str(ckpt),
                                    "--output_csv", str(csv_path)]), redirect_stdout(io.StringIO()):
                evaluate()
            with csv_path.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(int(row["samples"]), 2)
            self.assertGreater(float(row["mean_prediction_std"]), 0.)
            # Missing masks must fail instead of silently treating padding as labels.
            for record in labels:
                record.pop("valid_mask")
            torch.save(labels, label_path)
            with self.assertRaisesRegex(ValueError, "Missing aligned valid_mask"):
                ValuePredictorDataset(str(features), str(label_path), in_channels=4,
                                      require_valid_mask=True)

    def test_upstream_blocks_match_with_identical_parameters(self):
        reference = Path(os.environ.get("SDTV3_SOURCE", str(
            ROOT.parent / "Spike-Driven-Transformer-V3-main/SDT_V3/Classification/Model_Base/models.py"
        )))
        if not reference.is_file():
            self.skipTest("Set SDTV3_SOURCE for optional official-source equivalence test")
        names = {"Quant", "MultiSpike", "SepConv_Spike", "MS_ConvBlock_spike_SepConv",
                 "MS_MLP", "MS_Attention_linear", "MS_Block_Spike_SepConv"}
        tree = ast.parse(reference.read_text(encoding="utf-8"))
        # Import only the unchanged class definitions; avoid timm/torchinfo dependencies.
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        namespace = {"torch": torch, "nn": nn}
        exec(compile(tree, str(reference), "exec"), namespace)
        pairs = [
            (SpikeConvBlock(16), namespace["MS_ConvBlock_spike_SepConv"](16, mlp_ratio=4)),
            (SpikeTransformerBlock(16), namespace["MS_Block_Spike_SepConv"](16, 4, mlp_ratio=4)),
        ]
        for candidate, upstream in pairs:
            if isinstance(candidate, SpikeConvBlock):
                upstream.Conv.load_state_dict(candidate.sepconv.state_dict())
                for source, target in ((candidate.channel_conv[1][0], upstream.conv1),
                                       (candidate.channel_conv[1][1], upstream.bn1),
                                       (candidate.channel_conv[3][0], upstream.conv2),
                                       (candidate.channel_conv[3][1], upstream.bn2)):
                    target.load_state_dict(source.state_dict())
            else:
                upstream.conv.load_state_dict(candidate.sepconv.state_dict())
                upstream.attn.load_state_dict(candidate.attention.state_dict())
                for source, target in ((candidate.channel_mlp.fc1, upstream.mlp.fc1_conv),
                                       (candidate.channel_mlp.bn1, upstream.mlp.fc1_bn),
                                       (candidate.channel_mlp.fc2, upstream.mlp.fc2_conv),
                                       (candidate.channel_mlp.bn2, upstream.mlp.fc2_bn)):
                    target.load_state_dict(source.state_dict())
            a = torch.randn(2, 16, 5, 6, requires_grad=True)
            b = a.detach().clone().requires_grad_()
            expected, actual = upstream(b), candidate(a)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            probe = torch.randn_like(actual)
            (actual * probe).sum().backward()
            (expected * probe).sum().backward()
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
