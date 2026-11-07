"""Run with python -m unittest discover -s tests -p test_e3dsnn_spike_residual.py."""

import unittest

import torch
from torch import nn

from opencood.models.sub_modules.e3dsnn_multiscale_enhancement_neck import (
    MultiScaleEnhancementNeck,
)
from opencood.models.sub_modules.spike_residual_conv import (
    MultiSpikeNorm4, SpikeResidualConvBlock,
)
from test_e3dsnn_multiscale_neck import make_dense_sparse


class SpikeResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_sdtv3_quantization_and_surrogate_boundaries(self):
        x = torch.tensor([-1., 0., .49, .5, .51, 1.5, 2.5, 3.5, 4., 5.],
                         requires_grad=True)
        spike = MultiSpikeNorm4()
        y = spike(x)
        # SDTv3 uses torch.round (ties to even), not floor(x + 0.5).
        expected = torch.tensor([0., 0., 0., 0., .25, .5, .5, 1., 1., 1.])
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        y.sum().backward()
        torch.testing.assert_close(
            x.grad, torch.tensor([0., .25, .25, .25, .25, .25, .25, .25, .25, 0.]),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(spike.eval()(x.detach()), expected, rtol=0, atol=0)

    def test_projected_residual_shapes_and_exact_computation(self):
        for batch in (1, 2):
            for height, width in ((20, 28), (21, 29)):
                for cin, cout, stride in ((4, 4, 2), (4, 8, 1), (4, 4, 1)):
                    with self.subTest(batch=batch, size=(height, width), spec=(cin, cout, stride)):
                        block = SpikeResidualConvBlock(cin, cout, stride).eval()
                        x = torch.randn(batch, cin, height, width)
                        expected = block.bn(block.conv(block.spike(x))) + block.shortcut(x)
                        actual = block(x)
                        self.assertEqual(tuple(actual.shape), (
                            batch, cout, (height + stride - 1) // stride,
                            (width + stride - 1) // stride,
                        ))
                        torch.testing.assert_close(actual, expected)
                        self.assertFalse(any(isinstance(m, nn.ReLU) for m in block.modules()))

    def test_shortcut_uses_unquantized_input_and_preserves_negative_values(self):
        block = SpikeResidualConvBlock(1, 2).eval()
        with torch.no_grad():
            block.conv.weight.zero_()
            block.shortcut[0].weight.fill_(1.)
        x = torch.full((1, 1, 3, 3), -2.)
        actual = block(x)
        self.assertTrue((actual < 0).all())
        torch.testing.assert_close(actual, block.shortcut(x))

    def test_three_blocks_backward_cpu(self):
        self._check_backward(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_three_blocks_backward_cuda(self):
        self._check_backward(torch.device("cuda"))

    def _check_backward(self, device):
        blocks = nn.Sequential(
            SpikeResidualConvBlock(4, 4, 2),
            SpikeResidualConvBlock(4, 4, 2),
            SpikeResidualConvBlock(4, 8, 1),
        ).to(device).train()
        x = torch.rand(2, 4, 21, 29, device=device, requires_grad=True)
        output = blocks(x)
        (output * torch.randn_like(output)).mean().backward()
        for name, parameter in blocks.named_parameters():
            with self.subTest(parameter=name, device=device):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(x.grad).all())

    def _neck(self, alpha=.1):
        return MultiScaleEnhancementNeck([4, 6, 8], [3, 2, 1], 12, 5, alpha)

    def test_legacy_ann_checkpoint_preserves_every_existing_weight(self):
        legacy = self._neck()
        for name, cin, cout, stride in (
            ("downsample1", 5, 5, 2), ("downsample2", 5, 5, 2),
            ("output_conv", 5, 12, 1),
        ):
            layers = [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                      nn.BatchNorm2d(cout, eps=1e-3, momentum=.01)]
            if name != "output_conv":
                layers.append(nn.ReLU(inplace=True))
            setattr(legacy, name, nn.Sequential(*layers))
        old_state = legacy.state_dict()
        new = self._neck()
        result = new.load_state_dict(old_state, strict=False)
        self.assertEqual(result.unexpected_keys, [])
        expected_missing = set(new.state_dict()) - set(old_state)
        # BatchNorm supplies a zero num_batches_tracked buffer on legacy loads.
        self.assertEqual(set(result.missing_keys), {
            key for key in expected_missing if not key.endswith("num_batches_tracked")
        })
        self.assertTrue(expected_missing)
        self.assertTrue(all(".shortcut." in key for key in expected_missing))
        for key, value in old_state.items():
            torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
        # New SNN checkpoint must be strictly reloadable with no missing keys.
        self._neck().load_state_dict(new.state_dict(), strict=True)

    def test_neck_outer_residual_and_full_gradient_path(self):
        _, s2 = make_dense_sparse(2, 4, 3, 20, 28)
        _, s3 = make_dense_sparse(2, 6, 2, 10, 14)
        _, s4 = make_dense_sparse(2, 8, 1, 5, 7)
        original = torch.randn(2, 12, 5, 7)
        neck = self._neck(alpha=0).eval()
        enhanced, debug = neck(s2, s3, s4, original, return_debug=True)
        torch.testing.assert_close(enhanced, original, rtol=0, atol=0)
        self.assertEqual(debug["f_ms"], tuple(original.shape))
        with torch.no_grad():
            neck.alpha.fill_(.1)
        neck.train()
        enhanced, _ = neck(s2, s3, s4, original)
        (enhanced * torch.randn_like(enhanced)).mean().backward()
        for name, parameter in neck.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name in ("downsample1", "downsample2", "output_conv"):
            self.assertGreater(getattr(neck, name).conv.weight.grad.abs().sum().item(), 0.)
            self.assertGreater(getattr(neck, name).shortcut[0].weight.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
