"""Behavioral checks for ESSA agent fusion; no sparse backbone/data required."""

import copy
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from opencood.models.fuse_modules.sdtv3_attention_fusion import (
    MultiSpike,
    Sdtv3AttentionFusion,
)
from opencood.models.fuse_modules.where2comm_attn import AttenFusion, Where2comm
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.train_utils import load_pretrained_model, load_saved_model


def make_args(multi_scale=False, mode='SDTV3_ESSA', channels=8):
    return {
        'voxel_size': [0.2, 0.2, 0.1], 'downsample_rate': 8,
        'multi_scale': multi_scale, 'layer_nums': [1], 'num_filters': [channels],
        'agg_operator': {
            'mode': mode, 'feature_dim': channels, 'num_heads': 2,
            'v_expansion': 4, 'spike_max': 4, 'spike_norm': 4,
        },
    }


def transforms(batch, agents):
    return torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(batch, agents, agents, 1, 1)


def controlled_fusion(channels=8, heads=2):
    """Non-saturated, nonzero spikes make the semantic assertions meaningful."""
    module = Sdtv3AttentionFusion(channels, heads).eval()
    with torch.no_grad():
        for branch in (module.essa.q_conv, module.essa.k_conv, module.essa.v_conv):
            branch[0].weight.zero_()
            for out_channel in range(branch[0].out_channels):
                branch[0].weight[out_channel, out_channel % channels, 0, 0] = 4
        module.essa.proj_conv[0].weight.fill_(1 / module.essa.value_dim)
    return module


def test_multispike_forward_and_surrogate_boundaries():
    x = torch.tensor([-1., 0., 0.4, 0.6, 2., 4., 5.], requires_grad=True)
    y = MultiSpike()(x)
    torch.testing.assert_close(y, torch.tensor([0., 0., 0., .25, .5, 1., 1.]))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0., .25, .25, .25, .25, .25, 0.]))


def test_attention_matches_independent_per_pixel_reference():
    torch.manual_seed(21)
    module = controlled_fusion()
    x = torch.rand(3, 8, 3, 5) * 2 + 1.3
    attention = module.essa
    spike_x = x.clamp(0, 4).round() / 4
    projections = [branch(spike_x).clamp(0, 4).round() / 4 for branch in (
        attention.q_conv, attention.k_conv, attention.v_conv
    )]
    # Independent loops expose accidental mixing of pixels, channels or agents.
    context = torch.zeros(3, 32, 3, 5)
    for row in range(3):
        for col in range(5):
            for head in range(2):
                q = projections[0][:, head * 4:(head + 1) * 4, row, col]
                k = projections[1][:, head * 4:(head + 1) * 4, row, col]
                v = projections[2][:, head * 16:(head + 1) * 16, row, col]
                context[:, head * 16:(head + 1) * 16, row, col] = (
                    (q @ k.T) @ v
                ) * (2 / (4 ** .5))
    expected = (x + attention.proj_conv(context.clamp(0, 4).round() / 4))[0]
    torch.testing.assert_close(module(x), expected)


@pytest.mark.parametrize('training', [False, True])
def test_masked_values_cannot_change_fusion_even_with_bn_offsets(training):
    torch.manual_seed(22)
    module = controlled_fusion().train(training)
    with torch.no_grad():
        for branch in (module.essa.q_conv, module.essa.k_conv, module.essa.v_conv):
            branch[1].bias.fill_(1.0)
    other = copy.deepcopy(module)
    x = torch.rand(2, 8, 4, 5) * 3
    mask = torch.ones(2, 1, 4, 5, dtype=torch.bool)
    mask[1, :, :, 2:] = False
    corrupt = x.masked_fill(~mask, float('nan'))
    torch.testing.assert_close(module(x, mask), other(corrupt, mask))
    # With entirely masked neighbor, BN bias must still not create neighbor K/V.
    module.eval()
    mask[1] = False
    torch.testing.assert_close(module(x, mask), module(x[:1], mask[:1]))


def test_valid_neighbor_changes_ego_and_other_pixels_do_not_in_eval():
    module = controlled_fusion()
    x = torch.full((2, 8, 3, 5), 2.)
    baseline = module(x)
    changed = x.clone()
    changed[1, :, 1, 2] = 3.3
    output = module(changed)
    assert not torch.allclose(output[:, 1, 2], baseline[:, 1, 2])
    output[:, 1, 2] = baseline[:, 1, 2]
    torch.testing.assert_close(output, baseline)


def test_residual_is_continuous_and_not_quantized():
    module = controlled_fusion()
    with torch.no_grad():
        module.essa.proj_conv[0].weight.zero_()
    x = torch.rand(2, 8, 3, 5) * 10 - 3
    torch.testing.assert_close(module(x), x[0], rtol=0, atol=0)


def test_all_attention_projections_and_input_receive_gradients():
    torch.manual_seed(23)
    module = controlled_fusion()
    x = (torch.rand(2, 8, 4, 5) * .8 + 1.6).requires_grad_()
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    before = module.essa.q_conv[0].weight.detach().clone()
    module(x).square().mean().backward()
    for branch in (module.essa.q_conv, module.essa.k_conv,
                   module.essa.v_conv, module.essa.proj_conv):
        grad = branch[0].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all() and x.grad[1].abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, module.essa.q_conv[0].weight)


def test_agent_permutation_and_single_agent_shape():
    module = controlled_fusion()
    x = torch.rand(3, 8, 4, 5) * 3
    torch.testing.assert_close(module(x), module(x[[0, 2, 1]]))
    assert module(x[:1]).shape == (8, 4, 5)
    with pytest.raises(ValueError, match='divisible'):
        Sdtv3AttentionFusion(10, 3)
    with pytest.raises(ValueError, match='valid_mask'):
        module(x, torch.ones(3, 4, 5))


class OneLevelBackbone(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity()])
        self.deblocks = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )])


@pytest.mark.parametrize('multi_scale', [False, True])
def test_where2comm_override_masks_and_variable_scene_sizes(multi_scale):
    torch.manual_seed(24)
    fusion = Where2comm(make_args(multi_scale)).eval()
    backbone = OneLevelBackbone().eval() if multi_scale else None
    x = torch.rand(3, 8, 4, 5) * 3
    confidence = torch.zeros(3, 2, 4, 5)
    mask = torch.ones(3, 1, 4, 5)
    mask[1, :, :, 2:] = 0
    packed, rate, debug = fusion(
        x, confidence, torch.tensor([2, 1]), transforms(2, 2), backbone,
        override_communication_mask=mask, return_communication_debug=True,
    )
    separate = []
    for start, count in ((0, 2), (2, 1)):
        output, _, _ = fusion(
            x[start:start + count], confidence[start:start + count],
            torch.tensor([count]), transforms(1, count), backbone,
            override_communication_mask=mask[start:start + count],
        )
        separate.append(output)
    torch.testing.assert_close(packed, torch.cat(separate))
    torch.testing.assert_close(rate, torch.tensor(.2))
    assert debug['communication_source'] == 'override'
    torch.testing.assert_close(debug['communication_masks'], mask)


def test_source_mask_warp_matches_bilinear_feature_support():
    fusion = Where2comm(make_args()).eval()
    module = controlled_fusion()
    x = torch.full((2, 8, 4, 5), 2.)
    mask = torch.ones(2, 1, 4, 5)
    mask[1] = 0
    mask[1, :, 1, 2] = 1
    theta = torch.tensor([[[1., 0., 0.], [0., 1., 0.]],
                          [[1., 0., .2], [0., 1., .25]]])
    coverage = warp_affine_simple(mask, theta, (4, 5))
    assert ((coverage > 0) & (coverage < 1)).any()
    warped = warp_affine_simple(x * mask, theta, (4, 5))
    torch.testing.assert_close(
        fusion._fuse_agents(module, x, theta, mask), module(warped, coverage > 0)
    )
    theta[1, 0, 2] = 10
    # Even full communication has invalid cells when the neighbor is out of FOV.
    torch.testing.assert_close(
        fusion._fuse_agents(module, x, theta), module(x[:1])
    )


def test_confidence_selection_and_full_communication_routes():
    args = make_args(True)
    args['communication'] = {'thre': .5}
    fusion = Where2comm(args).eval()
    backbone = OneLevelBackbone().eval()
    x = torch.rand(2, 8, 4, 5)
    confidence = torch.full((2, 2, 4, 5), -10.)
    _, _, debug = fusion(x, confidence, torch.tensor([2]), transforms(1, 2), backbone,
                         return_communication_debug=True)
    assert debug['communication_source'] == 'confidence_threshold'
    assert debug['communication_masks'][0].all()
    assert not debug['communication_masks'][1].any()
    del args['communication']
    fusion = Where2comm(args).eval()
    output, _, debug = fusion(x, confidence, torch.tensor([2]), transforms(1, 2),
                             backbone, return_communication_debug=True)
    assert output.shape == (1, 8, 4, 5)
    assert debug['communication_source'] == 'disabled'


def test_legacy_attention_still_matches_original_formula():
    torch.manual_seed(25)
    x = torch.rand(2, 8, 4, 5)
    fusion = Where2comm(make_args(mode='ATTEN')).eval()
    output, _, _ = fusion(x, torch.zeros(2, 2, 4, 5), torch.tensor([2]), transforms(1, 2))
    tokens = x.flatten(2).permute(2, 0, 1)
    expected = (torch.softmax(tokens @ tokens.transpose(1, 2) / (8 ** .5), -1)
                @ tokens).permute(1, 2, 0).reshape_as(x)[0]
    assert isinstance(fusion.fuse_modules, AttenFusion)
    torch.testing.assert_close(output[0], expected)


def test_formal_config_fusion_neck_and_detection_heads_backward():
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(str(root / 'opencood/hypes_yaml/dair-v2x/dair_where2comm_e3dsnn_sdtv3_essa.yaml'))
    args = config['model']['args']
    fusion = Where2comm(args['fusion_args']).train()
    backbone = OneLevelBackbone(256).train()
    shrink = DownsampleConv(args['shrink_header']).train()
    heads = nn.ModuleList([nn.Conv2d(256, n, 1) for n in (2, 14, 2)])
    x = torch.rand(2, 256, 4, 5, requires_grad=True)
    mask = torch.ones(2, 1, 4, 5)
    mask[1, :, :, 3:] = 0
    y, _, _ = fusion(x, torch.zeros(2, 2, 4, 5), torch.tensor([2]), transforms(1, 2),
                     backbone, override_communication_mask=mask)
    feature = shrink(y)
    predictions = [head(feature) for head in heads]
    assert [p.shape for p in predictions] == [(1, 2, 4, 5), (1, 14, 4, 5), (1, 2, 4, 5)]
    loss = sum(F.mse_loss(p, torch.ones_like(p)) for p in predictions)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    for parameter in fusion.fuse_modules.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


class CheckpointModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.backbone = nn.Conv2d(8, 8, 1)
        self.fusion_net = Where2comm(make_args(mode=mode))


def test_checkpoint_initialization_resume_and_mismatch_rejection(tmp_path):
    baseline = CheckpointModel('ATTEN')
    checkpoint = tmp_path / 'net_epoch1.pth'
    torch.save(baseline.state_dict(), checkpoint)
    model = CheckpointModel('SDTV3_ESSA')
    load_pretrained_model(str(checkpoint), model)
    torch.testing.assert_close(model.backbone.weight, baseline.backbone.weight)
    with pytest.raises(RuntimeError, match='ESSA checkpoint/config mismatch'):
        load_saved_model(str(tmp_path), model)
    torch.save(model.state_dict(), checkpoint)
    reloaded = CheckpointModel('SDTV3_ESSA')
    epoch, _ = load_saved_model(str(tmp_path), reloaded)
    assert epoch == 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[key])
    with pytest.raises(RuntimeError, match='ESSA checkpoint/config mismatch'):
        load_saved_model(str(tmp_path), baseline)
    # Whitelisting ESSA must not hide missing backbone weights.
    broken = dict(baseline.state_dict())
    del broken['backbone.weight']
    torch.save(broken, checkpoint)
    with pytest.raises(RuntimeError, match='outside allowed new modules'):
        load_pretrained_model(str(checkpoint), model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_cuda_autocast_full_bev_forward_backward():
    torch.manual_seed(26)
    module = Sdtv3AttentionFusion(256, 8).cuda().train()
    x = torch.rand(2, 256, 50, 126, device='cuda', requires_grad=True)
    mask = torch.ones(2, 1, 50, 126, device='cuda', dtype=torch.bool)
    mask[1, :, :, 60:] = False
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        y = module(x, mask)
        loss = y.float().square().mean()
    loss.backward()
    assert y.shape == (256, 50, 126)
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
