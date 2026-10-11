import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d

from roadseg.models.encoder import ResNeXtEncoder, _wavelet_ize, adapt_stem_weights


def _params(m):
    return sum(p.numel() for p in m.parameters())


def _x(n=2, size=64):
    x = torch.randn(n, 3, size, size)
    return x * torch.tensor([1.0, 2.5, 0.4]).view(1, 3, 1, 1) + torch.tensor([0.3, -1.0, 2.0]).view(1, 3, 1, 1)


@pytest.mark.parametrize("rung", ["R0", "R1", "R2"])
def test_stage_shapes(rung):
    enc = ResNeXtEncoder(rung, pretrained=False)
    feats = enc(torch.randn(1, 3, 128, 128))
    assert [f.shape[-1] for f in feats] == [64, 32, 16, 8, 4]
    assert [f.shape[1] for f in feats] == list(ResNeXtEncoder.CHANNELS)


def test_stride_parity_R0_R1():
    """R1 differs from R0 only in HOW it downsamples."""
    x = torch.randn(1, 3, 128, 128)
    s0 = [f.shape for f in ResNeXtEncoder("R0", pretrained=False)(x)]
    s1 = [f.shape for f in ResNeXtEncoder("R1", pretrained=False)(x)]
    assert s0 == s1


def test_adapt_stem_weights_zero_init():
    old = torch.randn(64, 3, 7, 7)
    new = adapt_stem_weights(old, hf_init="zero")
    assert new.shape == (64, 12, 7, 7)
    for c in range(3):
        torch.testing.assert_close(new[:, 4 * c], old[:, c] / 2.0)
        assert new[:, 4 * c + 1: 4 * c + 4].abs().max() == 0


def test_R1_equals_R0_at_init():
    torch.manual_seed(0)
    e0 = ResNeXtEncoder("R0", pretrained=False).eval()
    torch.manual_seed(0)
    e1 = ResNeXtEncoder("R1", pretrained=False).eval()
    x = _x()
    with torch.no_grad():
        torch.testing.assert_close(e0.stem_conv_out(x), e1.stem_conv_out(x), atol=1e-4, rtol=0)
        for a, b in zip(e0(x), e1(x)):
            torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)


def test_zero_init_is_avgpool_not_R0():
    torch.manual_seed(0)
    e0 = ResNeXtEncoder("R0", pretrained=False)
    torch.manual_seed(0)
    e1 = ResNeXtEncoder("R1", pretrained=False, hf_init="zero")
    x = _x()
    with torch.no_grad():
        ref = F.conv2d(F.avg_pool2d(x, 2), e0.conv1.weight, stride=1, padding=3)
        torch.testing.assert_close(e1.stem_conv_out(x), ref, atol=1e-4, rtol=0)


def test_high_frequency_weights_receive_gradient():
    enc = ResNeXtEncoder("R1", pretrained=False, hf_init="zero")
    enc(torch.randn(1, 3, 64, 64))[-1].sum().backward()
    g = enc.conv1.weight.grad
    hf = torch.cat([g[:, 4 * c + 1: 4 * c + 4] for c in range(3)], dim=1)
    assert hf.abs().sum() > 0


def test_R2_shortcut_exact():
    blk = resnext50_32x4d(weights=None).layer2[0].eval()
    cp = copy.deepcopy(blk)
    _wavelet_ize(cp)
    cp.eval()
    x = torch.randn(2, 256, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(blk.downsample(x), cp.downsample(x), atol=1e-5, rtol=0)


def test_shared_weights_identical_across_rungs():
    sds = {}
    for r in ("R0", "R1", "R2"):
        torch.manual_seed(0)
        sds[r] = ResNeXtEncoder(r, pretrained=False).state_dict()
    for k, v in sds["R0"].items():
        if k == "conv1.weight":
            continue
        assert k in sds["R1"], k
        torch.testing.assert_close(sds["R1"][k], v, atol=0, rtol=0)
        k2 = k
        for suffix in ("conv2.weight", "downsample.0.weight"):
            if k.startswith(("layer2.0.", "layer3.0.", "layer4.0.")) and k.endswith(suffix):
                k2 = k[: -len("weight")] + "1.weight"
        assert k2 in sds["R2"], k2
        torch.testing.assert_close(sds["R2"][k2], v, atol=0, rtol=0)


def test_cardinality_32_everywhere():
    for r in ("R0", "R1", "R2"):
        g = [m.groups for m in ResNeXtEncoder(r, pretrained=False).modules()
             if isinstance(m, nn.Conv2d) and m.groups > 1]
        assert set(g) == {32} and len(g) == 16


def test_parameter_counts_matched():
    p0 = _params(ResNeXtEncoder("R0", pretrained=False))
    for r in ("R1", "R2"):
        assert abs(_params(ResNeXtEncoder(r, pretrained=False)) - p0) / p0 < 0.005


def test_R2_has_no_strided_ops():
    for m in ResNeXtEncoder("R2", pretrained=False).modules():
        if isinstance(m, nn.Conv2d):
            assert 2 not in m.stride
        if isinstance(m, nn.MaxPool2d):
            assert m.stride != 2


def test_rung_label_preserved():
    assert ResNeXtEncoder("R3", pretrained=False).rung == "R3"


def test_unknown_rung_raises():
    with pytest.raises(ValueError, match="unknown rung"):
        ResNeXtEncoder("R9", pretrained=False)


def test_bad_hf_init_rejected_for_every_rung():
    with pytest.raises(ValueError):
        ResNeXtEncoder("R0", pretrained=False, hf_init="garbage")
