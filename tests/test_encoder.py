import pytest, torch
from roadseg.models.encoder import ResNeXtEncoder, adapt_stem_weights


@pytest.mark.parametrize("rung", ["R0", "R1", "R2"])
def test_stage_shapes(rung):
    enc = ResNeXtEncoder(rung, pretrained=False)
    feats = enc(torch.randn(1, 3, 128, 128))
    assert [f.shape[-1] for f in feats] == [64, 32, 16, 8, 4]
    assert [f.shape[1] for f in feats] == list(ResNeXtEncoder.CHANNELS)


def test_stride_parity_R0_R1():
    """The spec's core guarantee: R1 differs from R0 only in HOW it downsamples."""
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


def test_R1_stem_matches_R0_on_constant_input():
    """With zero-init high-frequency weights, R1 starts equivalent to R0 on a
    half-resolution image. Checked on the interior to avoid padding effects."""
    torch.manual_seed(0)
    e0 = ResNeXtEncoder("R0", pretrained=False)
    e1 = ResNeXtEncoder("R1", pretrained=False)
    e1.load_adapted_stem_from(e0)
    x = torch.full((1, 3, 64, 64), 0.7)
    with torch.no_grad():
        a = e0.stem_conv_out(x)[..., 4:-4, 4:-4]
        b = e1.stem_conv_out(x)[..., 4:-4, 4:-4]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_high_frequency_weights_receive_gradient():
    """Zero-initialised is not dead: dL/dW = grad_out * input, both nonzero."""
    enc = ResNeXtEncoder("R1", pretrained=False, hf_init="zero")
    enc(torch.randn(1, 3, 64, 64))[-1].sum().backward()
    g = enc.conv1.weight.grad
    hf = torch.cat([g[:, 4 * c + 1: 4 * c + 4] for c in range(3)], dim=1)
    assert hf.abs().sum() > 0


def test_unknown_rung_raises():
    with pytest.raises(ValueError, match="unknown rung"):
        ResNeXtEncoder("R9", pretrained=False)
