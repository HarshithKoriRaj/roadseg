import pytest
import torch
import torch.nn.functional as F

from roadseg.models.ops import BilinearUp, WaveletPool, WaveletUnpool, WaveletUp, make_up


def _params(m):
    return sum(p.numel() for p in m.parameters())


def test_pool_ll_is_avgpool():
    x = torch.randn(2, 8, 16, 16)
    torch.testing.assert_close(WaveletPool(8, "ll")(x), F.avg_pool2d(x, 2), atol=1e-6, rtol=0)


def test_pool_select_is_subsample():
    x = torch.randn(2, 8, 16, 16)
    torch.testing.assert_close(WaveletPool(8, "select")(x), x[..., ::2, ::2], atol=1e-6, rtol=0)


def test_unpool_is_nearest():
    x = torch.randn(2, 8, 8, 8)
    torch.testing.assert_close(
        WaveletUnpool(8)(x), F.interpolate(x, scale_factor=2, mode="nearest"), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("op", [WaveletPool(8), WaveletUnpool(8)])
def test_param_count_and_requires_grad(op):
    ps = list(op.parameters())
    assert sum(p.numel() for p in ps) == 4 * 8
    assert all(p.requires_grad for p in ps)


def test_gradients_reach_weights():
    for op in (WaveletPool(8), WaveletUnpool(8)):
        op(torch.randn(1, 8, 8, 8)).sum().backward()
        assert op.weight.grad is not None and op.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["bilinear", "wavelet"])
def test_up_doubles_resolution(kind):
    out = make_up(kind, 32, 16)(torch.randn(2, 32, 16, 16))
    assert out.shape == (2, 16, 32, 32)


def test_up_seed_pairing():
    torch.manual_seed(0)
    a = make_up("bilinear", 32, 16)
    torch.manual_seed(0)
    b = make_up("wavelet", 32, 16)
    torch.testing.assert_close(a.conv.weight, b.conv.weight, atol=0, rtol=0)


def test_up_parameter_parity():
    assert _params(WaveletUp(32, 16)) - _params(BilinearUp(32, 16)) == 4 * 32


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown"):
        make_up("bogus", 4, 4)


def test_unknown_pool_init_raises():
    with pytest.raises(ValueError, match="unknown"):
        WaveletPool(4, "bogus")
