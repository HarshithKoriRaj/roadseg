import pytest, torch
from roadseg.models.wavelet import DWT2d, IDWT2d, haar_filters


def test_filters_are_orthonormal():
    f = haar_filters(torch.float64).reshape(4, 4)   # each row a flattened 2x2 kernel
    torch.testing.assert_close(f @ f.T, torch.eye(4, dtype=torch.float64))


def test_perfect_reconstruction():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64, dtype=torch.float64)
    y = IDWT2d(3)(DWT2d(3)(x))
    assert y.shape == x.shape
    torch.testing.assert_close(y, x, atol=1e-10, rtol=0)


def test_shapes_and_channel_order():
    x = torch.zeros(1, 3, 8, 8)
    x[:, 1] = 5.0                       # constant in the green channel only
    out = DWT2d(3)(x)
    assert out.shape == (1, 12, 4, 4)
    # green LL is channel 4; orthonormal Haar gives LL = 2 * mean
    torch.testing.assert_close(out[0, 4], torch.full((4, 4), 10.0))
    # green LH/HL/HH are zero for a constant input
    assert out[0, 5:8].abs().max() == 0


def test_wavelet_weights_are_frozen():
    d = DWT2d(3)
    assert all(not p.requires_grad for p in d.parameters())
    assert len(list(d.parameters())) == 0   # registered as a buffer, not a parameter


def test_gradients_flow_through():
    x = torch.randn(1, 3, 16, 16, requires_grad=True)
    DWT2d(3)(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
