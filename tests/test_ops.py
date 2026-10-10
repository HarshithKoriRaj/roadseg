import pytest
import torch
from roadseg.models.ops import make_down, make_up


@pytest.mark.parametrize("kind", ["stride", "wavelet"])
def test_down_halves_resolution(kind):
    op = make_down(kind, 16, 32)
    out = op(torch.randn(2, 16, 32, 32))
    assert out.shape == (2, 32, 16, 16)


@pytest.mark.parametrize("kind", ["bilinear", "wavelet"])
def test_up_doubles_resolution(kind):
    op = make_up(kind, 32, 16)
    out = op(torch.randn(2, 32, 16, 16))
    assert out.shape == (2, 16, 32, 32)


def test_wavelet_down_is_information_preserving_before_projection():
    """The 1x1 projection may lose information, but the DWT stage must not."""
    from roadseg.models.wavelet import DWT2d, IDWT2d
    x = torch.randn(1, 8, 16, 16, dtype=torch.float64)
    torch.testing.assert_close(IDWT2d(8)(DWT2d(8)(x)), x, atol=1e-10, rtol=0)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown"):
        make_down("bogus", 4, 4)
