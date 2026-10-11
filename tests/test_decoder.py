import pytest, torch
from roadseg.models.decoder import UNetDecoder


def _feats(n=1, size=128):
    ch, res = (64, 256, 512, 1024, 2048), (size, size // 2, size // 4, size // 8, size // 16)
    return [torch.randn(n, c, r, r) for c, r in zip(ch, res)]


@pytest.mark.parametrize("up", ["bilinear", "wavelet"])
def test_returns_full_resolution(up):
    out = UNetDecoder(up=up)(_feats(size=128))
    assert out.shape == (1, 32, 256, 256)   # stem is at H/2, so output is 2x stem


def test_all_skips_are_consumed():
    """A decoder that silently ignores a skip would still produce right shapes."""
    dec = UNetDecoder()
    feats = _feats()
    for f in feats:
        f.requires_grad_(True)
    dec(feats).sum().backward()
    assert all(f.grad is not None and f.grad.abs().sum() > 0 for f in feats)


def test_decoder_weights_are_seed_paired_across_up_kinds():
    """At a fixed seed the bilinear and wavelet decoders must start from identical
    conv/BN weights, so a rung comparison is not confounded by initialisation."""
    torch.manual_seed(0); a = UNetDecoder(up="bilinear")
    torch.manual_seed(0); b = UNetDecoder(up="wavelet")
    sa, sb = a.state_dict(), b.state_dict()
    shared = [k for k in sa if k in sb]
    assert len(shared) == len(sa), "every bilinear-decoder tensor must exist in the wavelet decoder"
    for k in shared:
        assert torch.equal(sa[k], sb[k]), k
    extra = [k for k in sb if k not in sa]
    assert extra and all(("unpool" in k) or ("idwt" in k) for k in extra), extra
