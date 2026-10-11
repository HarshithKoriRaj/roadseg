import pytest, torch
from roadseg.models.segmenter import RoadSegmenter


@pytest.mark.parametrize("rung", ["R0", "R1", "R2", "R3"])
def test_output_resolution_equals_input(rung):
    """The draft's architecture emits a half-resolution mask. This must not."""
    m = RoadSegmenter(rung, pretrained=False)
    x = torch.randn(1, 3, 128, 128)
    out = m(x)
    assert out.shape == (1, 1, 128, 128)


def test_output_is_logits_not_probabilities():
    m = RoadSegmenter("R0", pretrained=False)
    with torch.no_grad():
        out = m(torch.randn(2, 3, 64, 64) * 5)
    assert out.min() < 0.0, "logits should span negatives; model must not apply sigmoid"


@pytest.mark.parametrize("bad", [(1, 3, 100, 100), (1, 3, 128, 130), (1, 3, 33, 33)])
def test_rejects_dims_not_divisible_by_32(bad):
    """R2 applies the DWT five times; odd dims at any level corrupt shapes.
    Fail loudly at the entry point rather than mismatching deep in the decoder."""
    m = RoadSegmenter("R2", pretrained=False)
    with pytest.raises(ValueError, match="divisible by 32"):
        m(torch.randn(*bad))


def test_R0_and_R1_have_identical_output_shapes_for_many_sizes():
    for s in (64, 128, 256):
        x = torch.randn(1, 3, s, s)
        a = RoadSegmenter("R0", pretrained=False)(x)
        b = RoadSegmenter("R1", pretrained=False)(x)
        assert a.shape == b.shape == (1, 1, s, s)


def _build(rung, seed=0):
    torch.manual_seed(seed)
    return RoadSegmenter(rung, pretrained=False).eval()


def test_decoder_and_head_are_seed_paired_across_rungs():
    """At a fixed seed, every decoder/head tensor of R0 must be identical in R1, R2, R3."""
    ref = {k: v for k, v in _build("R0").state_dict().items() if k.startswith(("decoder.", "head."))}
    assert ref
    for rung in ("R1", "R2", "R3"):
        sd = _build(rung).state_dict()
        for k, v in ref.items():
            assert k in sd and torch.equal(sd[k], v), f"{rung}: {k}"


def test_R1_is_the_same_function_as_R0_at_init():
    """Exact stem adaptation + paired seeds => R1 and R0 produce the same logits at step 0."""
    x = torch.randn(2, 3, 64, 64) * torch.tensor([1.0, 2.0, 0.5]).view(1, 3, 1, 1)
    with torch.no_grad():
        a, b = _build("R0")(x), _build("R1")(x)
    torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)


def test_R3_forwards_and_matches_R2_architecture():
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a, b = _build("R2")(x), _build("R3")(x)
    assert a.shape == b.shape == (1, 1, 64, 64)
    torch.testing.assert_close(a, b)   # R3 differs from R2 only in the loss


def test_gradients_reach_every_parameter():
    m = RoadSegmenter("R2", pretrained=False)
    m(torch.randn(2, 3, 64, 64)).sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, dead[:5]
