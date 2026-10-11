import pytest
import torch

from roadseg.models.baselines import build_baseline
from roadseg.models.segmenter import build_model

NAMES = ["unet", "dlinknet"]


def _build(name):
    return build_baseline(name, pretrained=False)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("size", [64, 128, 256])
def test_output_contract_matches_segmenter(name, size):
    m = _build(name).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, size, size))
    assert out.shape == (1, 1, size, size)


@pytest.mark.parametrize("name", NAMES)
def test_non_square_input(name):
    m = _build(name).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 64, 128))
    assert out.shape == (1, 1, 64, 128)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("hw", [(48, 64), (64, 48), (65, 64)])
def test_rejects_size_not_divisible_by_32(name, hw):
    m = _build(name).eval()
    with pytest.raises(ValueError, match="divisible by 32"):
        m(torch.randn(1, 3, *hw))


@pytest.mark.parametrize("name", NAMES)
def test_emits_logits(name):
    with torch.no_grad():
        out = _build(name)(torch.randn(2, 3, 64, 64) * 5)
    assert out.min() < 0.0


@pytest.mark.parametrize("name", NAMES)
def test_gradients_reach_every_parameter(name):
    m = _build(name).train()
    m(torch.randn(2, 3, 64, 64)).sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert dead == []


def test_unknown_baseline_raises():
    with pytest.raises(ValueError, match="unknown baseline"):
        build_baseline("nope")


def test_build_model_dispatches_to_baselines():
    for cfg in ({"model": "unet"}, {"model": "dlinknet", "pretrained": False}):
        m = build_model(cfg).eval()
        with torch.no_grad():
            assert m(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="unknown baseline"):
        build_model({"model": "nope"})
