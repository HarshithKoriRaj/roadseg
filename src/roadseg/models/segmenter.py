"""Full segmentation model: rung-configurable encoder + U-Net decoder + head.

Emits logits at exactly the input resolution. The shape assertion in forward()
is deliberate: the draft paper's architecture silently produced a
half-resolution mask, and that class of bug should be impossible here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import UNetDecoder
from .encoder import ResNeXtEncoder

_UP_FOR_RUNG = {"R0": "bilinear", "R1": "bilinear", "R2": "wavelet", "R3": "wavelet"}


class RoadSegmenter(nn.Module):
    def __init__(self, rung: str = "R1", pretrained: bool = True, hf_init: str = "exact") -> None:
        super().__init__()
        self.rung = rung
        # Decoder and head are built BEFORE the encoder on purpose: R1/R2's encoder
        # draws from the global RNG (throwaway 12-channel stem conv) and R0's does
        # not, so anything built after it would be initialised differently per rung
        # at the same seed. Building these first keeps them seed-paired across rungs.
        self.decoder = UNetDecoder(
            enc_channels=ResNeXtEncoder.CHANNELS, up=_UP_FOR_RUNG[rung]
        )
        self.head = nn.Conv2d(32, 1, 1)
        self.encoder = ResNeXtEncoder(rung, pretrained=pretrained, hf_init=hf_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(
                f"input spatial dims must be divisible by 32, got {(h, w)}; "
                "pad or crop the tile before the forward pass"
            )
        out = self.head(self.decoder(self.encoder(x)))
        assert out.shape[-2:] == (h, w), f"resolution invariant violated: {out.shape[-2:]} != {(h, w)}"
        return out


def build_model(cfg: dict) -> nn.Module:
    name = cfg.get("model", "roadsegmenter").lower()
    if name == "roadsegmenter" and cfg.get("rung") in ("GA", "GB", "GAc", "GAw"):
        from .geo import GeoSegmenter
        from ..data.geometry import RADII
        return GeoSegmenter(cfg["rung"], radii=cfg.get("geo_radii", RADII),
                            pretrained=cfg.get("pretrained", True))
    if name == "roadsegmenter":
        return RoadSegmenter(
            rung=cfg.get("rung", "R1"),
            pretrained=cfg.get("pretrained", True),
            hf_init=cfg.get("hf_init", "exact"),
        )
    from .baselines import build_baseline
    return build_baseline(name, pretrained=cfg.get("pretrained", True))
