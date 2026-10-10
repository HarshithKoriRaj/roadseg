"""Swappable resolution-changing operators.

The whole ablation reduces to which of these the encoder and decoder use, so
they share a signature (in_ch, out_ch) and are constructed by name.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import DWT2d, IDWT2d


class StridedDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class WaveletDown(nn.Module):
    """Lossless halving: keep all four sub-bands, then project to out_ch."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.dwt = DWT2d(in_ch)
        self.proj = nn.Conv2d(4 * in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dwt(x))


class BilinearUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class WaveletUp(nn.Module):
    """Predict four sub-bands, then invert the transform."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 4 * out_ch, 1, bias=False)
        self.idwt = IDWT2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.idwt(self.proj(x))


_DOWN = {"stride": StridedDown, "wavelet": WaveletDown}
_UP = {"bilinear": BilinearUp, "wavelet": WaveletUp}


def make_down(kind: str, in_ch: int, out_ch: int) -> nn.Module:
    if kind not in _DOWN:
        raise ValueError(f"unknown down op {kind!r}; expected one of {sorted(_DOWN)}")
    return _DOWN[kind](in_ch, out_ch)


def make_up(kind: str, in_ch: int, out_ch: int) -> nn.Module:
    if kind not in _UP:
        raise ValueError(f"unknown up op {kind!r}; expected one of {sorted(_UP)}")
    return _UP[kind](in_ch, out_ch)
