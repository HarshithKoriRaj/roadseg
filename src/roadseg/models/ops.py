"""Swappable resolution-changing operators.

The whole ablation reduces to which of these the encoder and decoder use.
Every operator is initialised so that it reproduces the stock operator it
replaces, and adds only a handful of parameters (4 per channel), so rungs
differ in the down/up-sampling kernel and nothing else.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import DWT2d, IDWT2d


class WaveletPool(nn.Module):
    """Learned per-channel 2x2 analysis filter, expressed in the Haar basis.

    DWT, then a depthwise 4->1 combination of each channel's own sub-bands.
    [B,C,H,W] -> [B,C,H/2,W/2]. Adds 4*C parameters and mixes no channels, so it
    can sit in front of a pretrained conv without disturbing it.

    init="ll"      weights (1/2,0,0,0): 2x2 average pooling.
    init="select"  weights (1/2,1/2,1/2,1/2): the top-left pixel of each block,
                   i.e. exactly x[..., ::2, ::2] -- what a stride-2 1x1 conv reads.
    """

    def __init__(self, channels: int, init: str = "ll") -> None:
        super().__init__()
        self.channels = channels
        self.dwt = DWT2d(channels)
        w = torch.zeros(channels, 4, 1, 1)
        if init == "ll":
            w[:, 0] = 0.5
        elif init == "select":
            w[:] = 0.5
        else:
            raise ValueError(f"unknown WaveletPool init {init!r}; expected 'll' or 'select'")
        self.weight = nn.Parameter(w)       # deliberate: no nn.Conv2d, so no RNG draw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(self.dwt(x), self.weight, groups=self.channels)


class WaveletUnpool(nn.Module):
    """Learned per-channel 2x2 synthesis filter: depthwise 1->4 expansion, then IDWT.

    [B,C,H,W] -> [B,C,2H,2W]. Adds 4*C parameters. Initialised to (2,0,0,0), which
    the inverse transform turns into nearest-neighbour upsampling.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        w = torch.zeros(4 * channels, 1, 1, 1)
        w[0::4] = 2.0
        self.weight = nn.Parameter(w)
        self.idwt = IDWT2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.idwt(F.conv2d(x, self.weight, groups=self.channels))


class BilinearUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class WaveletUp(nn.Module):
    """WaveletUnpool -> the SAME 3x3 conv BilinearUp uses (attribute `conv`).

    Differs from BilinearUp only in the 2x upsampling kernel.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.unpool = WaveletUnpool(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.unpool(x))


_UP = {"bilinear": BilinearUp, "wavelet": WaveletUp}


def make_up(kind: str, in_ch: int, out_ch: int) -> nn.Module:
    if kind not in _UP:
        raise ValueError(f"unknown up op {kind!r}; expected one of {sorted(_UP)}")
    return _UP[kind](in_ch, out_ch)
