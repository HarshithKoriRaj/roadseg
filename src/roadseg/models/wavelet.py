"""Orthonormal Haar DWT/IDWT as fixed-weight convolutions.

Implemented as grouped conv2d / conv_transpose2d with frozen 2x2 kernels so the
transform runs on GPU, batched, differentiable, and exactly invertible. A CPU
library such as pywt cannot be used here: R2 applies the transform *inside* the
network at every downsampling stage.

Because the four 2x2 Haar kernels form an orthonormal basis of each 2x2 block,
conv_transpose2d (the adjoint of conv2d) is an exact inverse.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_S = 1.0 / math.sqrt(2.0)
_LO = (_S, _S)
_HI = (_S, -_S)


def haar_filters(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Four 2x2 analysis kernels, shape [4,1,2,2], order LL, LH, HL, HH."""
    lo = torch.tensor(_LO, dtype=dtype)
    hi = torch.tensor(_HI, dtype=dtype)
    # outer(rows, cols): LL=lo*lo, LH=hi*lo, HL=lo*hi, HH=hi*hi
    bank = torch.stack([
        torch.outer(lo, lo),
        torch.outer(hi, lo),
        torch.outer(lo, hi),
        torch.outer(hi, hi),
    ])
    return bank.unsqueeze(1)


class DWT2d(nn.Module):
    """[B,C,H,W] -> [B,4C,H/2,W/2], channels ordered c0-LL,c0-LH,c0-HL,c0-HH,c1-LL,..."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        w = haar_filters(torch.float64).repeat(channels, 1, 1, 1)   # [4C,1,2,2]
        self.register_buffer("weight", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 or x.shape[-2] % 2:
            raise ValueError(f"DWT2d needs even spatial dims, got {tuple(x.shape[-2:])}")
        return F.conv2d(x, self.weight.to(x.dtype), stride=2, groups=self.channels)


class IDWT2d(nn.Module):
    """[B,4C,H,W] -> [B,C,2H,2W]. Exact inverse of DWT2d(channels)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        w = haar_filters(torch.float64).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 4 * self.channels:
            raise ValueError(f"IDWT2d({self.channels}) expects {4*self.channels} channels, got {x.shape[1]}")
        return F.conv_transpose2d(x, self.weight.to(x.dtype), stride=2, groups=self.channels)
