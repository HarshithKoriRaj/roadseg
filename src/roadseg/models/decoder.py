"""U-Net decoder with explicit encoder skip connections.

The draft paper hedges ("skip connections can be used"); this does not. Skips
are the main determinant of thin-road and boundary quality, so they are wired
unconditionally and a test asserts every one of them carries gradient.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ops import make_up


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoder(nn.Module):
    def __init__(
        self,
        enc_channels: tuple[int, ...] = (64, 256, 512, 1024, 2048),
        widths: tuple[int, ...] = (512, 256, 128, 64, 32),
        up: str = "bilinear",
    ) -> None:
        super().__init__()
        f0, f1, f2, f3, f4 = enc_channels
        skips = [f3, f2, f1, f0, 0]          # d5 has no encoder skip
        ins = [f4] + list(widths[:-1])
        # Construction order is load-bearing: both up kinds draw RNG only in their
        # 3x3 conv (WaveletUnpool is deterministic), so identical order => identical
        # seeded init across rungs.
        self.ups = nn.ModuleList([make_up(up, i, w) for i, w in zip(ins, widths)])
        self.blocks = nn.ModuleList([_ConvBlock(w + s, w) for w, s in zip(widths, skips)])

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        f0, f1, f2, f3, f4 = feats
        x = f4
        for up, block, skip in zip(self.ups, self.blocks, [f3, f2, f1, f0, None]):
            x = up(x)
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x
