"""Plain U-Net, no pretrained encoder. Reference point for the comparison table."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, widths=(64, 128, 256, 512), bottleneck=1024) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        c = 3
        for w in widths:
            self.downs.append(_block(c, w)); c = w
        self.bottleneck = _block(c, bottleneck)
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.up_blocks.append(_block(w * 2, w)); c = w
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(
                f"input spatial dims must be divisible by 32, got {(h, w)}; "
                "pad or crop the tile before the forward pass"
            )
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, blk, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            x = blk(torch.cat([up(x), skip], dim=1))
        out = self.head(x)
        assert out.shape[-2:] == (h, w), f"resolution invariant violated: {out.shape[-2:]} != {(h, w)}"
        return out
