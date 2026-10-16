"""Rungs GA and GB: the R0 network with geometric heads (centre-line + width class).

  GA  R0 + a centre-line head and a width-class head trained as AUXILIARY outputs. The mask
      comes from R0's own mask head, so GA - R0 is the effect of the geometric supervision.
  GB  the same heads and targets, but the mask is DRAWN from them by `render`: each
      centre-line pixel dilated by its predicted radius. R0's mask head is kept (unused, so
      every parameter is initialised as in GA at the same seed). GB - GA is the effect of
      the representation, with supervision held fixed.

Both return mask LOGITS of the input's size from `forward(x)`, so threshold selection,
tiled inference and scoring are R0's code unchanged; `forward(x, aux=True)` returns the dict
the training loss needs. GB's logits are logit(m) of the drawn probability m, clamped away
from 0 and 1 so the logit is finite.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.geometry import RADII
from .segmenter import RoadSegmenter

GEO_RUNGS = {"GA": "aux", "GB": "render", "GAc": "aux", "GAw": "aux",   # GAc/GAw: GA, one loss term off
             "GD": "aux", "GO": "aux"}       # GD/GO: GA's model with prior targets (DDT, orientation)
EPS = 1e-4


def _square(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, stride=1, padding=1)


def _cross(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(F.max_pool2d(x, (3, 1), stride=1, padding=(1, 0)),
                         F.max_pool2d(x, (1, 3), stride=1, padding=(0, 1)))


def render(contrib: torch.Tensor, radii=RADII) -> torch.Tensor:
    """Variable-radius dilation: [B,K,H,W] -> [B,1,H,W], max over k of dilate_{radii[k]}(contrib[:,k]).

    One pass of max(radii) 3x3 max-pools. Bin k enters when the countdown reaches its radius
    and is dilated by every remaining step; dilation distributes over max, so each bin ends
    up dilated by exactly its own radius. Steps alternate square (odd countdown) and cross
    (even), an octagonal approximation of a disc. Differentiable (max-pool's gradient goes
    to the arg-max). Call in fp32.
    """
    radii = [int(r) for r in radii]
    if contrib.shape[1] != len(radii):
        raise ValueError(f"{contrib.shape[1]} channels for {len(radii)} radii")
    out = None
    for j in range(max(radii), -1, -1):
        for k, r in enumerate(radii):
            if r == j:
                c = contrib[:, k:k + 1]
                out = c if out is None else torch.maximum(out, c)
        if j > 0 and out is not None:
            out = _square(out) if j % 2 else _cross(out)
    return out


class GeoSegmenter(RoadSegmenter):
    def __init__(self, rung: str, radii=RADII, pretrained: bool = True,
                 n_classes: int | None = None) -> None:
        if rung not in GEO_RUNGS:
            raise ValueError(f"unknown geometric rung {rung!r}; expected one of {list(GEO_RUNGS)}")
        # R0's decoder, mask head and encoder, built in R0's order: at the same seed they are
        # initialised exactly as R0's. The geometric heads are built after them.
        super().__init__(rung="R0", pretrained=pretrained)
        self.rung, self.mode, self.radii = rung, GEO_RUNGS[rung], tuple(int(r) for r in radii)
        self.centre_head = nn.Conv2d(32, 1, 1)
        # class head: width classes (GA/GB), or a baseline's classes (GD: DDT scales, GO: angles)
        self.width_head = nn.Conv2d(32, n_classes or len(self.radii), 1)
        if self.mode == "render" and self.width_head.out_channels != len(self.radii):
            raise ValueError("render needs one class per radius")

    def forward(self, x: torch.Tensor, aux: bool = False):
        h, w = x.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(f"input spatial dims must be divisible by 32, got {(h, w)}; "
                             "pad or crop the tile before the forward pass")
        f = self.decoder(self.encoder(x))
        centre, width = self.centre_head(f), self.width_head(f)
        if self.mode == "aux":
            mask = self.head(f)
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                contrib = torch.sigmoid(centre.float()) * torch.softmax(width.float(), dim=1)
                m = render(contrib, self.radii).clamp(EPS, 1.0 - EPS)
                mask = torch.log(m) - torch.log1p(-m)
        assert mask.shape[-2:] == (h, w), f"resolution invariant violated: {mask.shape[-2:]} != {(h, w)}"
        if aux:
            return {"mask": mask, "centre": centre, "width": width}
        return mask
