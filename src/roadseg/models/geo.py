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


# ---------------------------------------------------------------- geometry feedback (GFa, GFb)
# GA discards its centre/width predictions at inference. GF predicts them PART-WAY up the decoder
# and concatenates the predicted geometry (centre probability + width-class probabilities) into
# every later decoder stage, so the mask head is computed from features that saw the geometry.
# Same targets and loss as GA (losses.GeoLoss); the geometry logits are upsampled bilinearly to
# the input size for the loss. Two variants, fixed before any run (campaign gfsel picks one on
# VALIDATION IoU): the decoder stage after which geometry is predicted.

GF_STAGES = {"GFa": 3, "GFb": 2}     # after stage 3: 64 ch at 1/2 res; after stage 2: 128 ch at 1/4


class FeedbackDecoder(nn.Module):
    """UNetDecoder (same widths, skips and up-sampling) plus a geometry head after stage
    `after`, whose probabilities are concatenated into the input of every later stage."""

    def __init__(self, enc_channels, n_geo: int, after: int,
                 widths=(512, 256, 128, 64, 32), up: str = "bilinear") -> None:
        super().__init__()
        from .decoder import _ConvBlock
        from .ops import make_up
        f0, f1, f2, f3, f4 = enc_channels
        skips = [f3, f2, f1, f0, 0]
        ins = [f4] + list(widths[:-1])
        extra = [n_geo if k > after else 0 for k in range(len(widths))]
        self.after = after
        self.ups = nn.ModuleList([make_up(up, i, w) for i, w in zip(ins, widths)])
        self.blocks = nn.ModuleList([_ConvBlock(w + s + e, w) for w, s, e in zip(widths, skips, extra)])
        self.geo_head = nn.Conv2d(widths[after], n_geo, 1)

    def forward(self, feats: list[torch.Tensor]):
        f0, f1, f2, f3, f4 = feats
        x, geo, geo_logits = f4, None, None
        for k, (up, block, skip) in enumerate(zip(self.ups, self.blocks, [f3, f2, f1, f0, None])):
            x = up(x)
            parts = [x] if skip is None else [x, skip]
            if geo is not None:
                parts.append(F.interpolate(geo, size=x.shape[-2:], mode="bilinear", align_corners=False))
            x = block(torch.cat(parts, dim=1))
            if k == self.after:
                geo_logits = self.geo_head(x)
                geo = torch.cat([torch.sigmoid(geo_logits[:, :1]),
                                 torch.softmax(geo_logits[:, 1:], dim=1)], dim=1)
        return x, geo_logits


class GeoFeedbackSegmenter(nn.Module):
    def __init__(self, rung: str, radii=RADII, pretrained: bool = True) -> None:
        super().__init__()
        if rung not in GF_STAGES:
            raise ValueError(f"unknown feedback rung {rung!r}; expected one of {list(GF_STAGES)}")
        from .encoder import ResNeXtEncoder
        self.rung, self.mode, self.radii = rung, "feedback", tuple(int(r) for r in radii)
        # R0's build order (decoder, head, encoder); the decoder's shapes differ from R0's.
        self.decoder = FeedbackDecoder(ResNeXtEncoder.CHANNELS, 1 + len(self.radii), GF_STAGES[rung])
        self.head = nn.Conv2d(32, 1, 1)
        self.encoder = ResNeXtEncoder("R0", pretrained=pretrained)

    def forward(self, x: torch.Tensor, aux: bool = False):
        h, w = x.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(f"input spatial dims must be divisible by 32, got {(h, w)}; "
                             "pad or crop the tile before the forward pass")
        f, g = self.decoder(self.encoder(x))
        mask = self.head(f)
        assert mask.shape[-2:] == (h, w), f"resolution invariant violated: {mask.shape[-2:]} != {(h, w)}"
        if aux:
            g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
            return {"mask": mask, "centre": g[:, :1], "width": g[:, 1:]}
        return mask
