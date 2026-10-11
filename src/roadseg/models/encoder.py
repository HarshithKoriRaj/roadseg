"""ResNeXt-50 32x4d encoder with a configurable downsampling scheme.

Rungs
-----
R0  stock: conv7x7 s2 -> maxpool s2 -> layers with strided convs
R1  the paper: DWT at input (3->12 ch), conv7x7 **stride 1**, rest as R0
R2  R1 plus wavelet downsampling replacing every remaining stride-2 op

R1's stem convolution is stride 1 because the DWT has already halved the input.
That is what keeps total stride at 32 for every rung and makes R1 - R0
attributable to the wavelet rather than to a resolution change.

Note a documented confound: R1's 7x7 kernel sees a half-resolution LL band, so
its footprint is 14x14 original pixels versus 7x7 for R0. This is intrinsic to
the paper's design, not an artifact of this implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d

from .ops import WaveletDown
from .wavelet import DWT2d

RUNGS = ("R0", "R1", "R2", "R3")


def adapt_stem_weights(old_w: torch.Tensor, hf_init: str = "zero") -> torch.Tensor:
    """Expand a [O,3,k,k] RGB stem kernel to [O,12,k,k] for DWT sub-band input.

    Sub-band order is [c-LL, c-LH, c-HL, c-HH] per colour channel c.

    The LL band of an orthonormal Haar transform equals 2x the local mean, so
    the pretrained kernel is halved to preserve response magnitude. The three
    high-frequency bands start at zero, which makes the adapted network exactly
    equivalent at initialisation to the pretrained one applied to a
    half-resolution image.
    """
    out_c, in_c, kh, kw = old_w.shape
    new_w = old_w.new_zeros(out_c, in_c * 4, kh, kw)
    for c in range(in_c):
        new_w[:, 4 * c] = old_w[:, c] / 2.0
        if hf_init == "small-random":
            new_w[:, 4 * c + 1: 4 * c + 4].normal_(0.0, float(old_w.std()) * 0.01)
        elif hf_init != "zero":
            raise ValueError(f"unknown hf_init {hf_init!r}; expected 'zero' or 'small-random'")
    return new_w


def _wavelet_ize(block: nn.Module) -> None:
    """Swap a bottleneck block's stride-2 downsampling for a wavelet one.

    The grouped 3x3 conv is KEPT, at stride 1, with the DWT placed in front of
    it. Replacing conv2 outright would drop both its 3x3 spatial mixing and its
    32-way grouping, changing far more than the downsampling operator and
    making R2 - R0 unattributable.
    """
    conv = block.conv2                                   # 3x3, groups=32, stride 2
    conv.stride = (1, 1)
    block.conv2 = nn.Sequential(WaveletDown(conv.in_channels, conv.in_channels), conv)
    ds = block.downsample[0]                             # 1x1 stride-2 projection
    block.downsample[0] = WaveletDown(ds.in_channels, ds.out_channels)


class ResNeXtEncoder(nn.Module):
    CHANNELS = (64, 256, 512, 1024, 2048)

    def __init__(self, rung: str, pretrained: bool = True, hf_init: str = "zero") -> None:
        super().__init__()
        if rung not in RUNGS:
            raise ValueError(f"unknown rung {rung!r}; expected one of {RUNGS}")
        self.rung = "R2" if rung == "R3" else rung   # R3 differs only in loss
        self.hf_init = hf_init

        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnext50_32x4d(weights=weights)

        self.dwt = DWT2d(3) if self.rung in ("R1", "R2") else None
        if self.dwt is None:
            self.conv1 = net.conv1
        else:
            self.conv1 = nn.Conv2d(12, 64, 7, stride=1, padding=3, bias=False)
            with torch.no_grad():
                self.conv1.weight.copy_(adapt_stem_weights(net.conv1.weight, hf_init))

        self.bn1, self.relu = net.bn1, net.relu
        self.maxpool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

        if self.rung == "R2":
            self.maxpool = WaveletDown(64, 64)
            for layer in (self.layer2, self.layer3, self.layer4):
                _wavelet_ize(layer[0])

    def load_adapted_stem_from(self, other: "ResNeXtEncoder") -> None:
        """Copy another encoder's 3-channel stem in, adapted to 12 channels."""
        with torch.no_grad():
            self.conv1.weight.copy_(adapt_stem_weights(other.conv1.weight, self.hf_init))
            self.bn1.load_state_dict(other.bn1.state_dict())

    def stem_conv_out(self, x: torch.Tensor) -> torch.Tensor:
        """Stem convolution output, before BN/ReLU. Used by parity tests."""
        return self.conv1(self.dwt(x) if self.dwt is not None else x)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.relu(self.bn1(self.stem_conv_out(x)))
        f0 = x
        f1 = self.layer1(self.maxpool(x))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]
