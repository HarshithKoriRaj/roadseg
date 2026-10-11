"""ResNeXt-50 32x4d encoder with a configurable downsampling scheme.

Rungs
-----
R0  stock: conv7x7 s2 -> maxpool s2 -> layers with strided convs
R1  the paper: DWT at input (3->12 ch), conv7x7 **stride 1**, rest as R0
R2  R1 plus wavelet pooling replacing every remaining stride-2 op (R3 uses the
    R2 architecture; it differs only in the loss)

Every rung keeps total stride 32 and cardinality 32, and keeps every pretrained
weight. R1's stem is initialised EXACTLY (hf_init="exact"): the 7x7 kernel,
zero-padded to 8x8, is transformed by the same Haar DWT, so R1 reproduces R0's
stem output and hence all of R0's features at step 0. R2 puts a WaveletPool
(DWT + depthwise 4->1 combination, 4 params/channel) in front of each KEPT
pretrained strided conv, now stride 1; the pools start as 2x2 average pooling
(main path, stem) or exact top-left subsampling (1x1 shortcut). R2 is therefore
a close, not exact, function-preserving initialisation: the 3x3 grouped conv
sees average-pooled instead of strided input.

Measured encoder parameters (pretrained=False): R0 22,979,904;
R1-R0 = +28,224 (+0.12%, the 12-channel stem), R2-R0 = +42,816 (+0.19%, WaveletPools).

Documented confound: R1's 7x7 kernel sees a half-resolution LL band, so its
footprint is 14x14 original pixels versus 7x7 for R0 (intrinsic to the paper).
The "zero" and "small-random" stem inits are NOT equivalent to R0: "zero" is
the pretrained kernel applied at stride 1 to a 2x2-average-pooled image.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d

from .ops import WaveletPool
from .wavelet import DWT2d

RUNGS = ("R0", "R1", "R2", "R3")


HF_INITS = ("exact", "zero", "small-random")


def adapt_stem_weights(old_w: torch.Tensor, hf_init: str = "exact") -> torch.Tensor:
    """Expand a [O,3,k,k] RGB stem kernel to [O,12,k,k] for DWT sub-band input.

    Sub-band order is [c-LL, c-LH, c-HL, c-HH] per colour channel c.

    "exact" (needs k == 7): function-preserving. R0 computes
    y[i,j] = sum W[c,u,v] x[c, 2i+u-3, 2j+v-3]. Padding W by one zero row/column
    at the top/left makes its 8x8 support cover four whole 2x2 blocks per axis;
    since Haar is orthonormal, the response is the DWT of the padded kernel
    applied to the DWT coefficients, placed at taps 1:5 of a 7-tap stride-1,
    pad-3 conv. Exact including borders for even image sizes.
    "zero": LL = W/2, HF = 0. This is the pretrained kernel at stride 1 on a
    2x2-average-pooled image; it is NOT equivalent to R0.
    "small-random": as "zero" but HF ~ N(0, (0.01*std(W))^2).
    """
    if hf_init not in HF_INITS:
        raise ValueError(f"unknown hf_init {hf_init!r}; expected one of {HF_INITS}")
    out_c, in_c, kh, kw = old_w.shape
    if hf_init == "exact":
        if (kh, kw) != (7, 7):
            raise ValueError(f"hf_init='exact' requires a 7x7 stem kernel, got {kh}x{kw}")
        k = F.pad(old_w, (1, 0, 1, 0))                    # [O,3,8,8]
        with torch.no_grad():
            g = DWT2d(in_c).to(old_w.device, old_w.dtype)(k)   # [O,12,4,4]
        new_w = old_w.new_zeros(out_c, in_c * 4, kh, kw)
        new_w[:, :, 1:5, 1:5] = g
        return new_w
    new_w = old_w.new_zeros(out_c, in_c * 4, kh, kw)
    for c in range(in_c):
        new_w[:, 4 * c] = old_w[:, c] / 2.0
        if hf_init == "small-random":
            new_w[:, 4 * c + 1: 4 * c + 4].normal_(0.0, float(old_w.std()) * 0.01)
    return new_w


def _wavelet_ize(block: nn.Module) -> None:
    """Make a stride-2 bottleneck downsample with WaveletPool, keeping all weights.

    The pretrained strided convs are KEPT at stride 1 with a WaveletPool in front,
    so no pretrained weight is discarded and no dense random projection is added.
    """
    conv = block.conv2
    if conv.stride != (2, 2) or conv.groups != 32:
        raise RuntimeError(f"unexpected bottleneck layout: conv2 stride={conv.stride} groups={conv.groups}")
    conv.stride = (1, 1)
    block.conv2 = nn.Sequential(WaveletPool(conv.in_channels, "ll"), conv)

    ds = block.downsample[0]
    if ds.stride != (2, 2) or ds.kernel_size != (1, 1):
        raise RuntimeError(f"unexpected shortcut layout: stride={ds.stride} kernel={ds.kernel_size}")
    ds.stride = (1, 1)
    block.downsample[0] = nn.Sequential(WaveletPool(ds.in_channels, "select"), ds)


class ResNeXtEncoder(nn.Module):
    CHANNELS = (64, 256, 512, 1024, 2048)

    def __init__(self, rung: str, pretrained: bool = True, hf_init: str = "exact") -> None:
        super().__init__()
        if hf_init not in HF_INITS:
            raise ValueError(f"unknown hf_init {hf_init!r}; expected one of {HF_INITS}")
        if rung not in RUNGS:
            raise ValueError(f"unknown rung {rung!r}; expected one of {RUNGS}")
        self.rung = rung
        self._arch = "R2" if rung == "R3" else rung   # R3 differs only in loss
        self.hf_init = hf_init

        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnext50_32x4d(weights=weights)

        self.dwt = DWT2d(3) if self._arch in ("R1", "R2") else None
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

        if self._arch == "R2":
            self.maxpool = WaveletPool(64, "ll")
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
