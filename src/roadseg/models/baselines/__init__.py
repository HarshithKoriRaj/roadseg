from __future__ import annotations

import torch.nn as nn

from .dlinknet import DLinkNet
from .unet import UNet


def build_baseline(name: str, pretrained: bool = True) -> nn.Module:
    """`pretrained` applies to D-LinkNet only; U-Net has no pretrained weights."""
    if name == "unet":
        return UNet()
    if name == "dlinknet":
        return DLinkNet(pretrained=pretrained)
    raise ValueError(f"unknown baseline {name!r}; expected 'unet' or 'dlinknet'")
