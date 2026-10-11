"""D-LinkNet: ResNet-34 encoder + dilated centre block + LinkNet decoder.

Winner of the DeepGlobe 2018 road extraction challenge. The dilated block
widens the receptive field without losing resolution, which is what makes it
strong on long thin roads.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


class _Dilated(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(ch, ch, 3, padding=d, dilation=d) for d in (1, 2, 4, 8)]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for b in self.branches:
            out = out + self.relu(b(x))
        return out


class _Up(nn.Module):
    def __init__(self, i: int, o: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(i, i // 4, 1, bias=False), nn.BatchNorm2d(i // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(i // 4, i // 4, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(i // 4), nn.ReLU(inplace=True),
            nn.Conv2d(i // 4, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DLinkNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.e1, self.e2, self.e3, self.e4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.center = _Dilated(512)
        self.d4, self.d3, self.d2, self.d1 = _Up(512, 256), _Up(256, 128), _Up(128, 64), _Up(64, 64)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(
                f"input spatial dims must be divisible by 32, got {(h, w)}; "
                "pad or crop the tile before the forward pass"
            )
        s = self.stem(x)
        e1 = self.e1(self.pool(s))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.center(self.e4(e3))
        d = self.d4(e4) + e3
        d = self.d3(d) + e2
        d = self.d2(d) + e1
        d = self.d1(d)
        out = self.head(d)
        assert out.shape[-2:] == (h, w), f"resolution invariant violated: {out.shape[-2:]} != {(h, w)}"
        return out
