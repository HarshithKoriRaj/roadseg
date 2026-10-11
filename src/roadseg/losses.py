"""clDice topology-preservation loss (Shit et al., CVPR 2021).

Differentiable skeleton via iterated soft erosion + morphological closing,
then a Dice-style overlap between predicted skeleton and ground-truth mask
(and vice versa). Encourages connectivity, which is exactly what road
extraction needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    # erosion = min over neighbours = -max-pool of -x
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Differentiable skeletonisation. Input/output in [0, 1]."""
    img = x
    skel = F.relu(img - _soft_open(img))
    for _ in range(iters):
        img = _soft_erode(img)
        opened = _soft_open(img)
        skel = skel + F.relu(img - opened) * (1.0 - skel)
    return skel.clamp(0, 1)


def _soft_dice(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (a * b).sum(dim=(-1, -2, -3))
    return (2 * inter + eps) / (a.sum(dim=(-1, -2, -3)) + b.sum(dim=(-1, -2, -3)) + eps)


def soft_cl_dice_loss(
    pred: torch.Tensor, target: torch.Tensor, iters: int = 6,
) -> torch.Tensor:
    """clDice loss; pred and target are (B, 1, H, W) in [0, 1].

    `pred` MUST be post-sigmoid (a probability map), not raw logits — clDice
    relies on the values being bounded for the soft-skeletonisation to behave.
    """
    if pred.min() < 0 or pred.max() > 1:
        raise ValueError("soft_cl_dice_loss: pred must be in [0,1] (post-sigmoid)")
    s_pred = soft_skeletonize(pred, iters=iters)
    s_target = soft_skeletonize(target, iters=iters)
    t_prec = (s_pred * target).sum(dim=(-1, -2, -3)) / (s_pred.sum(dim=(-1, -2, -3)) + 1e-6)
    t_rec = (s_target * pred).sum(dim=(-1, -2, -3)) / (s_target.sum(dim=(-1, -2, -3)) + 1e-6)
    cl_dice = 2 * t_prec * t_rec / (t_prec + t_rec + 1e-6)
    dice = _soft_dice(pred, target)
    return ((1 - dice) + (1 - cl_dice)).mean() * 0.5


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice on logits.

    eps=1.0 (not 1e-6) is deliberate: at ~4% road prevalence most crops are
    entirely background, and a tiny epsilon makes the empty-mask case numerically
    unstable rather than merely near-zero.
    """
    prob = torch.sigmoid(logits)
    dims = (-1, -2, -3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class ComboLoss(nn.Module):
    """BCE + Dice, optionally + soft-clDice (rung R3).

    Takes **logits**. The clDice term needs probabilities, so the sigmoid is
    applied here — callers must never pre-activate.
    """

    def __init__(self, use_cldice: bool = False, cldice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.use_cldice = use_cldice
        self.cldice_weight = cldice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bce(logits, target) + dice_loss(logits, target)
        if self.use_cldice:
            loss = loss + self.cldice_weight * soft_cl_dice_loss(torch.sigmoid(logits), target)
        return loss
