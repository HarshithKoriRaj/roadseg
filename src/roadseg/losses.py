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


def cl_dice_term(pred: torch.Tensor, target: torch.Tensor, iters: int = 6,
                 eps: float = 1e-6) -> torch.Tensor:
    """ONLY the clDice term: per-sample `1 - clDice` from the soft skeletons, mean over the batch.

    `pred` and `target` are (B, 1, H, W) in [0, 1]; `pred` must be post-sigmoid.
    clDice = 2 * Tprec * Tsens / (Tprec + Tsens), with Tprec = |S(pred) * target| / |S(pred)|
    and Tsens = |S(target) * pred| / |S(target)| -- the same quantities, skeleton iterations
    and eps as `soft_cl_dice_loss` above, without that function's extra `0.5 * (1 - Dice)`:
    rung R3 must differ from R2 by clDice alone (ComboLoss already has its Dice term).

    Empty target (a crop with no road on valid pixels). S(target) is all zero, so Tsens = 0;
    and S(pred) * target is all zero, so Tprec = 0 -- both EXACTLY, whatever is predicted.
    Then clDice = 0 / eps = 0: the term is exactly 1, a constant, and its gradient is exactly
    zero. clDice has nothing to say about a crop without roads and adds no pressure there --
    in particular none toward predicting road; BCE and Dice alone push such a crop to
    "predict nothing". (With B samples an empty one contributes the constant 1/B.)
    """
    if pred.min() < 0 or pred.max() > 1:
        raise ValueError("cl_dice_term: pred must be in [0,1] (post-sigmoid)")
    dims = (-1, -2, -3)
    s_pred = soft_skeletonize(pred, iters=iters)
    s_target = soft_skeletonize(target, iters=iters)
    t_prec = (s_pred * target).sum(dim=dims) / (s_pred.sum(dim=dims) + eps)
    t_sens = (s_target * pred).sum(dim=dims) / (s_target.sum(dim=dims) + eps)
    cl_dice = 2 * t_prec * t_sens / (t_prec + t_sens + eps)
    return (1 - cl_dice).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None,
              eps: float = 1.0) -> torch.Tensor:
    """Soft Dice on logits, computed in fp32, optionally restricted to `valid` pixels.

    eps=1.0 (not 1e-6) is deliberate: at ~4% road prevalence most crops are
    entirely background, and a tiny epsilon makes the empty-mask case numerically
    unstable rather than merely near-zero.
    """
    logits = logits.float()
    target = target.float()
    prob = torch.sigmoid(logits)
    if valid is not None:
        valid = valid.float()
        prob = prob * valid
        target = target * valid
    dims = (-1, -2, -3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class ComboLoss(nn.Module):
    """BCE + Dice, optionally + `cldice_weight` * (1 - soft-clDice) (rung R3), masked by `valid`.

    `use_cldice=True` adds `cl_dice_term` and nothing else, so R3 - R2 is clDice alone.
    (Until the final review this called the copied `soft_cl_dice_loss`, whose built-in
    `0.5 * (1 - Dice)` made R3 "R2 + 0.5 clDice + 0.5 extra Dice". The experiment uses
    `cldice_weight: 0.5`, the weight clDice carried there.)

    Takes **logits**. The clDice term needs probabilities, so the sigmoid is
    applied here — callers must never pre-activate. Always computed in fp32
    (bf16 sums over 512x512 pixels would round the Dice denominator).
    `valid` ([B,1,H,W], 1 = use pixel) excludes no-data pixels from every term.
    """

    def __init__(self, use_cldice: bool = False, cldice_weight: float = 1.0) -> None:
        super().__init__()
        self.use_cldice = use_cldice
        self.cldice_weight = cldice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if valid is None:
            bce = bce.mean()
        else:
            valid = valid.float()
            bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
        loss = bce + dice_loss(logits, target, valid)
        if self.use_cldice:
            prob = torch.sigmoid(logits)
            if valid is not None:
                prob, target = prob * valid, target * valid
            loss = loss + self.cldice_weight * cl_dice_term(prob, target)
        return loss


class GeoLoss(nn.Module):
    """Loss of rungs GA and GB: R0's BCE + Dice on the mask, plus
    `centre_weight` * (BCE + Dice) on the centre-line and `width_weight` * cross-entropy of
    the width class at the centre-line pixels (targets: data.geometry.geo_targets).

    Takes the model's `forward(x, aux=True)` dict. For GB the mask logits are logit(drawn
    probability), so the mask term is the same BCE + Dice on the drawn mask. `geo = True`
    tells engine.train_one_epoch to pass the geometric targets. fp32 throughout.
    """

    geo = True

    def __init__(self, centre_weight: float = 1.0, width_weight: float = 1.0) -> None:
        super().__init__()
        self.combo = ComboLoss()
        self.centre_weight, self.width_weight = centre_weight, width_weight

    def forward(self, out: dict, target: torch.Tensor, valid: torch.Tensor | None,
                centre: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        loss = self.combo(out["mask"], target, valid)
        loss = loss + self.centre_weight * self.combo(out["centre"], centre, valid)
        logits = out["width"].float()
        n = (width != -100).sum()
        ce = F.cross_entropy(logits, width, ignore_index=-100, reduction="sum")
        return loss + self.width_weight * ce / n.clamp_min(1)
