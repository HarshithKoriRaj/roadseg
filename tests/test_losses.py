import pytest, torch
import torch.nn.functional as F
from roadseg.losses import (ComboLoss, cl_dice_term, dice_loss, soft_cl_dice_loss,
                            soft_skeletonize)


def test_dice_loss_is_zero_for_perfect_prediction():
    t = (torch.rand(2, 1, 32, 32) > 0.5).float()
    logits = torch.where(t > 0, 20.0, -20.0)
    assert dice_loss(logits, t).item() < 1e-3


def test_dice_loss_handles_empty_mask():
    """~95% of crops are background-only; a 0/0 here would poison training."""
    t = torch.zeros(2, 1, 32, 32)
    loss = dice_loss(torch.full((2, 1, 32, 32), -20.0), t)
    assert torch.isfinite(loss) and loss.item() < 1e-2


def test_dice_loss_empty_pred_nonempty_target_is_penalised():
    t = torch.zeros(2, 1, 32, 32); t[:, :, 10:20, 10:20] = 1.0
    loss = dice_loss(torch.full((2, 1, 32, 32), -20.0), t)
    assert loss.item() > 0.9


def test_combo_loss_is_finite_on_empty_and_full_masks():
    loss_fn = ComboLoss(use_cldice=True)
    logits = torch.randn(2, 1, 64, 64)
    for t in (torch.zeros(2, 1, 64, 64), torch.ones(2, 1, 64, 64)):
        assert torch.isfinite(loss_fn(logits, t))


def test_cldice_rejects_logits():
    """The guard that stops the model's logits being passed where probabilities
    are required. Its absence is a silent correctness bug, not a crash."""
    logits = torch.randn(1, 1, 32, 32) * 5
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        soft_cl_dice_loss(logits, torch.zeros(1, 1, 32, 32))


def test_combo_loss_applies_sigmoid_before_cldice():
    """ComboLoss takes logits; it must sigmoid them before the clDice term."""
    loss_fn = ComboLoss(use_cldice=True)
    assert torch.isfinite(loss_fn(torch.randn(1, 1, 64, 64) * 10, torch.zeros(1, 1, 64, 64)))


def test_gradient_flows():
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    ComboLoss()(logits, (torch.rand(2, 1, 32, 32) > 0.5).float()).backward()
    assert logits.grad.abs().sum() > 0


# ---- masking by `valid` (no-data pixels) -------------------------------------

def _batch(seed=0, size=64):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 1, size, size, generator=g) * 2
    target = (torch.rand(2, 1, size, size, generator=g) > 0.7).float()
    valid = (torch.rand(2, 1, size, size, generator=g) > 0.3).float()
    return logits, target, valid


@pytest.mark.parametrize("use_cldice", [False, True])
def test_all_ones_valid_equals_no_valid(use_cldice):
    logits, target, _ = _batch()
    fn = ComboLoss(use_cldice=use_cldice)
    torch.testing.assert_close(fn(logits, target, torch.ones_like(target)),
                               fn(logits, target), atol=1e-6, rtol=0)


@pytest.mark.parametrize("use_cldice", [False, True])
def test_invalid_pixels_get_exactly_zero_gradient(use_cldice):
    logits, target, valid = _batch()
    logits.requires_grad_(True)
    ComboLoss(use_cldice=use_cldice)(logits, target, valid).backward()
    assert torch.all(logits.grad[valid == 0] == 0)
    assert logits.grad[valid == 1].abs().sum() > 0


@pytest.mark.parametrize("use_cldice", [False, True])
def test_target_inside_invalid_region_does_not_change_loss(use_cldice):
    logits, target, valid = _batch()
    flipped = torch.where(valid == 0, 1.0 - target, target)
    fn = ComboLoss(use_cldice=use_cldice)
    torch.testing.assert_close(fn(logits, target, valid), fn(logits, flipped, valid),
                               atol=1e-6, rtol=0)


@pytest.mark.parametrize("use_cldice", [False, True])
def test_fully_invalid_batch_is_finite_with_zero_gradient(use_cldice):
    logits, target, _ = _batch()
    logits.requires_grad_(True)
    loss = ComboLoss(use_cldice=use_cldice)(logits, target, torch.zeros_like(target))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(logits.grad)) and logits.grad.abs().sum() == 0


@pytest.mark.parametrize("use_cldice", [False, True])
def test_bf16_logits_give_fp32_loss_close_to_fp32(use_cldice):
    logits, target, valid = _batch(size=256)
    fn = ComboLoss(use_cldice=use_cldice)
    ref = fn(logits, target, valid)
    got = fn(logits.to(torch.bfloat16), target, valid)
    assert got.dtype == torch.float32
    assert abs(got.item() - ref.item()) / abs(ref.item()) < 1e-2


def test_cldice_gradient_flows_on_nonempty_target():
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.zeros(2, 1, 64, 64); target[:, :, 30:34, 5:60] = 1.0
    ComboLoss(use_cldice=True)(logits, target).backward()
    assert torch.all(torch.isfinite(logits.grad)) and logits.grad.abs().sum() > 0


# ---- W1: R3 = R2 + cldice_weight * (1 - clDice), and nothing else ---------------

def _roads(seed=0, size=64):
    """Logits, a target with two thin roads, and a valid mask with a no-data block."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 1, size, size, generator=g) * 2
    target = torch.zeros(2, 1, size, size)
    target[0, :, 30:33, 4:60] = 1.0
    target[0, :, 6:58, 20:23] = 1.0
    target[1, :, 10:13, :] = 1.0
    valid = torch.ones(2, 1, size, size)
    valid[:, :, 40:, 40:] = 0.0
    return logits, target, valid


@pytest.mark.parametrize("w", [0.5, 1.0, 0.125])
@pytest.mark.parametrize("masked", [False, True])
def test_r3_loss_is_the_r2_loss_plus_the_weighted_cldice_term_exactly(w, masked):
    """The rung R3 differs from R2 by soft-clDice ONLY: no second Dice term rides along."""
    logits, target, valid = _roads()
    valid = valid if masked else None
    prob = torch.sigmoid(logits)
    term = cl_dice_term(prob if valid is None else prob * valid,
                        target if valid is None else target * valid)
    r2 = ComboLoss(use_cldice=False)(logits, target, valid)
    r3 = ComboLoss(use_cldice=True, cldice_weight=w)(logits, target, valid)
    assert r3.item() == (r2 + w * term).item()                      # bit for bit
    assert 0.0 < term.item() < 1.0 and r3.item() > r2.item()
    assert abs((r3 - r2).item() - w * term.item()) < 1e-6


@pytest.mark.parametrize("masked", [False, True])
def test_loss_without_cldice_is_bce_plus_dice_recomputed_by_hand(masked):
    """R0-R2 and the baselines never touch clDice: their loss is numerically what it was."""
    logits, target, valid = _roads(seed=3)
    v = valid if masked else torch.ones_like(target)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * v).sum() / v.sum()
    p, t = torch.sigmoid(logits) * v, target * v
    inter, denom = (p * t).sum((1, 2, 3)), p.sum((1, 2, 3)) + t.sum((1, 2, 3))
    dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
    got = ComboLoss(use_cldice=False)(logits, target, valid if masked else None)
    torch.testing.assert_close(got, bce + dice, rtol=0, atol=1e-6)
    assert ComboLoss()(logits, target).item() == ComboLoss(cldice_weight=7.0)(logits, target).item()


def test_cldice_term_is_one_minus_cldice_of_the_soft_skeletons():
    logits, target, _ = _roads(seed=1)
    prob = torch.sigmoid(logits)
    sp, st = soft_skeletonize(prob, iters=6), soft_skeletonize(target, iters=6)
    dims = (1, 2, 3)
    tprec = (sp * target).sum(dims) / (sp.sum(dims) + 1e-6)
    tsens = (st * prob).sum(dims) / (st.sum(dims) + 1e-6)
    expected = (1 - 2 * tprec * tsens / (tprec + tsens + 1e-6)).mean()     # per sample, then mean
    torch.testing.assert_close(cl_dice_term(prob, target), expected, rtol=0, atol=1e-7)
    # the copied loss = 0.5 * (1 - Dice, eps 1e-6) + 0.5 * this term, per sample
    inter = (prob * target).sum(dims)
    dice = (2 * inter + 1e-6) / (prob.sum(dims) + target.sum(dims) + 1e-6)
    torch.testing.assert_close(soft_cl_dice_loss(prob, target),
                               (0.5 * (1 - dice)).mean() + 0.5 * expected, rtol=0, atol=1e-6)
    perfect = cl_dice_term(target, target)
    assert perfect.item() < 1e-4                                    # a perfect prediction: ~0
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        cl_dice_term(logits * 5, target)                            # logits are refused


def test_cldice_term_on_an_empty_target_is_the_constant_one_with_zero_gradient():
    """A crop with no road on valid pixels has an empty target skeleton. Both topology
    precision and sensitivity are then exactly 0, whatever is predicted: the term is exactly
    1 and its gradient exactly 0 -- it adds no pressure at all, in particular none toward
    more road. On such a crop R3's gradient IS R2's (BCE + Dice), which pushes every logit
    down."""
    g = torch.Generator().manual_seed(5)
    empty = torch.zeros(2, 1, 64, 64)
    for scale, shift in ((2.0, 0.0), (0.1, -8.0), (0.1, 8.0)):       # mixed / nothing / everything
        logits = (torch.randn(2, 1, 64, 64, generator=g) * scale + shift).requires_grad_(True)
        term = cl_dice_term(torch.sigmoid(logits), empty)
        term.backward()
        assert term.item() == 1.0 and torch.all(logits.grad == 0)
    logits = (torch.randn(2, 1, 64, 64, generator=g) * 2).requires_grad_(True)
    grads = []
    for fn in (ComboLoss(use_cldice=True, cldice_weight=0.5), ComboLoss(use_cldice=False)):
        logits.grad = None
        fn(logits, empty).backward()
        grads.append(logits.grad.clone())
    assert torch.equal(grads[0], grads[1])                          # clDice added nothing
    assert torch.all(grads[0] > 0)             # descent lowers every logit: "predict nothing"
    # one empty and one non-empty crop in a batch: the empty one contributes 1/B, constant
    _, target, _ = _roads()
    target[1] = 0
    prob = torch.sigmoid(torch.randn(2, 1, 64, 64, generator=g))
    both = cl_dice_term(prob, target)
    torch.testing.assert_close(both, (cl_dice_term(prob[:1], target[:1]) + 1.0) / 2, atol=1e-7,
                               rtol=0)
