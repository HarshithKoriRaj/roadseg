import pytest, torch
from roadseg.losses import ComboLoss, dice_loss, soft_cl_dice_loss


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
