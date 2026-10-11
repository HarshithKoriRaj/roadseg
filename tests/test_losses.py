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
