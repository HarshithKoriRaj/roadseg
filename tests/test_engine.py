import pytest, torch, torch.nn as nn
from roadseg.engine import load_checkpoint, save_checkpoint, train_one_epoch
from roadseg.losses import ComboLoss


class _Tiny(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width),
                                 nn.ReLU(), nn.Conv2d(width, 1, 1))

    def forward(self, x):
        return self.net(x)


def _batches(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        y = (torch.rand(2, 1, 16, 16, generator=g) > 0.5).float()
        out.append({"image": torch.randn(2, 3, 16, 16, generator=g), "mask": y,
                    "valid": torch.ones_like(y)})
    return out


def test_checkpoint_preserves_batchnorm_buffers(tmp_path):
    """sam2_road_dac's checkpointer saved only requires_grad params, silently
    dropping BN running stats. That is correct for LoRA and wrong here: eval
    would use uninitialised statistics."""
    m = _Tiny()
    m.train()
    for _ in range(5):
        m(torch.randn(4, 3, 16, 16))
    assert m.net[1].running_mean.abs().sum() > 0
    p = tmp_path / "ck.pt"
    save_checkpoint(p, m, None, None, epoch=1, best=0.5, cfg={})
    m2 = _Tiny()
    load_checkpoint(p, m2)
    torch.testing.assert_close(m2.net[1].running_mean, m.net[1].running_mean)
    torch.testing.assert_close(m2.net[1].running_var, m.net[1].running_var)


def test_checkpoint_roundtrips_optimizer_and_epoch(tmp_path):
    m = _Tiny(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m(torch.randn(2, 3, 16, 16)).sum().backward(); opt.step()
    p = tmp_path / "ck.pt"
    save_checkpoint(p, m, opt, None, epoch=7, best=0.42, cfg={"rung": "R1"})
    m2 = _Tiny(); opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    meta = load_checkpoint(p, m2, opt2)
    assert meta["epoch"] == 7 and meta["best"] == pytest.approx(0.42)
    assert meta["cfg"]["rung"] == "R1"
    assert len(opt2.state) > 0


def test_checkpoint_records_torch_version_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "sub" / "ck.pt"
    save_checkpoint(p, _Tiny(), None, None, epoch=0, best=0.0, cfg={})
    assert torch.load(p, map_location="cpu", weights_only=False)["torch_version"] == torch.__version__
    assert [f.name for f in p.parent.iterdir()] == ["ck.pt"]


def test_checkpoint_architecture_mismatch_fails_loudly(tmp_path):
    p = tmp_path / "ck.pt"
    save_checkpoint(p, _Tiny(width=8), None, None, epoch=1, best=0.0, cfg={})
    with pytest.raises(RuntimeError):
        load_checkpoint(p, _Tiny(width=16))


def test_overfit_one_batch():
    """Integration check: the whole train step actually reduces loss."""
    torch.manual_seed(0)
    m = _Tiny()
    x = torch.randn(2, 3, 32, 32)
    # Labels are a function of the input (per-pixel learnable). i.i.d. random labels
    # cannot be memorised by this 8-channel net, so the brief's version of this test
    # could never pass regardless of engine correctness.
    y = (x[:, :1] > 0).float()
    batch = [{"image": x, "mask": y, "valid": torch.ones_like(y)}]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    first = train_one_epoch(m, batch, ComboLoss(), opt, use_amp=False, device="cpu")
    for _ in range(60):
        last = train_one_epoch(m, batch, ComboLoss(), opt, use_amp=False, device="cpu")
    assert last < first * 0.5, f"loss did not fall: {first:.4f} -> {last:.4f}"


def test_scheduler_stepped_once_per_batch():
    m = _Tiny(); opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0)
    calls = []
    orig = sched.step
    sched.step = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    loader = _batches(4)
    train_one_epoch(m, loader, ComboLoss(), opt, use_amp=False, device="cpu", scheduler=sched)
    assert len(calls) == len(loader)


def test_nan_loss_raises_naming_batch():
    m = _Tiny(); opt = torch.optim.SGD(m.parameters(), lr=0.1)
    loader = _batches(3)
    loader[1]["image"] = torch.full_like(loader[1]["image"], float("nan"))
    with pytest.raises(FloatingPointError, match="batch 1"):
        train_one_epoch(m, loader, ComboLoss(), opt, use_amp=False, device="cpu")


def test_max_batches_limits_batches_run():
    m = _Tiny(); opt = torch.optim.SGD(m.parameters(), lr=0.1)
    seen = []
    def loss_fn(logits, mask, valid):
        seen.append(1)
        return ComboLoss()(logits, mask, valid)
    train_one_epoch(m, _batches(5), loss_fn, opt, use_amp=False, device="cpu", max_batches=2)
    assert len(seen) == 2


def test_max_batches_never_fetches_an_extra_batch():
    """A fetched-and-dropped batch costs a full batch of crops and, worse, advances
    the data stream past what was trained on."""
    m = _Tiny(); opt = torch.optim.SGD(m.parameters(), lr=0.1)
    fetched = []
    def loader():
        for i, b in enumerate(_batches(5)):
            fetched.append(i)
            yield b
    train_one_epoch(m, loader(), ComboLoss(), opt, use_amp=False, device="cpu", max_batches=2)
    assert fetched == [0, 1]


def test_nonfinite_gradient_raises_before_the_weights_are_touched():
    """A finite loss with a NaN gradient would otherwise poison every weight silently."""
    m = _Tiny(); opt = torch.optim.SGD(m.parameters(), lr=0.1)
    before = {k: v.clone() for k, v in m.state_dict().items() if v.is_floating_point()}
    def loss_fn(logits, mask, valid):
        return torch.sqrt(logits.pow(2).sum() * 0.0)       # value 0.0, gradient inf * 0 = NaN
    with pytest.raises(FloatingPointError, match="non-finite gradient at batch 0"):
        train_one_epoch(m, _batches(2), loss_fn, opt, use_amp=False, device="cpu")
    for k, v in before.items():
        if "running_" in k:
            continue                                        # BN statistics move in forward()
        torch.testing.assert_close(m.state_dict()[k], v, rtol=0, atol=0)


def test_on_step_sees_every_step_and_can_stop_the_epoch():
    m = _Tiny(); opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0)
    seen = []
    def on_step(i, loss):
        seen.append((i, sched.last_epoch, loss))            # called AFTER optimizer + scheduler
        return i == 2
    out = train_one_epoch(m, _batches(6), ComboLoss(), opt, use_amp=False, device="cpu",
                          scheduler=sched, on_step=on_step)
    assert [s[0] for s in seen] == [0, 1, 2]
    assert [s[1] for s in seen] == [1, 2, 3]
    assert out == pytest.approx(sum(s[2] for s in seen) / 3)


def test_checkpoint_extra_roundtrips_and_defaults_to_empty(tmp_path):
    p = tmp_path / "ck.pt"
    save_checkpoint(p, _Tiny(), epoch=3, best=0.5, cfg={},
                    extra={"step_in_epoch": 4, "history": [{"epoch": 0, "val_iou": 0.5}]})
    meta = load_checkpoint(p, _Tiny())
    assert meta["extra"] == {"step_in_epoch": 4, "history": [{"epoch": 0, "val_iou": 0.5}]}
    save_checkpoint(p, _Tiny(), epoch=3, best=0.5, cfg={})
    assert load_checkpoint(p, _Tiny())["extra"] == {}


def test_failed_save_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """Killed between writing the temp file and the rename: the old file must survive whole."""
    import roadseg.engine as eng
    p = tmp_path / "ck.pt"
    save_checkpoint(p, _Tiny(), epoch=1, best=0.1, cfg={})
    def boom(src, dst):
        raise KeyboardInterrupt("killed before the rename")
    monkeypatch.setattr(eng.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(p, _Tiny(), epoch=2, best=0.2, cfg={})
    monkeypatch.undo()
    assert load_checkpoint(p, _Tiny())["epoch"] == 1
    save_checkpoint(p, _Tiny(), epoch=3, best=0.3, cfg={})   # a stale temp file does not block
    assert load_checkpoint(p, _Tiny())["epoch"] == 3
    assert [f.name for f in tmp_path.iterdir()] == ["ck.pt"]


def test_cpu_bf16_autocast_is_selected_on_cpu():
    """use_amp on a CPU device runs the forward under CPU bf16 autocast (smoke rehearsal)."""
    seen = []
    class Probe(_Tiny):
        def forward(self, x):
            out = super().forward(x)
            seen.append(out.dtype)
            return out
    m = Probe(); opt = torch.optim.SGD(m.parameters(), lr=0.1)
    train_one_epoch(m, _batches(1), ComboLoss(), opt, use_amp=True, device="cpu")
    train_one_epoch(m, _batches(1), ComboLoss(), opt, use_amp=False, device="cpu")
    assert seen == [torch.bfloat16, torch.float32]


def test_loss_ignores_invalid_pixels_in_train_step():
    """valid must reach the loss: garbage labels under valid==0 cannot change the step."""
    def run(flip):
        torch.manual_seed(0)
        m = _Tiny(); opt = torch.optim.SGD(m.parameters(), lr=0.1)
        b = _batches(1)[0]
        b["valid"][:, :, :8] = 0
        if flip:
            b["mask"][:, :, :8] = 1.0 - b["mask"][:, :, :8]
        return train_one_epoch(m, [b], ComboLoss(), opt, use_amp=False, device="cpu")
    assert run(False) == pytest.approx(run(True), abs=1e-6)
