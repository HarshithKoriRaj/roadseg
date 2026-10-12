"""Training entrypoint: config, naming, schedule, pairing, preemption and crash safety.

Most tests drive `roadseg.train.main` in-process against a tiny synthetic
Massachusetts-layout dataset with a 4-channel stub model, so the whole
train -> resume -> finish -> score pipeline runs in well under a second and can
be killed at every single durable operation. The two `slow` tests run the real
CLI on the real data with a real ResNeXt.
"""
import hashlib
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

import roadseg.data.dataset as dsmod
import roadseg.eval as ev
import roadseg.train as tr
from roadseg.engine import load_checkpoint
from roadseg.stats import aggregate_runs

ROOT = Path(__file__).resolve().parents[1]
REAL_MASS = ROOT / "data" / "massachusetts"
ENV = dict(os.environ, PYTHONPATH=f"{ROOT}/.deps:{ROOT}/src", OMP_NUM_THREADS="2",
           MKL_NUM_THREADS="2", TORCH_HOME="/scratch/hk4488/.cache/torch")
METRICS_KEYS = {"cfg", "threshold", "val", "per_image", "best_epoch", "best_val_iou", "history",
                "n_params", "git_commit", "torch_version", "device_name", "hostname",
                "wall_seconds"}


# ----------------------------------------------------------------------------- helpers

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                                 nn.Conv2d(4, 1, 1))

    def forward(self, x):
        return self.net(x)


def fake_mass(root, n_train=6, n_val=2, n_test=3, size=64):
    """Massachusetts directory layout; roads are brighter than background (learnable)."""
    rng = np.random.default_rng(0)
    for sub, n in (("train", n_train), ("valid", n_val), ("test", n_test)):
        if n == 0:
            continue
        (root / sub).mkdir(parents=True)
        (root / f"{sub}_labels").mkdir()
        for i in range(n):
            m = np.zeros((size, size), np.uint8)
            r, c = rng.integers(8, size - 8, 2)
            m[r - 2:r + 2, :] = 255
            m[:, c - 2:c + 2] = 255
            img = rng.integers(0, 90, (size, size, 3), dtype=np.uint8)
            img[m > 0] += 120
            Image.fromarray(img).save(root / sub / f"{sub}{i}.tiff")
            Image.fromarray(m).save(root / f"{sub}_labels" / f"{sub}{i}.tif")
    return root


def write_cfg(tmp_path, data_root, **over):
    cfg = dict(dataset="massachusetts", data_root=str(data_root), epochs=3, tile=32, stride=16,
               val_stride=32, batch_size=2, lr=0.05, weight_decay=0.0, warmup_epochs=1,
               road_bias=0.75, num_workers=0, eval_workers=0, amp=False, preload=False, seed=0,
               runs_dir=str(tmp_path / "runs"), max_train_batches=4, ckpt_minutes=10)
    cfg.update(over)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    """Stub model for train and eval, plus a fake dataset and a config pointing at it."""
    monkeypatch.setattr(tr, "build_model", lambda cfg: Tiny())
    monkeypatch.setattr(ev, "build_model", lambda cfg: Tiny())
    data = fake_mass(tmp_path / "data")

    class T:
        root = tmp_path
        data_root = data
        runs = tmp_path / "runs"

        @staticmethod
        def cfg(**over):
            return write_cfg(tmp_path, data, **over)

        @staticmethod
        def train(*extra, cfg=None, name=("--rung", "R1")):
            return tr.main(["--config", str(cfg or write_cfg(tmp_path, data)), *name,
                            "--device", "cpu", *extra])

    return T


def read(path):
    return json.loads(Path(path).read_text())


def digest(t: torch.Tensor) -> str:
    return hashlib.sha1(t.detach().cpu().numpy().tobytes()).hexdigest()


def weights_digest(path) -> str:
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    return hashlib.sha1(b"".join(sd[k].numpy().tobytes() for k in sorted(sd))).hexdigest()


def args_for(config, *extra):
    return tr._parser().parse_args(["--config", str(config), *extra])


# ----------------------------------------------------------------------------- config

def test_shipped_configs_resolve_their_base():
    mass = tr.load_config(ROOT / "configs/massachusetts.yaml")
    deep = tr.load_config(ROOT / "configs/deepglobe.yaml")
    assert "_base" not in mass and "_base" not in deep
    assert (mass["dataset"], mass["epochs"]) == ("massachusetts", 100)
    assert (deep["dataset"], deep["epochs"]) == ("deepglobe", 40)
    assert mass["data_root"] == "/scratch/hk4488/roadseg/data/massachusetts"
    assert deep["data_root"] == "/scratch/hk4488/roadseg/data/deepglobe"
    expected = dict(model="roadsegmenter", rung="R1", pretrained=True, hf_init="exact", tile=512,
                    stride=256, val_stride=512, batch_size=16, lr=3.0e-4, weight_decay=1.0e-4,
                    warmup_epochs=2, road_bias=0.75, num_workers=8, amp=True, preload=True,
                    seed=0, runs_dir="/scratch/hk4488/roadseg/runs", ckpt_minutes=10)
    for cfg in (mass, deep):
        assert {k: cfg[k] for k in expected} == expected
        assert isinstance(cfg["lr"], float) and isinstance(cfg["weight_decay"], float)


def test_base_is_overridden_by_the_file_and_the_file_by_the_cli(tmp_path):
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(
        yaml.safe_load((ROOT / "configs/base.yaml").read_text()) | {"lr": 1.0, "seed": 5}))
    (tmp_path / "c.yaml").write_text(
        "_base: b.yaml\ndataset: deepglobe\ndata_root: /x\nepochs: 7\nlr: 2.0\n")
    cfg = tr.load_config(tmp_path / "c.yaml")
    assert (cfg["lr"], cfg["seed"], cfg["epochs"]) == (2.0, 5, 7)        # file beats base
    cfg = tr.resolve_config(args_for(
        tmp_path / "c.yaml", "--seed", "9", "--epochs", "3", "--batch-size", "4", "--workers",
        "0", "--runs-dir", "/r", "--no-amp", "--no-preload", "--no-pretrained",
        "--max-train-batches", "2", "--limit-train", "8", "--limit-val", "1", "--rung", "R3"))
    assert (cfg["seed"], cfg["epochs"], cfg["batch_size"], cfg["num_workers"]) == (9, 3, 4, 0)
    assert (cfg["amp"], cfg["preload"], cfg["pretrained"], cfg["runs_dir"]) == (False,) * 3 + ("/r",)
    assert (cfg["max_train_batches"], cfg["limit_train"], cfg["limit_val"]) == (2, 8, 1)
    assert cfg["eval_workers"] == 0                                       # capped by --workers
    assert (cfg["rung"], cfg["model"], cfg["name"]) == ("R3", "roadsegmenter", "R3")
    assert cfg["lr"] == 2.0                                               # untouched by the CLI


def test_tile_override_keeps_the_stride_geometry():
    cfg = tr.resolve_config(args_for(ROOT / "configs/massachusetts.yaml", "--tile", "256"))
    assert (cfg["tile"], cfg["stride"], cfg["val_stride"]) == (256, 128, 256)
    cfg = tr.resolve_config(args_for(ROOT / "configs/massachusetts.yaml", "--tile", "128"))
    assert (cfg["tile"], cfg["stride"], cfg["val_stride"]) == (128, 64, 128)
    with pytest.raises(ValueError, match="multiple of 32"):
        tr.resolve_config(args_for(ROOT / "configs/massachusetts.yaml", "--tile", "100"))


@pytest.mark.parametrize("flags, expected", [
    (["--config", "massachusetts", "--rung", "R2", "--seed", "1"], "massachusetts_R2_s1"),
    (["--config", "deepglobe", "--model", "unet"], "deepglobe_unet_s0"),
    (["--config", "deepglobe", "--model", "dlinknet", "--seed", "2"], "deepglobe_dlinknet_s2"),
    (["--config", "massachusetts"], "massachusetts_R1_s0"),
])
def test_run_naming(flags, expected):
    flags = list(flags)
    flags[1] = str(ROOT / f"configs/{flags[1]}.yaml")
    cfg = tr.resolve_config(tr._parser().parse_args(flags))
    assert tr.run_name(cfg) == expected
    dataset, name, seed = expected.rsplit("_", 2)
    assert cfg["name"] == name and "_" not in name        # what aggregate_runs cross-checks
    if cfg["model"] != "roadsegmenter":
        assert cfg["rung"] is None                         # a baseline has no rung to mis-record


@pytest.mark.parametrize("flags, match", [
    (["--rung", "R9"], "unknown rung"),
    (["--model", "segformer"], "unknown model"),
    (["--model", "unet", "--rung", "R1"], "roadsegmenter only"),
])
def test_invalid_model_selection_is_rejected(flags, match, capsys):
    with pytest.raises(SystemExit) as e:
        tr.main(["--config", str(ROOT / "configs/deepglobe.yaml"), *flags, "--dry-run"])
    assert e.value.code == 2 and match in capsys.readouterr().err


@pytest.mark.parametrize("name, flags", [
    ("R0", ["--rung", "R0"]), ("R1", ["--rung", "R1"]), ("R2", ["--rung", "R2"]),
    ("R3", ["--rung", "R3"]), ("unet", ["--model", "unet"]), ("dlinknet", ["--model", "dlinknet"]),
])
def test_dry_run_builds_every_rung_and_baseline(name, flags, tmp_path, capsys, monkeypatch):
    """Real models (no weights download), no data: a config typo fails in seconds."""
    def no_data(*a, **k):
        raise AssertionError("--dry-run must not touch any dataset")
    monkeypatch.setattr(tr, "RoadTileDataset", no_data)
    monkeypatch.setattr(tr, "RoadFullDataset", no_data)
    rc = tr.main(["--config", str(ROOT / "configs/deepglobe.yaml"), *flags, "--seed", "3",
                  "--dry-run", "--no-pretrained", "--device", "cpu", "--runs-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    rung = name if name.startswith("R") else "-"
    model = "roadsegmenter" if name.startswith("R") else name
    assert f"rung={rung} seed=3 dataset=deepglobe model={model}" in out
    assert "dry-run ok" in out and "M params" in out
    assert f"use_cldice={name == 'R3'}" in out            # clDice is R3's loss and nobody else's
    assert not list(tmp_path.iterdir())                    # and writes nothing


# ----------------------------------------------------------------------------- LR schedule

def _lrs(warmup_steps, total_steps, lr=3e-4):
    opt = torch.optim.AdamW(Tiny().parameters(), lr=lr)
    sched = tr.build_scheduler(opt, warmup_steps, total_steps)
    out = [opt.param_groups[0]["lr"]]
    for _ in range(total_steps):
        opt.step()
        sched.step()
        out.append(opt.param_groups[0]["lr"])
    return out


def test_lr_schedule_warms_up_per_iteration_then_decays_by_cosine():
    lr = 3e-4
    lrs = _lrs(warmup_steps=2 * 5, total_steps=4 * 5, lr=lr)         # 2 warmup epochs of 5 steps
    assert lrs[0] == pytest.approx(0.01 * lr)
    assert lrs[10] == pytest.approx(lr)
    assert lrs[20] == pytest.approx(0.01 * lr)
    assert all(b > a for a, b in zip(lrs[:10], lrs[1:11]))           # strictly rising warmup
    assert all(b < a for a, b in zip(lrs[10:20], lrs[11:21]))        # strictly falling cosine
    assert max(lrs) == pytest.approx(lr)
    for k in range(10):                                               # linear in the step
        assert lrs[k] == pytest.approx(lr * (0.01 + 0.99 * k / 10))
    for k in range(10, 21):                                           # cosine in the step
        cos = 0.5 * (1 + np.cos(np.pi * (k - 10) / 10))
        assert lrs[k] == pytest.approx(0.01 * lr + 0.99 * lr * cos)


def test_lr_schedule_degenerate_lengths():
    assert _lrs(0, 4, lr=1.0)[0] == pytest.approx(1.0)                # no warmup: cosine only
    assert _lrs(0, 4, lr=1.0)[-1] == pytest.approx(0.01)
    ramp = _lrs(6, 6, lr=1.0)                                         # all warmup (smoke runs)
    assert ramp[0] == pytest.approx(0.01) and ramp[-1] == pytest.approx(1.0)
    assert _lrs(9, 6, lr=1.0)[-1] < 1.0                               # training ends mid-warmup
    with pytest.raises(ValueError):
        _lrs(0, 0)


# ----------------------------------------------------------------------------- pairing

def _stream(data_root, seed, epoch, start=0, steps=3, bs=2, workers=0):
    """The batches train.py's loader yields for the rest of `epoch`, as image digests."""
    base = dsmod.RoadTileDataset(data_root, "massachusetts", "train", tile=32)
    tokens = tr.EpochTokens(steps, bs)
    loader = torch.utils.data.DataLoader(
        tr.SeededCrops(base, seed), batch_size=bs, sampler=tokens, drop_last=True,
        num_workers=workers, worker_init_fn=tr._worker_init if workers else None,
        generator=torch.Generator().manual_seed(seed))
    tokens.set(epoch, start)
    return [digest(torch.cat([b["image"].flatten(), b["mask"].flatten(), b["valid"].flatten()]))
            for b in loader]


def test_crop_stream_depends_only_on_seed_and_position(tmp_path):
    root = fake_mass(tmp_path / "d")
    torch.manual_seed(0)
    ref = _stream(root, seed=0, epoch=1)
    assert len(ref) == 3 and len(set(ref)) == 3

    torch.manual_seed(12345)
    torch.randn(1000)                                   # "another rung built a different model"
    assert _stream(root, seed=0, epoch=1) == ref
    assert _stream(root, seed=0, epoch=1, start=2) == ref[2:]        # mid-epoch resume
    assert _stream(root, seed=0, epoch=1, start=3) == []             # nothing left to run
    assert set(_stream(root, seed=0, epoch=2)).isdisjoint(ref)       # new crops every epoch
    assert set(_stream(root, seed=1, epoch=1)).isdisjoint(ref)       # and for every seed


def test_crop_stream_is_the_same_with_worker_processes(tmp_path):
    root = fake_mass(tmp_path / "d")
    assert _stream(root, 0, 1, workers=2) == _stream(root, 0, 1, workers=0)


def test_crop_stream_with_persistent_workers_over_several_epochs(tmp_path):
    """The production loader shape: persistent workers, the sampler re-aimed before each
    pass, and each pass consumed with islice exactly as train_one_epoch does (so the
    iterator is never run to StopIteration). Includes a pass resumed at step 1."""
    root = fake_mass(tmp_path / "d")

    def run(workers):
        base = dsmod.RoadTileDataset(root, "massachusetts", "train", tile=32)
        tokens = tr.EpochTokens(3, 2)
        loader = torch.utils.data.DataLoader(
            tr.SeededCrops(base, 0), batch_size=2, sampler=tokens, drop_last=True,
            num_workers=workers, persistent_workers=workers > 0,
            worker_init_fn=tr._worker_init if workers else None,
            generator=torch.Generator().manual_seed(0))
        out = []
        for epoch, start in ((0, 0), (1, 0), (2, 1), (3, 0)):
            tokens.set(epoch, start)
            out.append([digest(torch.cat([b["image"].flatten(), b["mask"].flatten()]))
                        for b in itertools.islice(loader, 3 - start)])
        return out

    single = run(0)
    assert [len(e) for e in single] == [3, 3, 2, 3]
    assert len({d for e in single for d in e}) == 11
    assert run(2) == single


def test_seeded_crops_leaves_the_global_rng_alone(tmp_path):
    ds = tr.SeededCrops(dsmod.RoadTileDataset(fake_mass(tmp_path / "d"), "massachusetts",
                                              "train", tile=32), seed=0)
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    ds[5]
    torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)


REAL_TRAIN_ONE_EPOCH = tr.train_one_epoch
REAL_BUILD_INDEX = dsmod.build_index


def _record_training(monkeypatch):
    """Record the RNG state at model construction, the initial weights and every batch."""
    rec = {"rng_at_build": [], "init": [], "batches": [], "splits": []}
    real_train, real_index = REAL_TRAIN_ONE_EPOCH, REAL_BUILD_INDEX   # never wrap a wrapper

    def build(cfg):
        rec["rng_at_build"].append(torch.get_rng_state().clone())
        draws = {"R0": 0, "R1": 300, "R2": 300, "R3": 300}.get(cfg["name"], 77)
        torch.randn(draws)                              # rungs consume different amounts of RNG
        m = Tiny()
        rec["init"].append(digest(torch.cat([p.flatten() for p in m.parameters()])))
        return m

    def train_one_epoch(model, loader, *a, **k):
        def tap():
            for b in loader:
                rec["batches"].append(digest(torch.cat([b["image"].flatten(), b["mask"].flatten(),
                                                        b["valid"].flatten()])))
                yield b
        return real_train(model, tap(), *a, **k)

    def index(root, dataset, split):
        rec["splits"].append(split)
        return real_index(root, dataset, split)

    monkeypatch.setattr(tr, "build_model", build)
    monkeypatch.setattr(tr, "train_one_epoch", train_one_epoch)
    monkeypatch.setattr(dsmod, "build_index", index)
    return rec


def test_rungs_and_baselines_are_paired_for_a_seed(tiny, monkeypatch):
    """Same seed -> identical crop sequence for every rung and baseline, whatever each
    model's construction drew from the RNG; a different seed -> a different sequence."""
    streams = {}
    for name in ("R0", "R1", "R2", "R3", "unet", "dlinknet"):
        rec = _record_training(monkeypatch)
        torch.manual_seed(hash(name) % 1000)            # arbitrary RNG history before main()
        flag = ("--rung", name) if name.startswith("R") else ("--model", name)
        assert tiny.train(name=flag) == 0
        streams[name] = rec["batches"]
        assert len(rec["batches"]) == 3 * 4             # epochs x steps, none fetched and dropped
        assert len(set(rec["batches"])) == 12
    assert all(s == streams["R0"] for s in streams.values())

    rec = _record_training(monkeypatch)
    assert tiny.train("--seed", "1") == 0
    assert set(rec["batches"]).isdisjoint(streams["R0"])


def test_model_construction_is_seeded_immediately_before_build_model(tiny, monkeypatch):
    rec = _record_training(monkeypatch)
    torch.manual_seed(999)
    torch.randn(50)
    assert tiny.train() == 0
    torch.manual_seed(0)
    expected = torch.get_rng_state()
    assert torch.equal(rec["rng_at_build"][0], expected)         # nothing drew in between
    first = rec["init"][0]

    rec = _record_training(monkeypatch)
    torch.manual_seed(4)
    assert tiny.train(name=("--rung", "R2")) == 0                # same seed, other rung, other history
    assert torch.equal(rec["rng_at_build"][0], expected)
    rec = _record_training(monkeypatch)
    assert tiny.train("--seed", "1") == 0
    assert not torch.equal(rec["rng_at_build"][0], expected) and rec["init"][0] != first


# ----------------------------------------------------------------------------- fit: preemption

class Flag:
    def __init__(self):
        self.on = False

    def __bool__(self):
        return self.on


def _fit_parts(lr=0.1, epochs=2, steps=10):
    torch.manual_seed(0)
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = tr.build_scheduler(opt, steps, epochs * steps)
    return model, opt, sched


def _batches(epoch, start, steps=10):
    out = []
    for s in range(start, steps):
        g = torch.Generator().manual_seed(1000 * epoch + s)
        x = torch.randn(2, 3, 16, 16, generator=g)
        y = (x[:, :1] > 0).float()
        out.append({"image": x, "mask": y, "valid": torch.ones_like(y)})
    return out


class RecLoss:
    """ComboLoss that records the LR each step ran at and can raise a flag at step N."""

    def __init__(self, opt, flag=None, raise_at=None):
        self.opt, self.flag, self.raise_at = opt, flag, raise_at
        self.lrs = []
        self.loss = tr.ComboLoss()

    def __call__(self, logits, y, v):
        self.lrs.append(self.opt.param_groups[0]["lr"])
        if self.raise_at is not None and len(self.lrs) == self.raise_at:
            self.flag.on = True                                   # arrives DURING this step
        return self.loss(logits, y, v)


def _run_fit(run, parts, state, preempt, loss, *, epochs=2, steps=10, ckpt_seconds=1e9,
             saved=False, validate=None, batches=None):
    model, opt, sched = parts
    cfg = {"epochs": epochs, "lr": 0.1}
    vals = iter(np.linspace(0.1, 0.9, 50))
    done = tr.fit(model=model, optimizer=opt, scheduler=sched, loss_fn=loss,
                  batches_for=batches or (lambda e, s: _batches(e, s, steps)),
                  validate=validate or (lambda m: float(next(vals))),
                  run=run, cfg=cfg, state=state, steps_per_epoch=steps, use_amp=False,
                  device="cpu", preempt=preempt, ckpt_seconds=ckpt_seconds, saved=saved,
                  log=lambda *a, **k: None)
    return done, cfg


def test_preempt_after_step_3_checkpoints_and_resume_runs_exactly_the_other_7(tmp_path):
    """A5 unit test: flag raised during step 3 of a 10-step epoch."""
    # uninterrupted reference
    ref_parts = _fit_parts()
    ref_loss = RecLoss(ref_parts[1])
    ref_state = tr.fresh_state()
    done, _ = _run_fit(tmp_path / "ref", ref_parts, ref_state, Flag(), ref_loss)
    assert done and len(ref_loss.lrs) == 20 and ref_parts[2].last_epoch == 20

    # interrupted run
    run = tmp_path / "run"
    parts = _fit_parts()
    flag = Flag()
    loss = RecLoss(parts[1], flag, raise_at=3)
    state = tr.fresh_state()
    done, cfg = _run_fit(run, parts, state, flag, loss)
    assert done is False                                           # reports preemption
    assert len(loss.lrs) == 3 and parts[2].last_epoch == 3
    ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["extra"]["step_in_epoch"] == 3 and ck["extra"]["epoch"] == 0
    assert ck["extra"]["history"] == [] and "optimizer" in ck and "scheduler" in ck
    assert not (run / "best.pt").exists() and not (run / "history.json").exists()

    # resume in a "new process": fresh objects, state only from the file
    parts2 = _fit_parts(lr=0.1)
    with torch.no_grad():
        for p in parts2[0].parameters():
            p.add_(1.0)                                            # must be overwritten by the load
    state2 = tr.load_state(run / "latest.pt", *parts2, cfg, 10)
    assert (state2["epoch"], state2["step_in_epoch"], state2["resumes"]) == (0, 3, 1)
    loss2 = RecLoss(parts2[1])
    counts = []
    done, _ = _run_fit(run, parts2, state2, Flag(), loss2, saved=True,
                       batches=lambda e, s: counts.append((e, s)) or _batches(e, s))
    assert done is True
    assert counts == [(0, 3), (1, 0)]                              # the REMAINING steps of epoch 0
    assert len(loss2.lrs) == 7 + 10                                # exactly 7 more, then epoch 1
    assert parts2[2].last_epoch == ref_parts[2].last_epoch == 20   # schedule ends where it should
    assert loss.lrs + loss2.lrs == ref_loss.lrs                    # every step at the exact same LR
    for a, b in zip(parts2[0].state_dict().values(), ref_parts[0].state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)           # and the same weights, bitwise
    rows = state2["history"]
    assert [r["epoch"] for r in rows] == [0, 1]
    assert rows[0]["resumed"] is True and rows[0]["train_steps"] == 7 and "resumed" not in rows[1]
    assert [r["lr"] for r in rows] == [r["lr"] for r in ref_state["history"]]
    assert read(run / "history.json") == rows


def test_preempt_during_the_last_step_resumes_straight_into_validation(tmp_path):
    run = tmp_path / "run"
    parts = _fit_parts(epochs=1, steps=4)
    flag = Flag()
    loss = RecLoss(parts[1], flag, raise_at=4)
    state = tr.fresh_state()
    done, cfg = _run_fit(run, parts, state, flag, loss, epochs=1, steps=4)
    assert done is False and state["step_in_epoch"] == 4 and state["epoch"] == 0

    parts2 = _fit_parts(epochs=1, steps=4)
    state2 = tr.load_state(run / "latest.pt", *parts2, cfg, 4)
    loss2 = RecLoss(parts2[1])
    done, _ = _run_fit(run, parts2, state2, Flag(), loss2, epochs=1, steps=4, saved=True)
    assert done is True and loss2.lrs == [] and parts2[2].last_epoch == 4
    assert state2["history"] == [dict(state2["history"][0], epoch=0, train_loss=None,
                                      resumed=True, train_steps=0)]


def test_preempt_during_validation_stops_at_the_epoch_boundary_without_a_second_save(tmp_path,
                                                                                    monkeypatch):
    run = tmp_path / "run"
    parts = _fit_parts(epochs=2, steps=3)
    flag = Flag()
    saves = []
    real = tr.save_state

    def spy(*a, **k):
        saves.append(1)
        return real(*a, **k)

    monkeypatch.setattr(tr, "save_state", spy)

    def validate(m):
        flag.on = True                                             # SIGTERM lands in validation
        return 0.5

    state = tr.fresh_state()
    done, cfg = _run_fit(run, parts, state, flag, RecLoss(parts[1]), epochs=2, steps=3,
                         validate=validate)
    assert done is False and (state["epoch"], state["step_in_epoch"]) == (1, 0)
    assert len(saves) == 1                                         # the end-of-epoch save only
    assert read(run / "history.json")[0]["val_iou"] == 0.5 and (run / "best.pt").exists()
    ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["extra"]
    assert (ck["epoch"], ck["step_in_epoch"], ck["best_epoch"]) == (1, 0, 0)


def test_preempt_before_the_first_step_and_after_the_last_epoch(tmp_path):
    flag = Flag()
    flag.on = True
    parts = _fit_parts(epochs=1, steps=2)
    state = tr.fresh_state()
    done, cfg = _run_fit(tmp_path / "a", parts, state, flag, RecLoss(parts[1]), epochs=1, steps=2)
    assert done is False and parts[2].last_epoch == 0
    assert torch.load(tmp_path / "a" / "latest.pt", weights_only=False)["extra"]["step_in_epoch"] == 0

    parts = _fit_parts(epochs=1, steps=2)
    flag = Flag()

    def validate(m):
        flag.on = True
        return 0.5

    state = tr.fresh_state()
    done, _ = _run_fit(tmp_path / "b", parts, state, flag, RecLoss(parts[1]), epochs=1, steps=2,
                       validate=validate)
    assert done is False and state["epoch"] == 1                   # all epochs done, finish pending
    done, _ = _run_fit(tmp_path / "b", parts, state, Flag(), RecLoss(parts[1]), epochs=1, steps=2)
    assert done is True and len(state["history"]) == 1             # nothing is retrained


def test_mid_epoch_checkpoints_follow_the_wall_clock(tmp_path, monkeypatch):
    saved_at = []
    real = tr.save_state

    def spy(path, model, opt, sched, cfg, state, spe):
        saved_at.append((state["epoch"], state["step_in_epoch"]))
        return real(path, model, opt, sched, cfg, state, spe)

    monkeypatch.setattr(tr, "save_state", spy)
    now = [0.0]
    parts = _fit_parts(epochs=1, steps=6)

    class Tick(RecLoss):
        def __call__(self, *a):
            now[0] += 60.0                                         # every step takes a minute
            return super().__call__(*a)

    model, opt, sched = parts
    tr.fit(model=model, optimizer=opt, scheduler=sched, loss_fn=Tick(opt),
           batches_for=lambda e, s: _batches(e, s, 6), validate=lambda m: 0.3,
           run=tmp_path / "r", cfg={"epochs": 1}, state=tr.fresh_state(), steps_per_epoch=6,
           use_amp=False, device="cpu", preempt=Flag(), ckpt_seconds=150.0,
           clock=lambda: now[0], log=lambda *a, **k: None)
    assert saved_at == [(0, 3), (0, 6), (1, 0)]       # every 150 s inside the epoch, then its end


def test_a_short_loader_is_an_error_not_a_silently_shorter_schedule(tmp_path):
    parts = _fit_parts(epochs=1, steps=5)
    with pytest.raises(RuntimeError, match="yielded 3 batches, expected 5"):
        _run_fit(tmp_path / "r", parts, tr.fresh_state(), Flag(), RecLoss(parts[1]), epochs=1,
                 steps=5, batches=lambda e, s: _batches(e, s, 3))


def test_resume_refuses_a_checkpoint_from_another_configuration(tmp_path):
    run = tmp_path / "run"
    parts = _fit_parts(epochs=1, steps=4)
    flag = Flag()
    _, cfg = _run_fit(run, parts, tr.fresh_state(), flag, RecLoss(parts[1], flag, 2), epochs=1,
                      steps=4)
    with pytest.raises(SystemExit, match="epochs: 1 -> 2"):
        tr.load_state(run / "latest.pt", *_fit_parts(), {**cfg, "epochs": 2}, 4)
    with pytest.raises(SystemExit, match="steps_per_epoch: 4 -> 5"):
        tr.load_state(run / "latest.pt", *_fit_parts(), cfg, 5)
    free = {**cfg, "num_workers": 3, "runs_dir": "/elsewhere", "preload": False, "ckpt_minutes": 1}
    state = tr.load_state(run / "latest.pt", *_fit_parts(epochs=1, steps=4), free, 4)
    assert state["step_in_epoch"] == 2


# ----------------------------------------------------------------------------- end to end (tiny)

def test_end_to_end_writes_everything_once_and_is_idempotent(tiny, capsys):
    assert tiny.train() == 0
    run = tiny.runs / "massachusetts_R1_s0"
    assert sorted(p.name for p in run.iterdir()) == ["best.pt", "history.json", "latest.pt",
                                                     "metrics.json"]
    m = read(run / "metrics.json")
    assert METRICS_KEYS <= set(m)
    cfg = m["cfg"]
    assert (cfg["dataset"], cfg["model"], cfg["rung"], cfg["name"], cfg["seed"], cfg["tile"]) == (
        "massachusetts", "roadsegmenter", "R1", "R1", 0, 32)
    assert cfg["data_root"] == str(tiny.data_root)
    hist = read(run / "history.json")
    assert m["history"] == hist and [r["epoch"] for r in hist] == [0, 1, 2]
    assert all(set(r) == {"epoch", "train_loss", "val_iou", "lr", "seconds"} for r in hist)
    assert m["best_val_iou"] == max(r["val_iou"] for r in hist) > 0
    assert hist[m["best_epoch"]]["val_iou"] == m["best_val_iou"]
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == m["best_epoch"] and best["best"] == m["best_val_iou"]
    assert 0.2 <= m["threshold"] <= 0.8 and len(m["per_image"]) == m["val"]["n_images"] == 2
    assert "apls" in m["per_image"][0] and "cl_dice" in m["per_image"][0]
    assert m["device_name"] == "cpu" and m["torch_version"] == torch.__version__
    assert len(m["git_commit"]) == 40 and m["n_params"] == sum(p.numel() for p in Tiny().parameters())
    assert m["resumes"] == 0 and m["steps_per_epoch"] == 4

    before = (run / "metrics.json").stat().st_mtime_ns
    capsys.readouterr()
    assert tiny.train() == 0                                       # the same command again
    assert "already complete" in capsys.readouterr().out
    assert tiny.train("--resume", "auto") == 0                     # and as a requeued job
    assert "already complete" in capsys.readouterr().out
    assert (run / "metrics.json").stat().st_mtime_ns == before

    small = ["--tile", "32", "--stride", "16", "--workers", "0"]
    assert ev.main(["--run", str(run), "--split", "test", *small]) == 0
    t = read(run / "test_metrics.json")
    assert t["threshold"] == m["threshold"] and t["cfg"] == cfg and len(t["per_image"]) == 3
    assert t["eval"]["checkpoint_epoch"] == m["best_epoch"]
    assert ev.main(["--run", str(run), "--split", "test", *small]) == 2      # scored once


def test_runs_aggregate_by_directory_name(tiny):
    """train -> eval -> stats: the run directory and cfg['name'] agree the way
    aggregate_runs requires, for a rung and for a baseline."""
    small = ["--tile", "32", "--stride", "16", "--workers", "0"]
    for flag in (("--rung", "R0"), ("--rung", "R1"), ("--model", "unet")):
        for seed in ("0", "1"):
            assert tiny.train("--seed", seed, name=flag) == 0
            run = tiny.runs / f"massachusetts_{flag[1]}_s{seed}"
            assert ev.main(["--run", str(run), "--split", "test", *small]) == 0
    with pytest.raises(ValueError, match="baseline"):                # tile 32 is not the protocol
        aggregate_runs(tiny.runs, "massachusetts", n_boot=50)
    out = aggregate_runs(tiny.runs, "massachusetts", n_boot=50, allow_nonstandard=True)
    assert sorted(out["rungs"]) == ["R0", "R1", "unet"]
    assert all(r["seeds"] == [0, 1] for r in out["rungs"].values())
    assert sorted(out["vs_baseline"]) == ["R1", "unet"]


def test_training_never_opens_the_test_split(tmp_path, monkeypatch):
    """Train where the test split does not exist at all, and log every index built."""
    monkeypatch.setattr(ev, "build_model", lambda cfg: Tiny())
    data = fake_mass(tmp_path / "data", n_test=0)
    rec = _record_training(monkeypatch)
    cfg = write_cfg(tmp_path, data)
    assert tr.main(["--config", str(cfg), "--rung", "R1", "--device", "cpu"]) == 0
    assert set(rec["splits"]) == {"train", "val"}
    src = Path(tr.__file__).read_text()
    assert '"test"' not in src and "'test'" not in src            # no code path names the split

    # ...and the scorer cannot run before checkpoint and threshold are frozen:
    empty = tmp_path / "runs" / "massachusetts_R2_s0"
    empty.mkdir()
    rec["splits"].clear()
    with pytest.raises(FileNotFoundError, match="metrics.json"):
        ev.main(["--run", str(empty), "--split", "test", "--workers", "0"])
    assert rec["splits"] == []


def test_metrics_are_frozen_before_test_scoring_and_never_rewritten(tiny):
    assert tiny.train() == 0
    run = tiny.runs / "massachusetts_R1_s0"
    frozen = (run / "metrics.json").read_bytes()
    best = (run / "best.pt").read_bytes()
    assert ev.main(["--run", str(run), "--split", "test", "--tile", "32", "--stride", "16",
                    "--workers", "0"]) == 0
    assert tiny.train("--resume", "auto") == 0
    assert (run / "metrics.json").read_bytes() == frozen and (run / "best.pt").read_bytes() == best


def test_refuses_to_mix_runs_in_one_directory(tiny, capsys):
    assert tiny.train(cfg=tiny.cfg(epochs=2)) == 0
    with pytest.raises(SystemExit, match="different configuration.*epochs: 2 -> 3"):
        tiny.train(cfg=tiny.cfg(epochs=3))                         # finished with other settings
    with pytest.raises(SystemExit, match="different configuration.*lr"):
        tiny.train("--resume", "auto", cfg=tiny.cfg(epochs=2, lr=0.01))
    assert tiny.train("--workers", "0", "--resume", "auto", cfg=tiny.cfg(epochs=2)) == 0

    run = tiny.runs / "massachusetts_R1_s0"
    (run / "metrics.json").unlink()                                # unfinished run on disk
    with pytest.raises(SystemExit, match="refusing to start fresh"):
        tiny.train(cfg=tiny.cfg(epochs=2))                         # no --resume: would clobber it
    with pytest.raises(SystemExit, match="no such checkpoint"):
        tiny.train("--resume", str(tiny.root / "nope.pt"), cfg=tiny.cfg(epochs=2))
    # what a SIGKILL in the middle of publishing metrics.json leaves behind
    (run / ".metrics.json.4242.tmp").write_text('{"cfg": {"trunc')
    (run / ".history.json.tmp").write_text("[{")
    (run / "latest.tmp").write_bytes(b"torn")
    assert tiny.train("--resume", "auto", cfg=tiny.cfg(epochs=2)) == 0   # straight to the finish
    out = capsys.readouterr().out
    assert "resumed from" in out and "at epoch 2 step 0" in out
    assert read(run / "metrics.json")["resumes"] == 1
    assert sorted(p.name for p in run.iterdir()) == ["best.pt", "history.json", "latest.pt",
                                                     "metrics.json"]      # debris cleared


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGUSR1])
def test_a_real_signal_checkpoints_exits_75_and_resumes(tiny, monkeypatch, capsys, signum):
    """The real handlers, in this process: the signal arrives during step 6 (epoch 1, step 2)."""
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR1)}
    real = tr.ComboLoss
    calls = []

    class Loss(real):
        def forward(self, *a):
            calls.append(1)
            if len(calls) == 6:
                os.kill(os.getpid(), signum)
            return super().forward(*a)

    monkeypatch.setattr(tr, "ComboLoss", Loss)
    assert tiny.train() == tr.EXIT_REQUEUE == 75
    assert "preempted: checkpoint saved at epoch 1 step 2" in capsys.readouterr().out
    assert {s: signal.getsignal(s) for s in before} == before        # handlers put back
    run = tiny.runs / "massachusetts_R1_s0"
    extra = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["extra"]
    assert (extra["epoch"], extra["step_in_epoch"]) == (1, 2) and len(extra["history"]) == 1
    assert not (run / "metrics.json").exists() and len(read(run / "history.json")) == 1

    assert tiny.train("--resume", "auto") == 0
    assert "at epoch 1 step 2" in capsys.readouterr().out
    assert len(calls) == 3 * 4                                       # no step repeated or skipped
    hist = read(run / "history.json")
    assert [r["epoch"] for r in hist] == [0, 1, 2]
    assert [r.get("resumed", False) for r in hist] == [False, True, False]
    assert hist[1]["train_steps"] == 2 and read(run / "metrics.json")["resumes"] == 1


# ----------------------------------------------------------------------------- kill anywhere

class Kill(BaseException):
    """Stands in for SIGKILL: not an Exception, so nothing in the code under test catches it."""


class Faults:
    """Counts every durable operation (fsync / rename / link) and every unit of work
    (optimizer step, validation pass, threshold search); raises Kill before the k-th.

    Killed at an fsync, the file being written is truncated to half first: a torn write.
    """

    def __init__(self, monkeypatch):
        self.n, self.kill_at, self.events, self.published = 0, None, [], []
        real = {k: getattr(os, k) for k in ("fsync", "replace", "link")}
        loss, evaluate, select = tr.ComboLoss, tr.evaluate, tr.select_threshold
        ev_evaluate = ev.evaluate

        def fsync(fd):
            if self.n == self.kill_at:
                os.ftruncate(fd, os.fstat(fd).st_size // 2)
            self.tick("fsync")
            return real["fsync"](fd)

        def replace(src, dst):
            self.tick(f"replace:{Path(dst).name}")
            return real["replace"](src, dst)

        def link(src, dst):
            self.tick(f"link:{Path(dst).name}")
            real["link"](src, dst)
            self.published.append(Path(dst).name)

        class Loss(loss):
            def forward(inner, *a):
                self.tick("step")
                return super().forward(*a)

        def wrap(name, fn):
            def inner(*a, **k):
                self.tick(name)
                return fn(*a, **k)
            return inner

        for k, f in (("fsync", fsync), ("replace", replace), ("link", link)):
            monkeypatch.setattr(os, k, f)
        monkeypatch.setattr(tr, "ComboLoss", Loss)
        monkeypatch.setattr(tr, "evaluate", wrap("evaluate", evaluate))
        monkeypatch.setattr(tr, "select_threshold", wrap("select_threshold", select))
        monkeypatch.setattr(ev, "evaluate", wrap("score", ev_evaluate))

    def tick(self, what):
        k = self.n
        self.n += 1
        if k == self.kill_at:
            raise Kill(f"killed before event {k}: {what}")
        self.events.append(what)

    def arm(self, kill_at):
        self.n, self.kill_at, self.events, self.published = 0, kill_at, [], []


def assert_nothing_partial_is_visible(run: Path):
    """Whatever is on disk under a result name is complete and loadable."""
    if not run.exists():
        return
    for p in run.iterdir():
        if p.name.startswith(".") or p.suffix == ".tmp":
            continue                                   # temp names are never read by anything
        assert p.name in {"latest.pt", "best.pt", "history.json", "metrics.json",
                          "test_metrics.json"}, p.name
        if p.suffix == ".json":
            json.loads(p.read_text())
        else:
            torch.load(p, map_location="cpu", weights_only=False)["model"]
    if (run / "metrics.json").exists():                # published => everything it names exists
        m = read(run / "metrics.json")
        assert torch.load(run / "best.pt", weights_only=False)["epoch"] == m["best_epoch"]


def test_kill_at_every_point_then_resume_reaches_the_uninterrupted_result(tiny, monkeypatch):
    """Requirement 1. Kill before EVERY durable operation and unit of work of a whole run
    (mid-epoch, between epochs, in validation, in the finish phase, mid-write of every
    file), resume with --resume auto, and compare with the run that was never interrupted."""
    faults = Faults(monkeypatch)
    cfg = tiny.cfg(epochs=3, max_train_batches=3, ckpt_minutes=0)   # latest.pt after every step
    run = tiny.runs / "massachusetts_R1_s0"
    small = ["--tile", "32", "--stride", "16", "--workers", "0"]

    faults.arm(None)
    assert tiny.train(cfg=cfg) == 0
    n_train_events = faults.n
    ref_events = list(faults.events)
    assert faults.published == ["metrics.json"]
    assert ev.main(["--run", str(run), "--split", "test", *small]) == 0
    n_events = faults.n
    ref = {"metrics": read(run / "metrics.json"), "history": read(run / "history.json"),
           "best": weights_digest(run / "best.pt"), "latest": weights_digest(run / "latest.pt"),
           "test": read(run / "test_metrics.json")}
    # the sweep must actually cross the windows that matter
    assert ref_events.count("replace:best.pt") >= 2, "best.pt must be replaced mid-run"
    assert ref_events.count("step") == 9 and ref_events.count("replace:latest.pt") >= 9 + 3
    assert {"select_threshold", "link:metrics.json", "replace:history.json",
            "score", "link:test_metrics.json"} <= set(ref_events + faults.events)
    assert n_events > n_train_events > 40

    def strip(m):
        """Everything but timings and the bookkeeping of the interruption itself, as text
        (so NaN entries compare equal)."""
        m = json.loads(json.dumps(m))
        for k in ("wall_seconds", "resumes"):
            m.pop(k)
        for r in m["history"]:
            for k in ("seconds", "train_loss", "resumed", "train_steps"):
                r.pop(k, None)
        return json.dumps(m, sort_keys=True)

    def scores(t):
        return json.dumps([t["aggregate"], t["per_image"], t["threshold"]], sort_keys=True)

    for k in range(n_events):
        shutil.rmtree(tiny.runs)
        faults.arm(k)
        killed_in = "train"
        try:
            assert tiny.train(cfg=cfg) == 0
            killed_in = "score"
            ev.main(["--run", str(run), "--split", "test", *small])
            raise AssertionError(f"event {k} never happened")
        except Kill as e:
            what = str(e)
        assert_nothing_partial_is_visible(run)
        if killed_in == "train":
            assert not (run / "metrics.json").exists() or "metrics.json" in faults.published, what
        assert not (run / "test_metrics.json").exists(), what
        published = list(faults.published)
        resumes = int(killed_in == "train" and (run / "latest.pt").exists())

        faults.arm(None)                                   # the requeued job
        assert tiny.train("--resume", "auto", cfg=cfg) == 0, what
        for stale in run.glob(".test_metrics.json.*.tmp"):  # the SLURM wrapper does this
            stale.unlink()
        assert ev.main(["--run", str(run), "--split", "test", *small]) == 0, what
        assert ev.main(["--run", str(run), "--split", "test", *small]) == 2, what
        published += faults.published

        assert published.count("metrics.json") == 1, what           # written exactly once
        assert published.count("test_metrics.json") == 1, what      # scored exactly once
        assert sorted(p.name for p in run.iterdir()) == [
            "best.pt", "history.json", "latest.pt", "metrics.json", "test_metrics.json"], what
        assert_nothing_partial_is_visible(run)
        m, hist = read(run / "metrics.json"), read(run / "history.json")
        assert [r["epoch"] for r in hist] == [0, 1, 2] and m["history"] == hist, what
        assert sum(r.get("resumed", False) for r in hist) <= 1, what
        # not just the same shape: the same numbers, checkpoint and test scores
        assert strip(m) == strip(ref["metrics"]), what              # strip() returns text
        assert [r["lr"] for r in hist] == [r["lr"] for r in ref["history"]], what
        assert weights_digest(run / "best.pt") == ref["best"], what
        assert weights_digest(run / "latest.pt") == ref["latest"], what
        assert scores(read(run / "test_metrics.json")) == scores(ref["test"]), what
        assert m["resumes"] == resumes, what


def test_a_second_kill_during_the_resume_is_also_harmless(tiny, monkeypatch):
    """Preempted again while resuming -- including inside the best.pt repair and the
    history rewrite that a resume starts with -- and then left alone: same result."""
    faults = Faults(monkeypatch)
    cfg = tiny.cfg(epochs=3, max_train_batches=3, ckpt_minutes=0)
    run = tiny.runs / "massachusetts_R1_s0"
    faults.arm(None)
    assert tiny.train(cfg=cfg) == 0
    events = list(faults.events)
    ref = (weights_digest(run / "best.pt"), weights_digest(run / "latest.pt"),
           [(r["epoch"], r["val_iou"], r["lr"]) for r in read(run / "history.json")],
           read(run / "metrics.json")["threshold"])
    best_renames = [i for i, e in enumerate(events) if e == "replace:best.pt"]
    pairs = [(k1, k2) for k1 in best_renames for k2 in range(8)]             # repair interrupted
    pairs += [(k1, k2) for k1 in range(2, len(events), 9) for k2 in (3, 11)]  # arbitrary pairs
    for k1, k2 in pairs:
        shutil.rmtree(tiny.runs)
        published = []
        for kill_at, flags in ((k1, ()), (k2, ("--resume", "auto"))):
            faults.arm(kill_at)
            try:
                tiny.train(*flags, cfg=cfg)
            except Kill:
                pass
            published += faults.published
            assert_nothing_partial_is_visible(run)
        faults.arm(None)
        assert tiny.train("--resume", "auto", cfg=cfg) == 0, (k1, k2)
        published += faults.published
        assert published.count("metrics.json") == 1, (k1, k2)
        assert sorted(p.name for p in run.iterdir()) == ["best.pt", "history.json", "latest.pt",
                                                         "metrics.json"], (k1, k2)
        m = read(run / "metrics.json")
        got = (weights_digest(run / "best.pt"), weights_digest(run / "latest.pt"),
               [(r["epoch"], r["val_iou"], r["lr"]) for r in read(run / "history.json")],
               m["threshold"])
        assert got == ref, (k1, k2)
        assert m["history"] == read(run / "history.json"), (k1, k2)


def test_best_pt_is_repaired_when_killed_between_latest_and_best(tiny, monkeypatch, capsys):
    """The one window where the two checkpoints can disagree, exercised explicitly."""
    faults = Faults(monkeypatch)
    cfg = tiny.cfg(epochs=2, max_train_batches=3)
    run = tiny.runs / "massachusetts_R1_s0"
    faults.arm(None)
    assert tiny.train(cfg=cfg) == 0
    events = list(faults.events)
    ref = weights_digest(run / "best.pt"), read(run / "metrics.json")["best_epoch"]
    assert ref[1] == 1, "fixture must improve in the last epoch"
    k = len(events) - 1 - events[::-1].index("replace:best.pt")     # the rename of the final best

    shutil.rmtree(tiny.runs)
    faults.arm(k)
    with pytest.raises(Kill):
        tiny.train(cfg=cfg)
    stale = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["extra"]
    assert stale["epoch"] == 0 and latest["best_epoch"] == 1 and latest["epoch"] == 2
    faults.arm(None)
    capsys.readouterr()
    assert tiny.train("--resume", "auto", cfg=cfg) == 0
    assert "repaired best.pt from latest.pt (epoch 1; was 0)" in capsys.readouterr().out
    assert (weights_digest(run / "best.pt"), read(run / "metrics.json")["best_epoch"]) == ref


# ----------------------------------------------------------------------------- SLURM scripts

ARRAY = ROOT / "scripts" / "sbatch_train_array.sh"


def _print_map():
    r = subprocess.run(["bash", str(ARRAY), "--print-map"], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin"}, cwd="/")      # no SLURM, no project env
    assert r.returncode == 0, r.stderr
    rows = [line.split() for line in r.stdout.strip().splitlines()]
    return [(int(i), dataset, name, int(seed), run) for i, dataset, name, seed, run in rows]


def test_print_map_covers_the_grid_exactly_once():
    rows = _print_map()
    assert [r[0] for r in rows] == list(range(36))
    assert len({r[4] for r in rows}) == 36
    assert all(r[4] == f"{r[1]}_{r[2]}_s{r[3]}" for r in rows)
    datasets, seeds = ("massachusetts", "deepglobe"), (0, 1, 2)
    ladder = sorted((d, n, s) for _, d, n, s, _ in rows[:24])
    assert ladder == sorted((d, n, s) for d in datasets for n in ("R0", "R1", "R2", "R3")
                            for s in seeds)
    baselines = sorted((d, n, s) for _, d, n, s, _ in rows[24:])
    assert baselines == sorted((d, n, s) for d in datasets for n in ("unet", "dlinknet")
                               for s in seeds)
    # the documented formula, index by index
    for i, d, n, s, _ in rows[:24]:
        assert (s, n, d) == (i % 3, ("R0", "R1", "R2", "R3")[(i // 3) % 4], datasets[i // 12])
    for i, d, n, s, _ in rows[24:]:
        j = i - 24
        assert (s, n, d) == (j % 3, ("unet", "dlinknet")[(j // 3) % 2], datasets[j // 6])


def test_print_map_names_match_train_py():
    """The wrapper computes the run directory itself (to score it); it must agree with train.py."""
    for i, dataset, name, seed, run in _print_map():
        flag = ["--rung", name] if name.startswith("R") else ["--model", name]
        cfg = tr.resolve_config(args_for(ROOT / f"configs/{dataset}.yaml", *flag, "--seed",
                                         str(seed)))
        assert tr.run_name(cfg) == run


def _directives(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#SBATCH "):
            k, _, v = line[len("#SBATCH "):].strip().partition("=")
            out[k] = v
    return out


def test_sbatch_directives():
    d = _directives(ARRAY)
    assert d["--array"] == "0-35%16" and d["--partition"] == "l40s_public,h200_public"
    assert d["--gres"] == "gpu:1" and d["--account"] == "torch_pr_627_general"
    assert d["--signal"] == "B:USR1@180" and "--requeue" in d and d["--open-mode"] == "append"
    assert (d["--cpus-per-task"], d["--mem"], d["--time"]) == ("8", "80G", "24:00:00")
    assert not any(k.startswith("--qos") for k in d)
    agg = _directives(ROOT / "scripts/sbatch_aggregate.sh")
    assert (agg["--partition"], agg["--cpus-per-task"], agg["--mem"], agg["--time"]) == (
        "cpu_short", "4", "16G", "01:00:00")
    smoke = _directives(ROOT / "scripts/sbatch_smoke_cpu.sh")
    assert (smoke["--partition"], smoke["--cpus-per-task"], smoke["--mem"], smoke["--time"]) == (
        "cpu_short", "16", "64G", "03:00:00")
    for script in ("sbatch_train_array.sh", "sbatch_aggregate.sh", "sbatch_smoke_cpu.sh"):
        r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], capture_output=True,
                           text=True)
        assert r.returncode == 0, r.stderr


FAKE_PYTHON = r'''#!{python}
"""Stands in for `python -m roadseg.train|eval` under the SLURM wrapper (no torch import)."""
import json, os, signal, sys, time
log, spec = os.environ["FAKE_LOG"], json.loads(os.environ["FAKE_SPEC"])
mod = sys.argv[sys.argv.index("-m") + 1].rsplit(".", 1)[-1]
def note(msg):
    with open(log, "a") as f:
        f.write(msg + "\n")
note(mod + " argv " + " ".join(sys.argv[1:]))
s = spec.get(mod, {{}})
if "touch" in s:
    run = sys.argv[sys.argv.index("--run") + 1]
    open(os.path.join(run, s["touch"]), "w").write("{{}}")
if s.get("block"):                      # a long step that honours SIGTERM the way train.py does
    got = []
    if s.get("handle", True):
        signal.signal(signal.SIGTERM, lambda n, f: got.append(n))
    open(log + "." + mod + ".ready", "w").close()
    t0 = time.time()
    while not got and time.time() - t0 < 60:
        time.sleep(0.02)
    note(mod + (" got SIGTERM" if got else " timed out"))
    sys.exit(s.get("rc_on_term", 75) if got else 99)
sys.exit(s.get("rc", 0))
'''


def _wrapper(tmp_path, spec, task=4, signal_step=None, signum=signal.SIGUSR1, env_extra=None):
    """Run the REAL array script with a stub python and a stub scontrol; optionally signal it
    while `signal_step` (train | eval) is running. Returns (exit code, log lines, stdout)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "python"
    fake.write_text(FAKE_PYTHON.format(python=sys.executable))
    fake.chmod(0o755)
    scontrol = bin_dir / "scontrol"
    scontrol.write_text('#!/bin/bash\necho "scontrol $*" >> "$FAKE_LOG"\nexit ${FAKE_SCONTROL_RC:-0}\n')
    scontrol.chmod(0o755)
    log = tmp_path / "log"
    log.unlink(missing_ok=True)
    runs = tmp_path / "runs"
    (runs / "massachusetts_R1_s1").mkdir(parents=True, exist_ok=True)       # task 4
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "FAKE_LOG": str(log), "FAKE_SPEC": json.dumps(spec),
           "ROADSEG_PY": str(fake), "ROADSEG_RUNS": str(runs), "ROADSEG_LOGS": str(tmp_path / "logs"),
           "SLURM_ARRAY_TASK_ID": str(task), "SLURM_ARRAY_JOB_ID": "777", "SLURM_JOB_ID": "781",
           **(env_extra or {})}
    p = subprocess.Popen(["bash", str(ARRAY)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env)
    if signal_step:
        ready = Path(f"{log}.{signal_step}.ready")
        t0 = time.time()
        while not ready.exists():
            assert p.poll() is None and time.time() - t0 < 60, "step never started"
            time.sleep(0.02)
        ready.unlink()
        p.send_signal(signum)                          # to the batch shell only, like --signal=B:
    out, _ = p.communicate(timeout=120)
    lines = log.read_text().splitlines() if log.exists() else []
    return p.returncode, lines, out


def test_wrapper_runs_train_then_scores_the_test_split(tmp_path):
    rc, log, out = _wrapper(tmp_path, {})
    assert rc == 0, out
    run = tmp_path / "runs" / "massachusetts_R1_s1"
    assert log == [
        f"train argv -m roadseg.train --config configs/massachusetts.yaml --rung R1 --seed 1 "
        f"--resume auto --runs-dir {tmp_path / 'runs'}",
        f"eval argv -m roadseg.eval --run {run} --split test --workers 8"]
    assert "task 4 -> dataset=massachusetts name=R1 seed=1" in out
    rc, log, out = _wrapper(tmp_path, {}, task=33)                 # a baseline uses --model
    assert rc == 0, out
    assert "--config configs/deepglobe.yaml --model dlinknet --seed 0" in log[0]
    assert log[1].startswith(f"eval argv -m roadseg.eval --run {tmp_path / 'runs'}/deepglobe_dlinknet_s0 ")


def test_wrapper_forwards_usr1_to_python_and_requeues_on_75(tmp_path):
    """Requirement 5: bash defers traps while a foreground child runs, so the step runs in
    the background under `wait`; USR1 to the shell must reach python as SIGTERM."""
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True}}, signal_step="train")
    assert rc == 0, out
    assert log[1:] == ["train got SIGTERM", "scontrol requeue 777_4"], out   # and no eval
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True}}, signal_step="train",
                            signum=signal.SIGTERM)
    assert rc == 0 and log[1:] == ["train got SIGTERM", "scontrol requeue 777_4"], out
    # outside an array: fall back to the plain job id
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True}}, signal_step="train",
                            env_extra={"SLURM_ARRAY_JOB_ID": ""})
    assert rc == 0 and log[-1] == "scontrol requeue 781", out


def test_wrapper_requeues_when_python_dies_of_the_signal_before_its_handlers_exist(tmp_path):
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True, "handle": False}},
                            signal_step="train")
    assert rc == 0 and log[1:] == ["scontrol requeue 777_4"], out   # killed by SIGTERM: rc 143


def test_wrapper_requeues_instead_of_scoring_when_signalled_as_training_finishes(tmp_path):
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True, "rc_on_term": 0}},
                            signal_step="train")
    assert rc == 0 and log[1:] == ["train got SIGTERM", "scontrol requeue 777_4"], out


def test_wrapper_requeues_when_test_scoring_is_interrupted(tmp_path):
    rc, log, out = _wrapper(tmp_path, {"eval": {"block": True, "rc_on_term": 143}},
                            signal_step="eval")
    assert rc == 0 and log[2:] == ["eval got SIGTERM", "scontrol requeue 777_4"], out


def test_wrapper_exit_codes(tmp_path):
    rc, log, out = _wrapper(tmp_path, {"train": {"rc": 3}})
    assert rc == 3 and len(log) == 1, out                           # failed training: no scoring
    rc, log, out = _wrapper(tmp_path, {"train": {"rc": 75}})        # preempted without our trap
    assert rc == 0 and log[-1] == "scontrol requeue 777_4", out
    rc, log, out = _wrapper(tmp_path, {"eval": {"rc": 2, "touch": "test_metrics.json"}})
    assert rc == 0 and "already scored" in out, out                 # the once-only refusal
    (tmp_path / "runs" / "massachusetts_R1_s1" / "test_metrics.json").unlink()
    rc, log, out = _wrapper(tmp_path, {"eval": {"rc": 2}})
    assert rc == 2, out                              # exit 2 WITHOUT a result is a real failure
    rc, log, out = _wrapper(tmp_path, {"eval": {"rc": 1}})
    assert rc == 1, out
    rc, log, out = _wrapper(tmp_path, {"train": {"rc": 75}}, env_extra={"FAKE_SCONTROL_RC": "1"})
    assert rc != 0, out                              # could not requeue: do not report success


def test_wrapper_clears_stale_scoring_temp_files(tmp_path):
    run = tmp_path / "runs" / "massachusetts_R1_s1"
    run.mkdir(parents=True)
    (run / ".test_metrics.json.4242.tmp").write_text("{")
    rc, log, out = _wrapper(tmp_path, {})
    assert rc == 0 and not list(run.glob(".test_metrics.json.*.tmp")), out


# ----------------------------------------------------------------------------- real CLI

def test_cli_dry_run_subprocess():
    r = subprocess.run([sys.executable, "-m", "roadseg.train", "--config",
                        str(ROOT / "configs/deepglobe.yaml"), "--rung", "R2", "--seed", "3",
                        "--dry-run", "--no-pretrained", "--device", "cpu"],
                       capture_output=True, text=True, env=ENV, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "rung=R2" in r.stdout and "seed=3" in r.stdout and "dataset=deepglobe" in r.stdout
    assert r.stderr == ""


def test_cli_rejects_unknown_rung_subprocess():
    r = subprocess.run([sys.executable, "-m", "roadseg.train", "--config",
                        str(ROOT / "configs/deepglobe.yaml"), "--rung", "R9", "--dry-run"],
                       capture_output=True, text=True, env=ENV, cwd=ROOT)
    assert r.returncode != 0 and "unknown rung" in r.stderr


def _memory_limit_gib() -> float:
    """Smallest cgroup-v2 memory.max above this process (inf if none)."""
    limit = float("inf")
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().splitlines()[-1].split("::", 1)[1]
        node = Path("/sys/fs/cgroup") / rel.lstrip("/")
        while str(node).startswith("/sys/fs/cgroup"):
            f = node / "memory.max"
            if f.is_file() and f.read_text().strip() != "max":
                limit = min(limit, int(f.read_text()) / 2 ** 30)
            node = node.parent
    except (OSError, IndexError, ValueError):
        pass
    return limit


needs_data = pytest.mark.skipif(not (REAL_MASS / "train").is_dir(),
                                reason="Massachusetts data not present")
# A ResNeXt-50 U-Net with AdamW state needs ~2 GB next to the pytest process. The login node
# caps a whole user session at 3 GiB and OOM-kills past it; run these on a compute node
# (sbatch scripts/sbatch_tests.sh tests/test_train_cli.py -m slow).
needs_ram = pytest.mark.skipif(
    _memory_limit_gib() < 8,
    reason=f"needs ~4 GiB; this session's cgroup allows {_memory_limit_gib():.1f} GiB "
           f"(run on a compute node)")
SMALL_TRAIN = ["--rung", "R1", "--no-pretrained", "--limit-train", "4", "--limit-val", "1",
               "--batch-size", "2", "--workers", "0", "--tile", "128", "--device", "cpu",
               "--no-amp"]


def _cli(module, *args, timeout=1500):
    return subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True,
                          env=ENV, cwd=ROOT, timeout=timeout)


class Flat(nn.Module):
    """One trainable bias: predicts 'no road' everywhere, so the metrics stay cheap."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x):
        return self.bias + 0.0 * x[:, :1]


@pytest.mark.slow
@needs_data
def test_real_data_pipeline_with_a_stub_model(tmp_path, monkeypatch, capsys):
    """Real Massachusetts imagery (1500x1500 TIFFs, real no-data regions, the real val and
    test splits) through train -> finish -> test scoring, in-process with a one-parameter
    model: light enough for the login node, where the real-model tests below cannot run."""
    rec = _record_training(monkeypatch)
    monkeypatch.setattr(tr, "build_model", lambda cfg: Flat())
    monkeypatch.setattr(ev, "build_model", lambda cfg: Flat())
    args = ["--config", str(ROOT / "configs/massachusetts.yaml"), "--rung", "R1", "--epochs", "2",
            "--max-train-batches", "3", "--limit-train", "4", "--limit-val", "1", "--batch-size",
            "2", "--workers", "0", "--tile", "128", "--device", "cpu", "--no-amp", "--runs-dir",
            str(tmp_path)]
    assert tr.main(args) == 0
    assert set(rec["splits"]) == {"train", "val"}                   # preload=true decoded these
    run = tmp_path / "massachusetts_R1_s0"
    m = read(run / "metrics.json")
    assert METRICS_KEYS <= set(m) and len(read(run / "history.json")) == 2
    assert (m["cfg"]["tile"], m["cfg"]["stride"], m["cfg"]["val_stride"]) == (128, 64, 128)
    assert (m["n_train_images"], m["n_val_images"], m["steps_per_epoch"]) == (4, 1, 3)
    row = m["per_image"][0]
    assert 0 < row["n_valid"] <= 1500 * 1500 and row["n_gt"] > 0 and row["n_pred"] == 0
    assert {"apls", "cl_dice", "iou"} <= set(row)
    capsys.readouterr()
    assert tr.main(args) == 0 and "already complete" in capsys.readouterr().out

    score = ["--run", str(run), "--split", "test", "--tile", "128", "--stride", "64", "--limit",
             "1", "--workers", "0"]
    assert ev.main(score) == 0
    t = read(run / "test_metrics.json")
    assert t["protocol"] == "nonstandard" and len(t["per_image"]) == 1
    assert t["per_image"][0]["id"] != row["id"]                     # a test image, not the val one
    assert ev.main(score) == 2


BOOT = '''"""The real `python -m roadseg.train` in a real process, with the 4-channel stub model."""
import runpy, sys, time
import torch.nn as nn
import roadseg.models.segmenter as seg

DELAY = float(sys.argv[1])


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                                 nn.Conv2d(4, 1, 1))

    def forward(self, x):
        if self.training:
            time.sleep(DELAY)                   # stretch an optimizer step so it can be hit
        return self.net(x)


seg.build_model = lambda cfg: Tiny()
sys.argv = sys.argv[2:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
'''


@pytest.mark.slow
def test_a_real_sigkill_mid_epoch_then_resume_matches_the_uninterrupted_run(tmp_path):
    """The crash sweep above simulates the kill with an exception. This is the real thing:
    SIGKILL (no handler, no cleanup) to a real training process in the middle of an epoch."""
    boot = tmp_path / "boot.py"
    boot.write_text(BOOT)
    cfg = write_cfg(tmp_path, fake_mass(tmp_path / "data"), epochs=2, max_train_batches=6,
                    ckpt_minutes=0)
    env = dict(ENV, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def cmd(delay, runs):
        return [sys.executable, str(boot), str(delay), "roadseg.train", "--config", str(cfg),
                "--rung", "R1", "--device", "cpu", "--runs-dir", str(tmp_path / runs),
                "--resume", "auto"]

    def result(runs):
        run = tmp_path / runs / "massachusetts_R1_s0"
        m = read(run / "metrics.json")
        return (weights_digest(run / "best.pt"), weights_digest(run / "latest.pt"),
                [(r["epoch"], r["val_iou"], r["lr"]) for r in m["history"]], m["threshold"],
                json.dumps(m["val"], sort_keys=True))

    r = subprocess.run(cmd(0, "ref"), capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert read(tmp_path / "ref/massachusetts_R1_s0/metrics.json")["resumes"] == 0

    run = tmp_path / "killed" / "massachusetts_R1_s0"
    p = subprocess.Popen(cmd(0.5, "killed"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env, cwd=ROOT)
    t0 = time.time()
    while not (run / "latest.pt").exists():                 # written after the first step
        assert p.poll() is None, p.communicate()
        assert time.time() - t0 < 600
        time.sleep(0.05)
    time.sleep(0.7)                                         # somewhere inside step 2 or 3
    p.kill()
    p.communicate()
    assert p.returncode == -signal.SIGKILL
    assert_nothing_partial_is_visible(run)
    extra = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["extra"]
    assert extra["epoch"] == 0 and 0 < extra["step_in_epoch"] < 6
    assert not (run / "metrics.json").exists() and not (run / "history.json").exists()

    r = subprocess.run(cmd(0, "killed"), capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"at epoch 0 step {extra['step_in_epoch']}" in r.stdout
    m = read(run / "metrics.json")
    assert m["resumes"] == 1 and m["history"][0]["resumed"] is True
    assert m["history"][0]["train_steps"] == 6 - extra["step_in_epoch"]
    assert result("killed") == result("ref")                # bitwise the same model and scores
    assert_nothing_partial_is_visible(run)


# ----------------------------------------------------------------------------- G1: every exit ends

FAULT_BOOT = '''"""The real `python -m roadseg.train` in a real process, stub model, one injected event."""
import multiprocessing, os, runpy, signal, sys
import torch.nn as nn
import roadseg.eval as ev
import roadseg.losses as losses
import roadseg.models.segmenter as seg

FAULT, MARK = sys.argv[1], sys.argv[2]


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                                 nn.Conv2d(4, 1, 1))

    def forward(self, x):
        return self.net(x)


def mark():
    """Leave the pids of the worker processes alive at the moment of the event."""
    with open(MARK, "w") as f:
        f.write(" ".join(str(p.pid) for p in multiprocessing.active_children()))


real_forward, calls = losses.ComboLoss.forward, []


def forward(self, *a):
    calls.append(1)
    if len(calls) == 2:                                   # optimizer step 1 of epoch 0
        if FAULT == "loss":
            mark()
            raise RuntimeError("injected: the loss failed at step 1")
        if FAULT == "sigterm":                            # job-wide, as SLURM delivers it
            mark()
            os.killpg(os.getpgid(0), signal.SIGTERM)
    return real_forward(self, *a)


def select_threshold(*a, **k):
    mark()
    raise RuntimeError("injected: the finish phase failed")


seg.build_model = lambda cfg: Tiny()
losses.ComboLoss.forward = forward
if FAULT == "finish":
    ev.select_threshold = select_threshold
sys.argv = sys.argv[3:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
'''


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _train_with_fault(tmp_path, fault, limit=30.0):
    """Run the real CLI with 2 persistent DataLoader workers in its own process group and
    inject `fault`. Returns (exit code, stdout, stderr, seconds from the event to the exit,
    worker pids alive at the event, whether any process of the group outlived the parent)."""
    boot = tmp_path / "fault_boot.py"
    boot.write_text(FAULT_BOOT)
    data = tmp_path / "data"
    if not data.exists():
        fake_mass(data)
    cfg = write_cfg(tmp_path, data, epochs=2, max_train_batches=3)
    mark = tmp_path / f"{fault}.mark"
    mark.unlink(missing_ok=True)
    p = subprocess.Popen([sys.executable, str(boot), fault, str(mark), "roadseg.train", "--config",
                          str(cfg), "--rung", "R1", "--device", "cpu", "--workers", "2",
                          "--resume", "auto"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=dict(ENV, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                         cwd=ROOT, start_new_session=True)
    try:
        t0 = time.time()
        while fault != "none" and not mark.exists() and p.poll() is None:
            assert time.time() - t0 < 900, "the injected event never happened"
            time.sleep(0.02)
        t_event = time.time()
        try:
            out, err = p.communicate(timeout=limit if fault != "none" else 900)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, err = p.communicate()
            pytest.fail(f"still running {limit:.0f} s after the '{fault}' event (a hang):\n"
                        f"{out}\n{err}")
        seconds = time.time() - t_event
        deadline = time.time() + 5
        while _group_alive(p.pid) and time.time() < deadline:
            time.sleep(0.05)
        left_behind = _group_alive(p.pid)
    finally:
        if _group_alive(p.pid):
            os.killpg(p.pid, signal.SIGKILL)
    workers = mark.read_text().split() if mark.exists() else []
    return p.returncode, out, err, seconds, workers, left_behind


@pytest.mark.parametrize("fault, message", [
    ("loss", "injected: the loss failed at step 1"),
    ("finish", "injected: the finish phase failed"),
])
def test_a_crash_with_live_workers_exits_1_promptly_and_leaves_no_process(tmp_path, fault, message):
    """G1. DataLoader workers ignore SIGTERM, and Python's exit joins them without a timeout:
    an uncaught exception used to hang the process (and with it the GPU) forever."""
    rc, out, err, seconds, workers, left_behind = _train_with_fault(tmp_path, fault)
    assert len(workers) == 2, (workers, out, err)          # they WERE alive when it failed
    assert rc == tr.EXIT_CRASHED == 1, out + err
    assert message in err and "Traceback" in err           # the reason is in the log
    assert seconds < 30 and not left_behind
    run = tmp_path / "runs" / "massachusetts_R1_s0"
    assert not (run / "metrics.json").exists()

    rc, out, err, _, _, left_behind = _train_with_fault(tmp_path, "none")   # the next attempt
    assert rc == 0 and not left_behind, out + err
    if fault == "finish":
        assert "at epoch 2 step 0" in out                  # straight to the finish phase
    assert read(run / "metrics.json")["best_epoch"] is not None


def test_a_job_wide_sigterm_with_live_workers_exits_75_and_leaves_no_process(tmp_path):
    rc, out, err, seconds, workers, left_behind = _train_with_fault(tmp_path, "sigterm")
    assert len(workers) == 2, (workers, out, err)
    assert rc == tr.EXIT_REQUEUE == 75, out + err
    assert "preempted: checkpoint saved at epoch 0 step 2" in out
    assert seconds < 30 and not left_behind
    rc, out, err, _, _, left_behind = _train_with_fault(tmp_path, "none")
    assert rc == 0 and not left_behind and "at epoch 0 step 2" in out, out + err
    assert read(tmp_path / "runs" / "massachusetts_R1_s0" / "metrics.json")["resumes"] == 1


RUN_CLI = '''import sys, time, multiprocessing, signal
from roadseg import runtime as rt

def stubborn():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)          # like a DataLoader worker
    time.sleep(600)

def main(argv):
    multiprocessing.Process(target=stubborn, daemon=True).start()
    time.sleep(0.3)
    how = sys.argv[1]
    if how == "return": return 0
    if how == "none": return None
    if how == "requeue": return rt.EXIT_REQUEUE
    if how == "refused": raise rt.Refused("the run directory is locked")
    if how == "preflight": raise rt.PreflightFailed("no CUDA device")
    if how == "message": raise SystemExit("no epoch produced a finite validation IoU")
    if how == "usage": raise SystemExit(2)
    raise ValueError("boom")

rt.run_cli(main)
'''


@pytest.mark.parametrize("how, code, stderr", [
    ("return", 0, ""), ("none", 0, ""), ("requeue", 75, ""),
    ("refused", 2, "refused: the run directory is locked"),
    ("preflight", 3, "preflight failed: no CUDA device"),
    ("message", 1, "no epoch produced a finite validation IoU"), ("usage", 2, ""),
    ("crash", 1, "ValueError: boom"),
])
def test_run_cli_maps_every_ending_to_the_exit_code_contract(tmp_path, how, code, stderr):
    """The codes the SLURM wrapper acts on, each with a SIGTERM-ignoring child alive at exit."""
    script = tmp_path / "cli.py"
    script.write_text(RUN_CLI)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, str(script), how], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env=ENV, start_new_session=True)
    try:
        _, err = p.communicate(timeout=60)
        assert p.returncode == code, err
        assert stderr in err and (stderr or err == "")
        assert time.time() - t0 < 60 and not _group_alive(p.pid)     # the child went with it
    finally:
        if _group_alive(p.pid):
            os.killpg(p.pid, signal.SIGKILL)
    assert (tr.EXIT_OK, tr.EXIT_CRASHED, tr.EXIT_REFUSED, tr.EXIT_PREFLIGHT, tr.EXIT_REQUEUE) == (
        0, 1, 2, 3, 75)


@pytest.mark.slow
@needs_data
@needs_ram
def test_real_data_end_to_end(tmp_path):
    """Real ResNeXt, real Massachusetts images, the real CLI: train -> idempotent rerun ->
    score the test split once."""
    cmd = ["--config", str(ROOT / "configs/massachusetts.yaml"), *SMALL_TRAIN, "--epochs", "1",
           "--max-train-batches", "2", "--runs-dir", str(tmp_path)]
    r = _cli("roadseg.train", *cmd)
    assert r.returncode == 0, r.stdout + r.stderr
    run = tmp_path / "massachusetts_R1_s0"
    m = read(run / "metrics.json")
    assert METRICS_KEYS <= set(m)
    assert (m["cfg"]["name"], m["cfg"]["tile"], m["cfg"]["stride"]) == ("R1", 128, 64)
    assert m["cfg"]["data_root"] == str(REAL_MASS) and m["n_params"] > 20e6
    assert len(read(run / "history.json")) == 1 and len(m["per_image"]) == 1
    assert (run / "best.pt").exists() and (run / "latest.pt").exists()

    r = _cli("roadseg.train", *cmd)
    assert r.returncode == 0 and "already complete" in r.stdout

    score = ["--run", str(run), "--split", "test", "--tile", "128", "--stride", "64", "--limit",
             "1", "--workers", "0"]
    r = _cli("roadseg.eval", *score)
    assert r.returncode == 0, r.stdout + r.stderr
    t = read(run / "test_metrics.json")
    assert t["protocol"] == "nonstandard" and t["eval"]["limit"] == 1 and len(t["per_image"]) == 1
    assert t["threshold"] == m["threshold"]
    r = _cli("roadseg.eval", *score)
    assert r.returncode == 2 and "already exists" in r.stderr


@pytest.mark.slow
@needs_data
@needs_ram
def test_real_data_sigterm_mid_epoch_then_resume(tmp_path):
    """A5 (2): SIGTERM the real CLI mid-epoch -> exit 75 with a mid-epoch latest.pt;
    --resume auto finishes the run."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"_base: {ROOT / 'configs/massachusetts.yaml'}\nckpt_minutes: 0\n")
    cmd = ["--config", str(cfg), *SMALL_TRAIN, "--epochs", "1", "--max-train-batches", "8",
           "--runs-dir", str(tmp_path / "runs"), "--resume", "auto"]
    run = tmp_path / "runs" / "massachusetts_R1_s0"
    p = subprocess.Popen([sys.executable, "-m", "roadseg.train", *cmd], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env=ENV, cwd=ROOT)
    t0 = time.time()
    while not (run / "latest.pt").exists():            # ckpt_minutes 0: appears after step 1
        assert p.poll() is None, p.communicate()
        assert time.time() - t0 < 900, "no checkpoint after 15 min"
        time.sleep(0.1)
    p.send_signal(signal.SIGTERM)
    out, err = p.communicate(timeout=600)
    assert p.returncode == 75, out + err
    assert "preempted: checkpoint saved at epoch 0 step" in out
    extra = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["extra"]
    assert extra["epoch"] == 0 and 0 < extra["step_in_epoch"] < 8, extra
    assert not (run / "metrics.json").exists()

    r = _cli("roadseg.train", *cmd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"at epoch 0 step {extra['step_in_epoch']}" in r.stdout
    hist = read(run / "history.json")
    assert len(hist) == 1 and hist[0]["resumed"] is True
    assert hist[0]["train_steps"] == 8 - extra["step_in_epoch"]
    m = read(run / "metrics.json")
    assert m["resumes"] == 1 and m["history"] == hist
