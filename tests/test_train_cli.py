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
import socket
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
ENV.pop("SLURM_RESTART_COUNT", None)     # H4: a restarted job retries a held lock for 3 minutes
METRICS_KEYS = {"cfg", "threshold", "val", "per_image", "best_epoch", "best_val_iou", "history",
                "n_params", "git_commit", "torch_version", "device_name", "hostname",
                "wall_seconds", "code_commit", "code_dirty", "processes", "code_changes"}
# A finished run directory holds exactly these (G2: the lock file; G7: the saved threshold).
RUN_FILES = [".lock", "best.pt", "history.json", "latest.pt", "metrics.json", "threshold.json"]


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
                    seed=0, runs_dir="/scratch/hk4488/roadseg/runs",
                    ckpt_minutes=5)       # G7: the kill grace is 30 s; bound the loss by time
    assert tr._DEFAULTS["ckpt_minutes"] == 5
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
    assert sorted(p.name for p in run.iterdir()) == RUN_FILES
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
    assert m["code_commit"] == m["git_commit"] and m["code_changes"] == []
    assert [(p["code_commit"], p["first_epoch"], p["last_epoch"], p["device_name"])
            for p in m["processes"]] == [(m["code_commit"], 0, 2, "cpu")]    # one process did it all
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
    assert sorted(p.name for p in run.iterdir()) == RUN_FILES             # debris cleared


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
                          "threshold.json", "test_metrics.json"}, p.name
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
        for k in ("wall_seconds", "resumes", "processes"):     # one entry per process
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
        # (no cleanup in between: the scorer clears a killed scorer's temp file itself)
        assert ev.main(["--run", str(run), "--split", "test", *small]) == 0, what
        assert ev.main(["--run", str(run), "--split", "test", *small]) == 2, what
        published += faults.published

        assert published.count("metrics.json") == 1, what           # written exactly once
        assert published.count("test_metrics.json") == 1, what      # scored exactly once
        assert sorted(p.name for p in run.iterdir()) == sorted(
            [*RUN_FILES, "test_metrics.json"]), what
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
        assert sorted(p.name for p in run.iterdir()) == RUN_FILES, (k1, k2)
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
PROJECT = "/scratch/hk4488/roadseg"        # hard-coded in the scripts: data, .deps, runs, logs


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
    canary = _directives(ROOT / "scripts/sbatch_canary.sh")              # G6
    assert (canary["--partition"], canary["--gres"], canary["--cpus-per-task"], canary["--mem"],
            canary["--time"]) == ("l40s_public,h200_public", "gpu:1", "8", "48G", "00:40:00")
    assert "--requeue" not in canary and "--array" not in canary and "--qos" not in canary
    assert canary["--account"] == "torch_pr_627_general"
    for script in ("sbatch_train_array.sh", "sbatch_aggregate.sh", "sbatch_smoke_cpu.sh",
                   "sbatch_canary.sh", "launch.sh"):
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
if "env" in s:
    note(mod + " env " + " ".join(os.environ.get(k, "<unset>") for k in s["env"]))
if "touch" in s:
    run = sys.argv[sys.argv.index("--run") + 1]
    open(os.path.join(run, s["touch"]), "w").write("{{}}")
if s.get("spawn"):                      # a DataLoader worker: ignores SIGTERM, outlives nothing
    pid = os.fork()
    if pid == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(120)
        os._exit(0)
    open(log + "." + mod + ".worker", "w").write(str(pid))
if s.get("block"):                      # a long step that honours SIGTERM the way train.py does
    got = []
    if s.get("handle", True):
        signal.signal(signal.SIGTERM, lambda n, f: got.append(n))
    open(log + "." + mod + ".ready", "w").close()
    t0 = time.time()
    while not got and time.time() - t0 < 60:
        time.sleep(0.02)
    note(mod + (" got SIGTERM" if got else " timed out"))
    time.sleep(s.get("linger", 0))      # still checkpointing when the next signal arrives
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
        for s in (signum if isinstance(signum, tuple) else (signum,)):
            p.send_signal(s)                           # to the batch shell only, like --signal=B:
            time.sleep(0.2)
    out, _ = p.communicate(timeout=120)
    lines = log.read_text().splitlines() if log.exists() else []
    return p.returncode, lines, out


def test_wrapper_runs_train_then_scores_the_test_split(tmp_path):
    rc, log, out = _wrapper(tmp_path, {})
    assert rc == 0, out
    run = tmp_path / "runs" / "massachusetts_R1_s1"
    assert log == [                                    # G3: both steps name the GPU explicitly
        f"train argv -m roadseg.train --config {PROJECT}/configs/massachusetts.yaml --rung R1 "
        f"--seed 1 --resume auto --runs-dir {tmp_path / 'runs'} --device cuda",
        f"eval argv -m roadseg.eval --run {run} --split test --workers 8 --device cuda"]
    assert "task 4 -> dataset=massachusetts name=R1 seed=1" in out
    assert "WARNING: the LIVE tree, not a snapshot" in out          # no ROADSEG_CODE: say so
    rc, log, out = _wrapper(tmp_path, {}, task=33)                 # a baseline uses --model
    assert rc == 0, out
    assert f"--config {PROJECT}/configs/deepglobe.yaml --model dlinknet --seed 0" in log[0]
    assert log[1].startswith(f"eval argv -m roadseg.eval --run {tmp_path / 'runs'}/deepglobe_dlinknet_s0 ")


def test_wrapper_takes_code_and_configs_from_the_frozen_snapshot(tmp_path):
    """G5. With ROADSEG_CODE (exported by launch.sh) python imports the snapshot's src and
    reads the snapshot's configs; .deps, runs and logs stay under the project."""
    snap = tmp_path / "releases" / "abc1234"
    (snap / "src" / "roadseg").mkdir(parents=True)
    (snap / "configs").mkdir()
    (snap / "configs" / "massachusetts.yaml").write_text("dataset: massachusetts\n")
    (snap / "COMMIT").write_text("a" * 40 + "\n")
    spec = {"train": {"env": ["PYTHONPATH", "ROADSEG_CODE"]}, "eval": {"env": ["PYTHONPATH"]}}
    rc, log, out = _wrapper(tmp_path, spec, env_extra={"ROADSEG_CODE": str(snap)})
    assert rc == 0, out
    assert f"--config {snap}/configs/massachusetts.yaml --rung R1" in log[0]
    assert log[1] == f"train env {PROJECT}/.deps:{snap}/src {snap}"
    assert log[3] == f"eval env {PROJECT}/.deps:{snap}/src"
    assert f"code {snap}  commit {'a' * 40} (frozen snapshot)" in out and "WARNING" not in out
    # a snapshot that is not there must stop the task before python starts
    rc, log, out = _wrapper(tmp_path, {}, env_extra={"ROADSEG_CODE": str(tmp_path / "gone")})
    assert rc == 2 and log == [] and "no code at ROADSEG_CODE" in out, out


# ----------------------------------------------------------------------------- G5: code identity

SHA_A, SHA_B = "a" * 40, "b" * 40


def _as_snapshot(monkeypatch, root, sha):
    """Make train.py find itself in a frozen snapshot at `root` holding commit `sha`."""
    root.mkdir(exist_ok=True)
    (root / "COMMIT").write_text(sha + "\n")
    monkeypatch.setattr(tr, "PROJECT", root)


def test_code_identity_comes_from_the_commit_file_or_from_git(tmp_path, monkeypatch):
    monkeypatch.delenv("ROADSEG_CODE", raising=False)
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "COMMIT").write_text(SHA_A + "\n")
    assert tr.code_identity(snap) == {"commit": SHA_A, "dirty": False,
                                      "source": str(snap / "COMMIT")}
    monkeypatch.setenv("ROADSEG_CODE", str(snap))                  # as launch.sh exports it
    assert tr.code_identity(snap)["commit"] == SHA_A
    monkeypatch.setenv("ROADSEG_CODE", str(tmp_path / "elsewhere"))
    with pytest.raises(tr.Refused, match="ROADSEG_CODE=.*imported from"):
        tr.code_identity(snap)                                     # recording one, running another
    monkeypatch.delenv("ROADSEG_CODE")
    (snap / "COMMIT").write_text("HEAD\n")
    with pytest.raises(tr.Refused, match="full commit sha"):
        tr.code_identity(snap)

    repo, sha, _ = _launch_repo(tmp_path)                          # a working tree
    assert tr.code_identity(repo) == {"commit": sha, "dirty": False, "source": "git"}
    (repo / "src" / "roadseg" / "new.py").write_text("x = 1\n")    # untracked code counts
    assert tr.code_identity(repo) == {"commit": sha, "dirty": True, "source": "git"}
    inner = repo / "releases" / "broken"                           # ignored dir inside the repo,
    inner.mkdir(parents=True)                                      # no COMMIT: NOT the repo's HEAD
    assert tr.code_identity(inner) == {"commit": "unknown", "dirty": None, "source": "none"}
    assert tr.code_identity(tmp_path / "snap" / "..")["commit"] == "unknown"   # no git at all

    here = tr.code_identity()                                      # this checkout
    assert len(here["commit"]) == 40 and here["source"] == "git" and here["dirty"] in (True, False)
    a, b = {"commit": SHA_A, "dirty": False, "source": "x"}, {"commit": SHA_A, "dirty": False,
                                                              "source": "git"}
    assert tr.same_code(a, b) and not tr.same_code(a, {**b, "commit": SHA_B})
    assert not tr.same_code(a, {**b, "dirty": True}) and not tr.same_code(None, a)


def _preempt_at_step(monkeypatch, n):
    """SIGTERM this process during the n-th optimizer step (1-based) of the next main()."""
    real, calls = tr.ComboLoss, []

    class Loss(real):
        def forward(self, *a):
            calls.append(1)
            if len(calls) == n:
                os.kill(os.getpid(), signal.SIGTERM)
            return super().forward(*a)

    monkeypatch.setattr(tr, "ComboLoss", Loss)
    return calls


def test_resume_with_changed_code_is_refused_and_allowed_only_explicitly(tiny, monkeypatch, capsys):
    """G5. Every checkpoint carries the identity of the code that wrote it; a resume under
    different code exits 'refused' showing both, unless --allow-code-change, which records
    both. metrics.json lists every process that worked on the run."""
    snap = tiny.root / "snap"
    _as_snapshot(monkeypatch, snap, SHA_A)
    _preempt_at_step(monkeypatch, 6)                               # epoch 1, step 2
    assert tiny.train() == 75
    assert f"code={SHA_A[:12]}" in capsys.readouterr().out
    run = tiny.runs / "massachusetts_R1_s0"
    extra = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["extra"]
    assert extra["code"] == {"commit": SHA_A, "dirty": False, "source": str(snap / "COMMIT")}
    assert [(p["code_commit"], p["first_epoch"], p["last_epoch"]) for p in extra["processes"]] == [
        (SHA_A, 0, 1)]
    latest = (run / "latest.pt").read_bytes()

    _as_snapshot(monkeypatch, snap, SHA_B)                         # the code moved on
    with pytest.raises(tr.Refused, match=f"code changed.*{SHA_A}.*{SHA_B}.*--allow-code-change"):
        tiny.train("--resume", "auto")
    assert (run / "latest.pt").read_bytes() == latest and not (run / "metrics.json").exists()

    assert tiny.train("--resume", "auto", "--allow-code-change") == 0
    assert "code changed" in capsys.readouterr().out               # said so, loudly
    m = read(run / "metrics.json")
    assert (m["code_commit"], m["code_dirty"], m["git_commit"]) == (SHA_B, False, SHA_B)
    assert [(p["code_commit"], p["code_dirty"], p["first_epoch"], p["last_epoch"])
            for p in m["processes"]] == [(SHA_A, False, 0, 1), (SHA_B, False, 1, 2)]
    assert all(p["hostname"] == socket.gethostname() and p["device_name"] == "cpu"
               and p["started"] and p["pid"] == os.getpid() for p in m["processes"])
    assert m["code_changes"] == [{"epoch": 1, "step_in_epoch": 2, "from": extra["code"],
                                  "to": {"commit": SHA_B, "dirty": False,
                                         "source": str(snap / "COMMIT")}}]
    assert m["resumes"] == 1 and [r["epoch"] for r in m["history"]] == [0, 1, 2]


def test_an_unchanged_snapshot_resumes_and_one_process_is_one_entry(tiny, monkeypatch, capsys):
    _as_snapshot(monkeypatch, tiny.root / "snap", SHA_A)
    calls = _preempt_at_step(monkeypatch, 12)                      # during the very last step
    assert tiny.train() == 75
    assert tiny.train("--resume", "auto") == 0                     # same COMMIT: no flag needed
    assert len(calls) == 12
    m = read(tiny.runs / "massachusetts_R1_s0" / "metrics.json")
    assert m["code_changes"] == [] and m["code_commit"] == SHA_A
    # the second process only validated epoch 2 and ran the finish phase
    assert [(p["first_epoch"], p["last_epoch"]) for p in m["processes"]] == [(0, 2), (2, 2)]

    _as_snapshot(monkeypatch, tiny.root / "snap", SHA_B)           # "already complete" is not a
    assert tiny.train("--resume", "auto") == 0                     # resume: nothing is computed
    assert "already complete" in capsys.readouterr().out


def test_a_checkpoint_without_a_code_identity_is_not_resumed_silently(tiny, monkeypatch):
    _preempt_at_step(monkeypatch, 6)
    assert tiny.train() == 75
    latest = tiny.runs / "massachusetts_R1_s0" / "latest.pt"
    ck = torch.load(latest, map_location="cpu", weights_only=False)
    for key in ("code", "processes", "code_changes"):              # as written before G5
        del ck["extra"][key]
    torch.save(ck, latest)
    with pytest.raises(tr.Refused, match="code changed.*<not recorded>"):
        tiny.train("--resume", "auto")
    assert tiny.train("--resume", "auto", "--allow-code-change") == 0
    m = read(latest.parent / "metrics.json")
    assert m["code_changes"][0]["from"] is None and len(m["processes"]) == 1


# ----------------------------------------------------------------------------- G6: preflight

def _count_builds(monkeypatch, fail_first=None):
    """Stub build_model that records each call's cfg; the model of the FIRST call (the
    preflight's throwaway) can be made to fail in forward with `fail_first`."""
    builds = []

    class Failing(Tiny):
        def forward(self, x):
            raise fail_first

    def build(cfg):
        builds.append(dict(cfg))
        torch.randn(13)                         # construction draws from the global RNG
        return Failing() if fail_first is not None and len(builds) == 1 else Tiny()

    monkeypatch.setattr(tr, "build_model", build)
    return builds


def _result(run):
    m = read(run / "metrics.json")
    return (weights_digest(run / "best.pt"), weights_digest(run / "latest.pt"), m["threshold"],
            [(r["epoch"], r["train_loss"], r["val_iou"], r["lr"]) for r in m["history"]],
            json.dumps(m["val"], sort_keys=True))


def test_preflight_runs_first_on_a_throwaway_model_and_does_not_change_the_run(tiny, monkeypatch,
                                                                              capsys):
    """G6. Same final weights, history and scores with and without the preflight: the real
    model is seeded and built after it, and the crop stream does not depend on it."""
    run = tiny.runs / "massachusetts_R1_s0"
    builds = _count_builds(monkeypatch)
    assert tiny.train("--preflight") == 0                         # forced: this is a CPU
    out = capsys.readouterr().out
    assert out.index("preflight ok:") < out.index("steps/epoch")  # before the real model exists
    assert "name=R1 batch=2 tile=32 amp=False" in out and "steps_per_s=" in out and "peak_gb=" in out
    assert len(builds) == 2 and builds[0]["pretrained"] is False  # the throwaway loads no weights
    with_preflight = _result(run)

    shutil.rmtree(tiny.runs)
    builds.clear()
    assert tiny.train() == 0                                      # CPU default: no preflight
    assert "preflight ok" not in capsys.readouterr().out and len(builds) == 1
    assert _result(run) == with_preflight

    shutil.rmtree(tiny.runs)
    builds.clear()
    assert tiny.train("--preflight", "--skip-preflight") == 0     # --skip-preflight wins
    assert "preflight ok" not in capsys.readouterr().out and len(builds) == 1


def test_preflight_exercises_the_real_step_and_the_real_loss(tiny, monkeypatch):
    """2 optimizer steps at the configured batch and tile through engine.train_one_epoch
    (autocast + gradient guard) with the run's loss -- clDice for R3 -- then one eval-mode
    batch of 8 tiles through predict_full."""
    seen = {"loss": [], "forward": [], "steps": []}
    real_loss, real_train = tr.ComboLoss, tr.train_one_epoch

    class Loss(real_loss):
        def __init__(self, use_cldice=False, **k):
            seen["loss"].append(use_cldice)
            super().__init__(use_cldice=use_cldice, **k)

    class Probe(Tiny):
        def forward(self, x):
            seen["forward"].append((tuple(x.shape), self.training))
            return super().forward(x)

    def train_one_epoch(*a, **k):
        seen["steps"].append(k.get("max_batches"))
        return real_train(*a, **k)

    monkeypatch.setattr(tr, "ComboLoss", Loss)
    monkeypatch.setattr(tr, "build_model", lambda cfg: Probe())
    monkeypatch.setattr(tr, "train_one_epoch", train_one_epoch)
    cfg = tr.resolve_config(args_for(tiny.cfg(batch_size=3), "--rung", "R3"))
    res = tr.preflight(cfg, "cpu", use_amp=False)
    assert seen["loss"] == [True] and seen["steps"] == [2]
    assert seen["forward"] == [((3, 3, 32, 32), True), ((3, 3, 32, 32), True),
                               ((8, 3, 32, 32), False)]
    assert res["steps_per_s"] > 0 and res["batch"] == 3 and res["tile"] == 32 and res["name"] == "R3"
    seen["loss"].clear()
    tr.preflight(tr.resolve_config(args_for(tiny.cfg(), "--rung", "R2")), "cpu", use_amp=False)
    tr.preflight(tr.resolve_config(args_for(tiny.cfg(), "--model", "unet")), "cpu", use_amp=False)
    assert seen["loss"] == [False, False]


def _sentinel(runs, name, host):
    """What a failed preflight of run `name` on `host` leaves behind (host None: malformed)."""
    runs.mkdir(parents=True, exist_ok=True)
    p = runs / f"PREFLIGHT_FAILED.{name}.txt"
    p.write_text((f"run: {name}\nhost: {host}\n" if host else "half a fi")
                 + "device: cuda (NVIDIA L40S)\n\nTraceback (most recent call last): ...\n")
    return p


def test_a_failed_preflight_writes_a_sentinel_and_exits_3(tiny, monkeypatch, capsys):
    """G6. The task whose own preflight fails exits 3 (never requeued) and leaves
    PREFLIGHT_FAILED.<run>.txt. H1: that one file, from one host, stops nobody else."""
    builds = _count_builds(monkeypatch, fail_first=RuntimeError("CUDA out of memory (simulated)"))
    sentinel = tiny.runs / "PREFLIGHT_FAILED.massachusetts_R1_s0.txt"
    with pytest.raises(tr.PreflightFailed, match="PREFLIGHT_FAILED.massachusetts_R1_s0.txt"):
        tiny.train("--preflight")
    text = sentinel.read_text()
    assert "CUDA out of memory (simulated)" in text and "Traceback" in text
    assert f"host: {socket.gethostname()}\n" in text and tr.code_identity()["commit"] in text
    assert "device: cpu" in text and "run: massachusetts_R1_s0" in text
    run = tiny.runs / "massachusetts_R1_s0"
    assert len(builds) == 1 and not (run / "latest.pt").exists()  # the real model never existed
    assert tr.sentinel_hosts(tiny.runs) == {socket.gethostname(): [sentinel]}

    capsys.readouterr()
    assert tiny.train("--preflight", name=("--rung", "R2")) == 0  # another task: goes ahead
    out = capsys.readouterr().out
    assert "WARNING" in out and sentinel.name in out and socket.gethostname() in out
    assert out.index("WARNING") < out.index("preflight ok:")      # ... to its OWN preflight
    assert tiny.train("--preflight") == 0                         # and so does a relaunch of R1


def test_sentinels_stop_a_fresh_run_only_when_two_hosts_failed(tiny, monkeypatch, capsys):
    """H1. One host can be one bad GPU; two hosts is the code. Only then is a fresh run
    refused (exit 3) -- and a finished run is still 'already complete'."""
    builds = _count_builds(monkeypatch)
    a = _sentinel(tiny.runs, "deepglobe_R0_s0", "gl001")
    assert tiny.train() == 0                                      # fresh run, one sentinel
    out = capsys.readouterr().out
    assert "WARNING" in out and a.name in out and "gl001" in out
    _sentinel(tiny.runs, "deepglobe_R0_s1", "gl001")              # the same host again:
    assert tiny.train(name=("--rung", "R0")) == 0                 # still one host

    b = _sentinel(tiny.runs, "deepglobe_R3_s2", "gl002")
    n = len(builds)
    for flags in ((), ("--preflight",), ("--resume", "auto")):    # a fresh run, however started
        with pytest.raises(tr.PreflightFailed) as e:
            tiny.train(*flags, name=("--rung", "R2"))
        msg = str(e.value)
        assert a.name in msg and b.name in msg and "gl001" in msg and "gl002" in msg
        assert "2 different hosts" in msg
    assert len(builds) == n                                       # ... stops before building
    assert not (tiny.runs / "massachusetts_R2_s0" / "latest.pt").exists()
    assert tiny.train("--dry-run", name=("--rung", "R2")) == 0    # (a dry run is not a run)

    capsys.readouterr()
    for flags in ((), ("--resume", "auto"), ("--preflight",)):    # R1 has metrics.json:
        assert tiny.train(*flags) == 0                            # a requeue must reach scoring
        assert "already complete" in capsys.readouterr().out
    assert tiny.train("--skip-preflight", name=("--rung", "R2")) == 0   # disables both


def test_a_run_that_has_trained_never_consults_sentinels(tiny, monkeypatch, capsys):
    """H1. latest.pt proves the code path on some GPU: a resume is never held up by other
    tasks' sentinels (it still runs its own preflight)."""
    calls = _preempt_at_step(monkeypatch, 6)
    assert tiny.train() == 75
    _sentinel(tiny.runs, "deepglobe_R0_s0", "gl001")
    _sentinel(tiny.runs, "deepglobe_R3_s2", "gl002")
    capsys.readouterr()
    assert tiny.train("--resume", "auto", "--preflight") == 0
    out = capsys.readouterr().out
    assert "resumed from" in out and "at epoch 1 step 2" in out and "preflight ok:" in out
    assert "WARNING" not in out and "PREFLIGHT_FAILED" not in out
    assert len(calls) == 12 + 2                  # the 12 steps of the run + the preflight's 2


def test_a_malformed_sentinel_counts_as_its_own_unknown_host(tiny, monkeypatch, capsys):
    bad = _sentinel(tiny.runs, "deepglobe_unet_s0", None)         # no `host:` line
    assert list(tr.sentinel_hosts(tiny.runs)) == [f"unknown host ({bad.name})"]
    assert tiny.train() == 0                                      # alone: one host
    assert "WARNING" in capsys.readouterr().out
    good = _sentinel(tiny.runs, "deepglobe_R0_s0", "gl001")
    with pytest.raises(tr.PreflightFailed, match="unknown host.*gl001|gl001.*unknown host"):
        tiny.train(name=("--rung", "R2"))
    good.unlink()
    _sentinel(tiny.runs, "deepglobe_unet_s1", None)               # two malformed files are
    with pytest.raises(tr.PreflightFailed, match="2 different hosts"):   # two unknown hosts
        tiny.train(name=("--rung", "R2"))


def test_preflight_only_tests_and_exits(tiny, monkeypatch, capsys):
    builds = _count_builds(monkeypatch)
    assert tiny.train("--preflight-only") == 0
    assert "preflight ok: name=R1" in capsys.readouterr().out and len(builds) == 1
    assert not (tiny.runs / "massachusetts_R1_s0").exists()       # no run was started

    builds = _count_builds(monkeypatch, fail_first=FloatingPointError("non-finite loss nan"))
    with pytest.raises(tr.PreflightFailed, match="PREFLIGHT_FAILED.massachusetts_R3_s0.txt"):
        tiny.train("--preflight-only", name=("--rung", "R3"))
    assert "non-finite loss nan" in (tiny.runs / "PREFLIGHT_FAILED.massachusetts_R3_s0.txt").read_text()
    builds = _count_builds(monkeypatch)                           # the canary tests them ALL:
    assert tiny.train("--preflight-only", name=("--rung", "R0")) == 0   # an earlier failure
    assert len(builds) == 1                                             # does not skip the next
    with pytest.raises(SystemExit):                               # argparse: contradictory
        tiny.train("--preflight-only", "--skip-preflight")


# ----------------------------------------------------------------------------- G7: finish phase

def test_a_signal_during_threshold_selection_exits_75_and_only_the_scoring_is_redone(
        tiny, monkeypatch, capsys):
    """G7. The finish phase honours the preempt flag between its stages. The threshold is
    saved (threshold.json) before the flag is looked at, so the requeued job does not
    select it again -- and never retrains."""
    assert tiny.train() == 0
    run = tiny.runs / "massachusetts_R1_s0"
    ref = read(run / "metrics.json")
    assert read(run / "threshold.json")["threshold"] == ref["threshold"]
    shutil.rmtree(tiny.runs)

    calls = {"select": 0, "score": 0, "steps": 0}
    real_select, real_evaluate, real_loss = tr.select_threshold, tr.evaluate, tr.ComboLoss

    def select(*a, **k):
        calls["select"] += 1
        os.kill(os.getpid(), signal.SIGTERM)                # arrives DURING threshold selection
        return real_select(*a, **k)

    def evaluate(*a, **k):
        calls["score"] += bool(k.get("full_metrics"))
        return real_evaluate(*a, **k)

    class Loss(real_loss):
        def forward(self, *a):
            calls["steps"] += 1
            return super().forward(*a)

    monkeypatch.setattr(tr, "select_threshold", select)
    monkeypatch.setattr(tr, "evaluate", evaluate)
    monkeypatch.setattr(tr, "ComboLoss", Loss)
    assert tiny.train() == 75
    out = capsys.readouterr().out
    assert "preempted: finish phase stopped after threshold selection" in out
    assert calls == {"select": 1, "score": 0, "steps": 12}
    saved = read(run / "threshold.json")
    assert saved["threshold"] == ref["threshold"] and saved["best_epoch"] == ref["best_epoch"]
    assert not (run / "metrics.json").exists()

    monkeypatch.setattr(tr, "select_threshold", real_select)
    assert tiny.train("--resume", "auto") == 0
    out = capsys.readouterr().out
    assert "at epoch 3 step 0" in out and "threshold 0." in out and "reused from" in out
    assert calls == {"select": 1, "score": 1, "steps": 12}     # no retraining, no re-selection
    m = read(run / "metrics.json")
    assert (m["threshold"], m["threshold_val_iou"], m["val"], m["per_image"]) == (
        ref["threshold"], ref["threshold_val_iou"], ref["val"], ref["per_image"])
    assert m["resumes"] == 1 and len(m["processes"]) == 2
    assert sorted(p.name for p in run.iterdir()) == RUN_FILES


def test_a_signal_during_validation_scoring_still_publishes_the_finished_run(tiny, monkeypatch):
    """After scoring nothing is left but a rename: publish, do not throw the scores away."""
    real_evaluate = tr.evaluate

    def evaluate(*a, **k):
        if k.get("full_metrics"):
            os.kill(os.getpid(), signal.SIGUSR1)
        return real_evaluate(*a, **k)

    monkeypatch.setattr(tr, "evaluate", evaluate)
    assert tiny.train() == 0
    assert (tiny.runs / "massachusetts_R1_s0" / "metrics.json").exists()


def test_a_saved_threshold_is_only_reused_for_the_checkpoint_it_was_selected_on(tiny, monkeypatch):
    assert tiny.train() == 0
    run = tiny.runs / "massachusetts_R1_s0"
    ref = read(run / "metrics.json")
    (run / "metrics.json").unlink()
    saved = read(run / "threshold.json")
    for stale in ({**saved, "best_epoch": saved["best_epoch"] + 1, "threshold": 0.8},
                  {**saved, "stride": 1, "threshold": 0.8}, {"threshold": 0.8}):
        (run / "threshold.json").write_text(json.dumps(stale))
        assert tiny.train("--resume", "auto") == 0
        assert read(run / "metrics.json")["threshold"] == ref["threshold"]      # selected afresh
        assert read(run / "threshold.json")["threshold"] == ref["threshold"]
        (run / "metrics.json").unlink()


# ----------------------------------------------------------------------------- G7: aggregate

AGGREGATE = ROOT / "scripts" / "sbatch_aggregate.sh"


def _aggregate(tmp_path, missing=(), *args, spec=None):
    """The REAL aggregate script with a stub python; every run of the map has a
    test_metrics.json except `missing`. Returns (exit code, stub log, stdout+stderr)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "python"
    fake.write_text(FAKE_PYTHON.format(python=sys.executable))
    fake.chmod(0o755)
    runs, log = tmp_path / "runs", tmp_path / "log"
    log.unlink(missing_ok=True)
    shutil.rmtree(runs, ignore_errors=True)
    for _, _, _, _, name in _print_map():
        if name not in missing:
            (runs / name).mkdir(parents=True)
            (runs / name / "test_metrics.json").write_text("{}")
    env = {"PATH": os.environ["PATH"], "FAKE_LOG": str(log), "FAKE_SPEC": json.dumps(spec or {}),
           "ROADSEG_PY": str(fake), "ROADSEG_RUNS": str(runs)}
    r = subprocess.run(["bash", str(AGGREGATE), *args], capture_output=True, text=True, env=env)
    return r.returncode, (log.read_text().splitlines() if log.exists() else []), r.stdout + r.stderr


def test_aggregate_refuses_a_partial_set_but_still_prints_the_table(tmp_path):
    """G7. `afterany` starts the aggregation whatever happened to the array: a table over
    35 runs must not look like the result."""
    stats = [f"stats argv -m roadseg.stats --runs-dir {tmp_path / 'runs'} --dataset {d} --split test"
             for d in ("massachusetts", "deepglobe")]
    rc, log, out = _aggregate(tmp_path)
    assert rc == 0 and log == stats and "36 of 36" in out and "MISSING" not in out, out

    gone = ("deepglobe_R3_s2", "massachusetts_unet_s0")
    rc, log, out = _aggregate(tmp_path, gone)
    assert rc != 0 and log == stats, out                        # the table either way
    assert "34 of 36" in out and "INCOMPLETE" in out
    assert all(f"MISSING {name}" in out for name in gone) and out.count("MISSING ") == 2

    rc, log, out = _aggregate(tmp_path, gone, "--allow-partial", "--metric", "cl_dice")
    assert rc == 0, out
    assert log == [s + " --metric cl_dice" for s in stats]      # the flag is ours, not stats'
    assert "34 of 36" in out and "--allow-partial" in out and out.count("MISSING ") == 2

    # --allow-partial excuses missing runs, not a failed aggregation
    rc, log, out = _aggregate(tmp_path, gone, "--allow-partial", spec={"stats": {"rc": 1}})
    assert rc != 0 and "AGGREGATE FAILED for massachusetts" in out and len(log) == 2, out


# ----------------------------------------------------------------------------- G7: smoke script

SMOKE = ROOT / "scripts" / "sbatch_smoke_cpu.sh"


def test_smoke_acknowledgement_is_only_written_by_the_full_rehearsal(tmp_path):
    """G7. launch.sh trusts runs_smoke/SMOKE_PASSED. A narrowed rehearsal (here: no section
    at all) prints its verdict but must not write the acknowledgement."""
    runs = tmp_path / "runs_smoke"
    r = subprocess.run(["bash", str(SMOKE)], capture_output=True, text=True,
                       env=dict(ENV, ROADSEG_RUNS_SMOKE=str(runs), SMOKE_STEPS="none"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SMOKE: ALL PASSED" in r.stdout
    assert "SMOKE_PASSED not written" in r.stdout and "narrowed" in r.stdout
    assert not (runs / "SMOKE_PASSED").exists()
    src = SMOKE.read_text()
    assert 'kill -TERM -- "-$pid"' in src and "setsid" in src     # drills signal the GROUP
    assert "ROADSEG_CODE" in src and 'echo "$SHA" > "$RUNS/SMOKE_PASSED"' in src


# ----------------------------------------------------------------------------- G5: launch.sh

LAUNCH = ROOT / "scripts" / "launch.sh"
FAKE_SBATCH = '''#!/bin/bash
# records its arguments and the ROADSEG_* variables it was given; answers with a job id
n=$(( $(cat "$FAKE_SBATCH_LOG" 2>/dev/null | grep -c '^sbatch') + 1 ))
echo "sbatch $*" >> "$FAKE_SBATCH_LOG"
echo "env PY=${ROADSEG_PY-<unset>} RUNS=${ROADSEG_RUNS-<unset>} LOGS=${ROADSEG_LOGS-<unset>}" >> "$FAKE_SBATCH_LOG"
echo "$(( 9000 + n ));torch"
'''


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t.t",
                        *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _launch_repo(tmp_path):
    """A small committed repository standing in for the project, and a fake sbatch."""
    repo = tmp_path / "project"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src" / "roadseg").mkdir(parents=True)
    for name in ("sbatch_train_array.sh", "sbatch_aggregate.sh", "sbatch_canary.sh"):
        (repo / "scripts" / name).write_text(f"#!/bin/bash\necho {name}\n")
    (repo / "src" / "roadseg" / "train.py").write_text("VERSION = 1\n")
    (repo / ".gitignore").write_text("releases/\nruns/\nruns_smoke/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    sbatch = tmp_path / "fake_sbatch"
    sbatch.write_text(FAKE_SBATCH)
    sbatch.chmod(0o755)
    return repo, _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "--short", "HEAD")


def _launch(tmp_path, repo, *args, smoke=None, env_extra=None):
    """Run the REAL launch.sh against the stand-in repository. `smoke` is the content of the
    rehearsal's acknowledgement file (None: no file)."""
    flag = repo / "runs_smoke" / "SMOKE_PASSED"
    flag.parent.mkdir(exist_ok=True)
    flag.unlink(missing_ok=True)
    if smoke is not None:
        flag.write_text(smoke + "\n")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "ROADSEG_PROJECT": str(repo),
           "ROADSEG_SBATCH": str(tmp_path / "fake_sbatch"),
           "ROADSEG_SLURM_LOGS": str(tmp_path / "slurm_logs"),
           "FAKE_SBATCH_LOG": str(tmp_path / "sbatch.log"),
           "ROADSEG_PY": "/test/python", "ROADSEG_RUNS": "/test/runs", "ROADSEG_LOGS": "/test/logs",
           **(env_extra or {})}
    r = subprocess.run(["bash", str(LAUNCH), *args], capture_output=True, text=True, env=env)
    log = tmp_path / "sbatch.log"
    return r.returncode, r.stdout, r.stderr, (log.read_text().splitlines() if log.exists() else [])


def test_launch_dry_run_prints_the_plan_and_creates_nothing(tmp_path):
    repo, sha, short = _launch_repo(tmp_path)
    def listing():                       # (runs_smoke/SMOKE_PASSED is put there by _launch)
        return sorted(str(p) for p in tmp_path.rglob("*")
                      if not {".git", "runs_smoke"} & set(p.parts))

    before = listing()
    rc, out, err, calls = _launch(tmp_path, repo, "--dry-run", smoke=sha)
    assert rc == 0, out + err
    snap = repo / "releases" / short
    assert f"snapshot: {snap} (would be created" in out and "DRY RUN" in out
    export = f"--export=ALL,ROADSEG_CODE={snap}"
    assert f"would run: {tmp_path / 'fake_sbatch'} --parsable {export} {snap}/scripts/sbatch_train_array.sh" in out
    assert (f"would run: {tmp_path / 'fake_sbatch'} --parsable --dependency=afterany:<array job id> "
            f"{export} {snap}/scripts/sbatch_aggregate.sh") in out
    assert calls == []                                               # nothing submitted
    assert listing() == before                # no snapshot, no LAUNCHES.log, no log directory
    rc, out, err, calls = _launch(tmp_path, repo, "--dry-run", "--canary", smoke=sha)
    assert rc == 0 and calls == [] and out.count("would run:") == 1
    assert f"{export} {snap}/scripts/sbatch_canary.sh" in out


def test_launch_refuses_a_dirty_tree(tmp_path):
    repo, sha, short = _launch_repo(tmp_path)
    (repo / "src" / "roadseg" / "train.py").write_text("VERSION = 2\n")      # modified, tracked
    for args in (("--dry-run",), ()):
        rc, out, err, calls = _launch(tmp_path, repo, *args, smoke=sha)
        assert rc == 2 and calls == [], out + err
        assert "REFUSED" in err and "dirty" in err and "src/roadseg/train.py" in err
    _git(repo, "checkout", "-q", "--", ".")
    (repo / "src" / "roadseg" / "new_module.py").write_text("x = 1\n")       # untracked code
    rc, out, err, calls = _launch(tmp_path, repo, "--dry-run", smoke=sha)
    assert rc == 2 and "new_module.py" in err
    assert not (repo / "releases").exists() and not (repo / "runs").exists()


def test_launch_requires_the_cpu_rehearsal_to_have_passed_at_this_commit(tmp_path):
    repo, sha, short = _launch_repo(tmp_path)
    rc, out, err, calls = _launch(tmp_path, repo, smoke=None)
    assert rc == 2 and calls == [] and "REFUSED" in err and "SMOKE_PASSED" in err, out + err
    rc, out, err, calls = _launch(tmp_path, repo, smoke="b" * 40)            # another commit
    assert rc == 2 and calls == [] and "b" * 40 in err and sha in err, out + err
    assert not (repo / "releases").exists()
    rc, out, err, calls = _launch(tmp_path, repo, "--no-smoke-check", "--dry-run", smoke=None)
    assert rc == 0 and "WARNING --no-smoke-check" in err and "WITHOUT a passed CPU rehearsal" in err
    rc, out, err, calls = _launch(tmp_path, repo, "--bogus", smoke=sha)
    assert rc == 2 and calls == [] and "unknown argument" in err


def test_launch_freezes_a_snapshot_and_submits_from_it(tmp_path):
    repo, sha, short = _launch_repo(tmp_path)
    rc, out, err, calls = _launch(tmp_path, repo, smoke=sha)
    assert rc == 0, out + err
    snap = repo / "releases" / short
    assert (snap / "COMMIT").read_text().strip() == sha
    assert (snap / "src/roadseg/train.py").read_text() == "VERSION = 1\n"
    assert sorted(p.name for p in snap.iterdir()) == [".gitignore", "COMMIT", "scripts", "src"]
    assert not os.access(snap / "src/roadseg/train.py", os.W_OK)             # frozen
    export = f"--export=ALL,ROADSEG_CODE={snap}"
    assert calls == [
        f"sbatch --parsable {export} {snap}/scripts/sbatch_train_array.sh",
        "env PY=<unset> RUNS=<unset> LOGS=<unset>",                 # test overrides do not leak
        f"sbatch --parsable --dependency=afterany:9001 {export} {snap}/scripts/sbatch_aggregate.sh",
        "env PY=<unset> RUNS=<unset> LOGS=<unset>"]
    assert "array job: 9001" in out and "aggregate job: 9002" in out and str(snap) in out
    line = (repo / "runs" / "LAUNCHES.log").read_text().splitlines()
    assert len(line) == 1 and line[0].endswith(f"sha={sha} snapshot={snap} array=9001 aggregate=9002")
    assert (tmp_path / "slurm_logs").is_dir()
    assert _git(repo, "status", "--porcelain") == ""                # launching leaves it clean

    # the live tree moves on; the snapshot does not, and a relaunch of the old commit reuses it
    (repo / "src" / "roadseg" / "train.py").write_text("VERSION = 2\n")
    _git(repo, "commit", "-q", "-am", "second")
    assert (snap / "src/roadseg/train.py").read_text() == "VERSION = 1\n"
    _git(repo, "checkout", "-q", sha)
    rc, out, err, calls = _launch(tmp_path, repo, "--canary", smoke=sha)
    assert rc == 0 and "already there" in out, out + err
    assert calls[-2] == f"sbatch --parsable {export} {snap}/scripts/sbatch_canary.sh"
    assert len(calls) == 6                                           # the canary only
    log = (repo / "runs" / "LAUNCHES.log").read_text().splitlines()
    assert len(log) == 2 and log[1].endswith(f"snapshot={snap} canary=9003")
    # a snapshot directory that holds something else is never reused
    os.chmod(snap, 0o755)
    os.chmod(snap / "COMMIT", 0o644)
    (snap / "COMMIT").write_text("c" * 40 + "\n")
    rc, out, err, calls = _launch(tmp_path, repo, smoke=sha)
    assert rc == 2 and "REFUSED" in err and len(calls) == 6


def _requeues(log):
    return [line for line in log if line.startswith("scontrol")]


def test_wrapper_forwards_usr1_to_python_and_requeues_on_75(tmp_path):
    """Requirement 5: bash defers traps while a foreground child runs, so the step runs in
    the background under `wait`; USR1 to the shell must reach python as SIGTERM. USR1 is the
    time-limit warning (--signal=B:USR1@180): the ONLY signal that leads to a requeue."""
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True}}, signal_step="train")
    assert rc == 0, out
    assert log[1:] == ["train got SIGTERM", "scontrol requeue 777_4"], out   # once, and no eval
    # outside an array: fall back to the plain job id
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True}}, signal_step="train",
                            env_extra={"SLURM_ARRAY_JOB_ID": ""})
    assert rc == 0 and log[-1] == "scontrol requeue 781" and len(_requeues(log)) == 1, out


@pytest.mark.parametrize("spec, step", [
    ({"train": {"block": True}}, "train"),                        # checkpoints, exits 75
    ({"train": {"block": True, "handle": False}}, "train"),       # dies of the signal (143)
    ({"train": {"block": True, "rc_on_term": 0}}, "train"),       # finishes in the grace period
    ({"eval": {"block": True, "rc_on_term": 143}}, "eval"),       # scoring interrupted
])
def test_wrapper_never_requeues_on_term(tmp_path, spec, step):
    """G4. TERM is `scancel` or a real preemption. The signal is forwarded so python can
    checkpoint, but the wrapper must NOT requeue: that would undo the user's scancel, and
    SLURM requeues a preempted --requeue job itself."""
    rc, log, out = _wrapper(tmp_path, spec, signal_step=step, signum=signal.SIGTERM)
    assert _requeues(log) == [], out
    assert rc != 0, out                                  # not "completed": it was terminated
    assert "not requeueing" in out
    if spec[step].get("handle", True):
        assert log[-1] == f"{step} got SIGTERM", out     # forwarded; python had its chance
    if step == "train":
        assert not any(line.startswith("eval") for line in log), out     # and no scoring


def test_wrapper_term_after_usr1_still_does_not_requeue(tmp_path):
    """A scancel that lands after the time-limit warning wins."""
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True, "linger": 1.0}},
                            signal_step="train", signum=(signal.SIGUSR1, signal.SIGTERM))
    assert _requeues(log) == [] and rc != 0 and "not requeueing" in out, out
    assert log[-1] == "train got SIGTERM", out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}/stat").read_text().split(") ")[-1][0] != "Z"     # not a zombie


@pytest.mark.parametrize("step", ["train", "eval"])
def test_wrapper_watchdog_kills_a_step_that_does_not_stop_after_usr1_and_requeues(tmp_path, step):
    """H5. USR1 leaves 180 s. A python that is still running 120 s later (the finish phase's
    validation scoring cannot be interrupted) would end the job TIMEOUT, un-requeued. The
    watchdog SIGKILLs the step's whole process group -- on-disk state is kill-safe -- and
    requeues."""
    spec = {step: {"block": True, "linger": 100, "spawn": True}}          # deaf to the signal
    t0 = time.time()
    rc, log, out = _wrapper(tmp_path, spec, signal_step=step,
                            env_extra={"ROADSEG_USR1_WATCHDOG": "1"})
    assert rc == 0 and time.time() - t0 < 30, out
    assert _requeues(log) == ["scontrol requeue 777_4"] and log[-1].startswith("scontrol"), out
    assert f"{step} got SIGTERM" in log                            # it WAS told, and went on
    assert "watchdog:" in out and f"{step} exited with 137" in out
    worker = int(Path(f"{tmp_path / 'log'}.{step}.worker").read_text())
    assert not _alive(worker)                                      # the group, not just python


def test_wrapper_watchdog_is_cancelled_when_the_step_stops_by_itself(tmp_path):
    """Prompt exit after USR1: one requeue, nothing killed, and no watchdog left behind (a
    leftover would hold this test's pipe open for its whole 40 s)."""
    t0 = time.time()
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True}}, signal_step="train",
                            env_extra={"ROADSEG_USR1_WATCHDOG": "40"})
    assert rc == 0 and time.time() - t0 < 20, out
    assert log[1:] == ["train got SIGTERM", "scontrol requeue 777_4"], out
    assert "watchdog:" not in out and "train exited with 75" in out


@pytest.mark.parametrize("signals", [(signal.SIGTERM,), (signal.SIGUSR1, signal.SIGTERM)])
def test_wrapper_watchdog_never_fires_after_term(tmp_path, signals):
    """TERM (scancel / preemption) -- alone or after USR1 -- means: no kill by us, no requeue.
    The step outlives the 1 s watchdog delay and ends by itself with its own exit code."""
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True, "linger": 3}},
                            signal_step="train", signum=signals,
                            env_extra={"ROADSEG_USR1_WATCHDOG": "1"})
    assert rc != 0 and _requeues(log) == [], out
    assert "watchdog:" not in out and "train exited with 75" in out and "not requeueing" in out


def test_wrapper_stops_after_20_restarts_without_requeueing(tmp_path):
    """H2. Something that can never finish inside the time limit (a scoring that needs 25 h)
    would otherwise requeue for ever, a GPU-day each time."""
    rc, log, out = _wrapper(tmp_path, {}, env_extra={"SLURM_RESTART_COUNT": "20"})
    assert rc == 0 and len(log) == 2 and "restart 20" in out, out          # 20 is still allowed
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True}},
                            env_extra={"SLURM_RESTART_COUNT": "21"})
    assert rc != 0 and log == [], out                    # python never started, nothing requeued
    assert "restarted 21 times" in out and "limit 20" in out and "NOT requeueing" in out


def test_wrapper_requeues_when_python_dies_of_the_signal_before_its_handlers_exist(tmp_path):
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True, "handle": False}},
                            signal_step="train")
    assert rc == 0 and log[1:] == ["scontrol requeue 777_4"], out   # killed by SIGTERM: rc 143


def test_wrapper_requeues_instead_of_scoring_when_signalled_as_training_finishes(tmp_path):
    """USR1 leaves 180 s: not enough to score. Training is complete on disk; the requeued
    task finds it 'already complete' and goes straight to scoring."""
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True, "rc_on_term": 0}},
                            signal_step="train")
    assert rc == 0 and log[1:] == ["train got SIGTERM", "scontrol requeue 777_4"], out


def test_wrapper_requeues_when_test_scoring_is_interrupted(tmp_path):
    rc, log, out = _wrapper(tmp_path, {"eval": {"block": True, "rc_on_term": 143}},
                            signal_step="eval")
    assert rc == 0 and log[2:] == ["eval got SIGTERM", "scontrol requeue 777_4"], out


@pytest.mark.parametrize("code", [1, 2, 3])
def test_wrapper_never_requeues_a_failure_even_at_the_time_limit(tmp_path, code):
    """Crash (1), refusal (2), preflight (3): they would fail again. Also when the failure
    coincides with the time-limit warning."""
    rc, log, out = _wrapper(tmp_path, {"train": {"rc": code}})
    assert rc == code and len(log) == 1, out                         # no scoring, no requeue
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True, "rc_on_term": code}},
                            signal_step="train")
    assert rc == code and _requeues(log) == [], out
    if code != 2:                                    # eval's 2 is the once-only refusal, below
        rc, log, out = _wrapper(tmp_path, {"eval": {"rc": code}})
        assert rc == code and _requeues(log) == [], out
        rc, log, out = _wrapper(tmp_path, {"eval": {"block": True, "rc_on_term": code}},
                                signal_step="eval")
        assert rc == code and _requeues(log) == [], out


def test_wrapper_exit_codes(tmp_path):
    rc, log, out = _wrapper(tmp_path, {"train": {"rc": 75}})        # 75 with no USR1 seen here:
    assert rc == 75 and _requeues(log) == [] and len(log) == 1, out  # not ours to requeue
    rc, log, out = _wrapper(tmp_path, {"eval": {"rc": 2, "touch": "test_metrics.json"}})
    assert rc == 0 and "already scored" in out, out                 # the once-only refusal
    (tmp_path / "runs" / "massachusetts_R1_s1" / "test_metrics.json").unlink()
    rc, log, out = _wrapper(tmp_path, {"eval": {"rc": 2}})
    assert rc == 2, out                              # exit 2 WITHOUT a result is a real failure
    rc, log, out = _wrapper(tmp_path, {"train": {"block": True}}, signal_step="train",
                            env_extra={"FAKE_SCONTROL_RC": "1"})
    assert rc != 0 and len(_requeues(log)) == 1, out  # could not requeue: do not report success


def test_wrapper_leaves_temp_files_to_the_process_that_holds_the_lock(tmp_path):
    """G2. The wrapper used to `rm` scoring temp files before starting eval; without the run's
    lock that could delete the file of a scorer still alive. roadseg.eval clears debris
    itself, under the lock (test_temp_names_are_unique_per_process_...)."""
    run = tmp_path / "runs" / "massachusetts_R1_s1"
    run.mkdir(parents=True)
    (run / ".test_metrics.json.cl014.4242.tmp").write_text("{")
    rc, log, out = _wrapper(tmp_path, {})
    assert rc == 0 and len(list(run.glob(".test_metrics.json.*.tmp"))) == 1, out


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


def test_an_explicit_cuda_device_never_falls_back_to_cpu(tiny, monkeypatch, capsys):
    """G3. A node with a broken GPU must fail the task at once, not train on CPU for 24 h."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(tr.PreflightFailed, match="cuda.*is_available.*False.*not fall"):
        tr.main(["--config", str(tiny.cfg()), "--rung", "R1", "--device", "cuda"])
    with pytest.raises(tr.PreflightFailed, match="not fall"):
        tr.main(["--config", str(tiny.cfg()), "--rung", "R1", "--device", "cuda:0", "--dry-run"])
    assert not tiny.runs.exists()                                   # nothing written
    assert tr.resolve_device(None) == "cpu" and tr.resolve_device("cpu") == "cpu"   # unchanged

    assert tiny.train() == 0                                        # a finished run, on CPU
    run = tiny.runs / "massachusetts_R1_s0"
    before = sorted(p.name for p in run.iterdir())
    with pytest.raises(tr.PreflightFailed, match="not fall"):
        ev.main(["--run", str(run), "--split", "test", "--tile", "32", "--stride", "16",
                 "--workers", "0", "--device", "cuda"])
    assert sorted(p.name for p in run.iterdir()) == before          # no result, no temp file


@pytest.mark.parametrize("module", ["roadseg.train", "roadseg.eval"])
def test_cli_with_device_cuda_exits_3_without_a_gpu(tmp_path, module):
    """The real CLIs, CUDA hidden: exit code 3 (never requeued by the wrapper), clear message."""
    runs = tmp_path / "runs"
    if module == "roadseg.train":
        cfg = write_cfg(tmp_path, tmp_path / "no-data-needed")
        args = ["--config", str(cfg), "--rung", "R1", "--resume", "auto", "--device", "cuda"]
    else:
        args = ["--run", str(runs / "massachusetts_R1_s0"), "--split", "test", "--device", "cuda"]
    r = subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True,
                       env=dict(ENV, CUDA_VISIBLE_DEVICES=""), cwd=ROOT)
    assert r.returncode == tr.EXIT_PREFLIGHT == 3, r.stdout + r.stderr
    assert "preflight failed: --device cuda" in r.stderr and "is_available() is False" in r.stderr
    assert "Traceback" not in r.stderr and not runs.exists()


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
import os, runpy, sys, time
import torch.nn as nn
import roadseg.models.segmenter as seg

DELAY = sys.argv[1]                 # seconds per training step, or a file: wait while it exists


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                                 nn.Conv2d(4, 1, 1))

    def forward(self, x):
        if self.training and os.sep in DELAY:
            while os.path.exists(DELAY):
                time.sleep(0.02)
        elif self.training:
            time.sleep(float(DELAY))            # stretch an optimizer step so it can be hit
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


def test_stop_workers_kills_first_and_never_waits_on_a_stuck_dataloader_shutdown(monkeypatch):
    """G1. Killing is the part that matters. DataLoader's own shutdown joins its pin-memory
    thread (CUDA only) without a timeout; if that ever blocks, main must still return."""
    import threading
    killed, gate = [], threading.Event()

    class Worker:
        alive = True

        def is_alive(self):
            return self.alive

        def kill(self):
            killed.append(self)
            self.alive = False

        def join(self, timeout=None):
            pass

    class Iterator:
        _worker_pids_set, _shutdown = False, False

        def __init__(self):
            self._workers = [Worker(), Worker()]

        def _shutdown_workers(self):
            gate.wait(120)                         # stuck, as in a thread join that never ends

    class Loader:
        pass

    loader = Loader()
    loader._iterator = it = Iterator()
    monkeypatch.setattr(tr, "TIDY_SECONDS", 0.3)
    t0 = time.time()
    tr.stop_workers(loader)
    assert time.time() - t0 < 5 and len(killed) == 2
    assert it._shutdown is True and loader._iterator is None      # its destructor is disarmed
    gate.set()
    tr.stop_workers(loader)                                        # nothing left: a no-op
    tr.stop_workers(None)


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


# ----------------------------------------------------------------------------- G2: one writer

LOCK_HOLDER = '''import os, signal, sys, time
from roadseg import runtime as rt

def main(argv):
    run, how, ready = sys.argv[1:4]
    rt.RunLock(run).acquire()
    if how == "sigkill" and os.fork() == 0:       # like a DataLoader worker: inherits the fd,
        signal.signal(signal.SIGTERM, signal.SIG_IGN)     # and outlives its SIGKILLed parent
        time.sleep(600)
        os._exit(0)
    open(ready, "w").close()
    while os.path.exists(ready):                  # hold the lock until told to end
        time.sleep(0.02)
    if how == "return": return 0
    if how == "requeue": return rt.EXIT_REQUEUE
    if how == "crash": raise RuntimeError("boom")
    time.sleep(600)

rt.run_cli(main)
'''
LOCK_TRY = ("import sys\nfrom roadseg import runtime as rt\n"
            "rt.run_cli(lambda argv: print('acquired', rt.RunLock(sys.argv[1]).acquire().path))\n")


@pytest.mark.parametrize("how, code", [("return", 0), ("requeue", 75), ("crash", 1),
                                       ("sigkill", -signal.SIGKILL)])
def test_run_lock_refuses_a_second_process_and_is_released_however_the_first_ends(tmp_path, how,
                                                                                  code):
    """G2. Exclusive while the holder lives; free after a normal exit, exit 75, a crash and
    SIGKILL -- the last even while a forked child still holds the inherited descriptor."""
    holder, attempt = tmp_path / "holder.py", tmp_path / "try.py"
    holder.write_text(LOCK_HOLDER)
    attempt.write_text(LOCK_TRY)
    run, ready = tmp_path / "runs" / "massachusetts_R1_s0", tmp_path / "ready"

    def second():
        return subprocess.run([sys.executable, str(attempt), str(run)], capture_output=True,
                              text=True, env=ENV, timeout=60)

    p = subprocess.Popen([sys.executable, str(holder), str(run), how, str(ready)], env=ENV,
                         stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        t0 = time.time()
        while not ready.exists():
            assert p.poll() is None and time.time() - t0 < 60, p.stderr.read()
            time.sleep(0.02)
        r = second()
        assert r.returncode == tr.EXIT_REFUSED == 2, r.stdout + r.stderr
        assert "is locked by another process" in r.stderr and str(run) in r.stderr
        assert f"pid {p.pid} " in r.stderr                       # says who holds it
        assert second().returncode == 2                          # and trying did not break it
        if how == "sigkill":
            p.kill()
        else:
            ready.unlink()
        assert p.wait(timeout=60) == code      # (not communicate: the orphan keeps the pipe open)
        r = second()
        assert r.returncode == 0 and "acquired" in r.stdout, r.stdout + r.stderr
    finally:
        if _group_alive(p.pid):
            os.killpg(p.pid, signal.SIGKILL)


def test_run_lock_inside_one_process(tmp_path):
    """POSIX locks do not conflict within a process and die with ANY close of the file, so
    a second holder in the same process is refused explicitly; release makes it free."""
    run = tmp_path / "run"
    lock = tr.RunLock(run).acquire()
    assert (run / ".lock").read_text().startswith(f"{socket.gethostname()} pid {os.getpid()} ")
    with pytest.raises(tr.Refused, match="locked by this very process"):
        tr.RunLock(run).acquire()
    lock.release()
    lock.release()                                               # idempotent
    with tr.RunLock(run):
        with pytest.raises(tr.Refused):
            tr.RunLock(run).acquire()
    tr.RunLock(run).acquire().release()


def test_releasing_the_lock_never_changes_the_exit_code(tiny, monkeypatch, tmp_path):
    """H3. release() runs in a `finally`: an OSError there (NFS) would turn exit 0 or 75
    into a crash. The kernel drops the lock at process exit anyway."""
    import roadseg.runtime as rt
    real_lockf, real_close = rt.fcntl.lockf, os.close
    failures = []

    def lockf(fd, op, *a):
        if op == rt.fcntl.LOCK_UN:
            failures.append("unlock")
            raise OSError(5, "Input/output error (simulated)")
        return real_lockf(fd, op, *a)

    monkeypatch.setattr(rt.fcntl, "lockf", lockf)
    _preempt_at_step(monkeypatch, 6)
    assert tiny.train() == 75                                    # not a crash
    assert tiny.train("--resume", "auto") == 0                   # and the lock was free again
    assert failures == ["unlock", "unlock"]

    lock = tr.RunLock(tmp_path / "run").acquire()                # ... and a failing close()
    fd = lock.fd

    def close(f):
        if f == fd:
            failures.append("close")
            raise OSError(116, "Stale file handle (simulated)")
        return real_close(f)

    monkeypatch.setattr(os, "close", close)
    lock.release()                                               # must not raise
    with tr.RunLock(tmp_path / "run"):                           # usable afterwards
        pass
    assert failures[-3:] == ["unlock", "close", "unlock"]
    monkeypatch.setattr(os, "close", real_close)
    real_close(fd)


def test_a_restarted_job_waits_for_a_dead_nodes_lock_but_a_first_start_does_not(tmp_path,
                                                                                monkeypatch):
    """H4. A node that died keeps its NFS lock until the lease expires (~90 s); the requeued
    task must not be refused for that. Only with SLURM_RESTART_COUNT > 0; bounded."""
    import roadseg.runtime as rt
    assert (rt.LOCK_RETRY_SECONDS, rt.LOCK_RETRY_INTERVAL) == (180.0, 15.0)
    holder = tmp_path / "holder.py"
    holder.write_text(LOCK_HOLDER)
    run, ready = tmp_path / "runs" / "massachusetts_R1_s0", tmp_path / "ready"
    p = subprocess.Popen([sys.executable, str(holder), str(run), "return", str(ready)], env=ENV)
    try:
        t0 = time.time()
        while not ready.exists():
            assert p.poll() is None and time.time() - t0 < 60
            time.sleep(0.02)

        def timed(**kw):
            t = time.monotonic()
            try:
                tr.RunLock(run).acquire(**kw).release()
                return "acquired", time.monotonic() - t
            except tr.Refused as e:
                assert "is locked by another process" in str(e) and f"pid {p.pid} " in str(e)
                return "refused", time.monotonic() - t

        monkeypatch.setattr(rt, "LOCK_RETRY_SECONDS", 0.6)
        monkeypatch.setattr(rt, "LOCK_RETRY_INTERVAL", 0.05)
        for count in (None, "0"):                                 # a first start: at once
            monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
            if count is not None:
                monkeypatch.setenv("SLURM_RESTART_COUNT", count)
            assert timed() == ("refused", pytest.approx(0, abs=0.2))
        monkeypatch.setenv("SLURM_RESTART_COUNT", "2")            # a restart: retries, bounded
        what, seconds = timed()
        assert what == "refused" and 0.6 <= seconds < 2
        assert timed(wait=0)[1] < 0.2                             # explicit parameters win

        import threading
        threading.Timer(0.5, ready.unlink).start()                # the lease "expires"
        what, seconds = timed(wait=10, interval=0.05)
        assert what == "acquired" and 0.4 <= seconds < 5
        assert p.wait(timeout=30) == 0
    finally:
        if p.poll() is None:
            p.kill()


def test_temp_names_are_unique_per_process_and_stale_ones_are_cleared_under_the_lock(tiny,
                                                                                     monkeypatch):
    """G2. Every temp file carries host and pid (two writers never share an inode), and the
    startup cleanup -- which runs holding the lock, so every temp it sees is dead -- removes
    the debris of any earlier process, whatever its name."""
    me = f"{socket.gethostname()}.{os.getpid()}.tmp"
    assert tr.tmp_name("/x/latest.pt").name == f".latest.pt.{me}"
    renamed, real_replace, real_link = [], os.replace, os.link
    monkeypatch.setattr(os, "replace", lambda s, d: (renamed.append((Path(s).name, Path(d).name)),
                                                     real_replace(s, d))[1])
    monkeypatch.setattr(os, "link", lambda s, d: (renamed.append((Path(s).name, Path(d).name)),
                                                  real_link(s, d))[1])
    assert tiny.train(cfg=tiny.cfg(epochs=2)) == 0
    run = tiny.runs / "massachusetts_R1_s0"
    assert ev.main(["--run", str(run), "--split", "test", "--tile", "32", "--stride", "16",
                    "--workers", "0"]) == 0
    assert {d for _, d in renamed} == {"latest.pt", "best.pt", "history.json", "metrics.json",
                                       "threshold.json", "test_metrics.json"}
    assert all(s == f".{d}.{me}" for s, d in renamed), renamed    # train, engine and eval
    assert sorted(p.name for p in run.iterdir()) == sorted([*RUN_FILES, "test_metrics.json"])

    debris = [".latest.pt.cl014.4242.tmp", ".best.pt.gl003.7.tmp", ".history.json.cl014.4242.tmp",
              ".test_metrics.json.cl014.99.tmp", "latest.tmp", ".history.json.tmp"]
    for name in debris:
        (run / name).write_bytes(b"torn")
    assert ev.main(["--run", str(run), "--split", "test", "--workers", "0"]) == 2   # scored once
    assert not [p.name for p in run.iterdir() if p.name.endswith(".tmp")]    # eval cleared them
    for name in debris:
        (run / name).write_bytes(b"torn")
    assert tiny.train("--resume", "auto", cfg=tiny.cfg(epochs=2)) == 0       # "already complete"
    assert not [p.name for p in run.iterdir() if p.name.endswith(".tmp")]    # train did too


@pytest.mark.slow
def test_a_second_process_on_a_running_run_is_refused_and_the_first_finishes_correctly(tmp_path):
    """G2 with the real CLI: while one training process owns the run directory, a second
    `roadseg.train` and a `roadseg.eval --split test` on it both exit 2 untouched."""
    boot = tmp_path / "boot.py"
    boot.write_text(BOOT)
    cfg = write_cfg(tmp_path, fake_mass(tmp_path / "data"), epochs=2, max_train_batches=6,
                    ckpt_minutes=0)
    env = dict(ENV, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    run = tmp_path / "runs" / "massachusetts_R1_s0"

    def cmd(delay, runs="runs", module="roadseg.train"):
        tail = (["--config", str(cfg), "--rung", "R1", "--device", "cpu", "--runs-dir",
                 str(tmp_path / runs), "--resume", "auto"] if module == "roadseg.train" else
                ["--run", str(run), "--split", "test", "--tile", "32", "--stride", "16",
                 "--workers", "0"])
        return [sys.executable, str(boot), str(delay), module, *tail]

    ref = subprocess.run(cmd(0, "ref"), capture_output=True, text=True, env=env, cwd=ROOT)
    assert ref.returncode == 0, ref.stdout + ref.stderr

    hold = tmp_path / "hold"
    hold.touch()                                 # the first process stops inside its first step
    first = subprocess.Popen(cmd(hold), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env=env, cwd=ROOT)
    try:
        t0 = time.time()
        while not ((run / ".lock").exists() and (run / ".lock").stat().st_size):
            assert first.poll() is None, first.communicate()
            assert time.time() - t0 < 600
            time.sleep(0.05)
        for module in ("roadseg.train", "roadseg.eval"):
            r = subprocess.run(cmd(0, module=module), capture_output=True, text=True, env=env,
                               cwd=ROOT)
            assert first.poll() is None, "the first process ended before the second was refused"
            assert r.returncode == 2, r.stdout + r.stderr
            assert "is locked by another process" in r.stderr and str(run) in r.stderr
            assert f"pid {first.pid} " in r.stderr and "Traceback" not in r.stderr
        hold.unlink()
        out, err = first.communicate(timeout=900)
    finally:
        if first.poll() is None:
            first.kill()
    assert first.returncode == 0, out + err
    assert_nothing_partial_is_visible(run)
    assert sorted(p.name for p in run.iterdir()) == RUN_FILES
    m, expected = read(run / "metrics.json"), read(tmp_path / "ref/massachusetts_R1_s0/metrics.json")
    assert m["resumes"] == 0 and m["history"] and m["threshold"] == expected["threshold"]
    assert [(r["epoch"], r["val_iou"], r["lr"]) for r in m["history"]] == [
        (r["epoch"], r["val_iou"], r["lr"]) for r in expected["history"]]
    assert weights_digest(run / "best.pt") == weights_digest(
        tmp_path / "ref/massachusetts_R1_s0/best.pt")           # undisturbed by the intruders
    r = subprocess.run(cmd(0), capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0 and "already complete" in r.stdout  # lock released on normal exit


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
def test_the_smoke_drills_rehearsed_with_a_job_wide_sigterm(tmp_path):
    """G7. The two preemption drills of the CPU rehearsal, as the script runs them: real
    ResNeXt, real data, DataLoader workers, SIGTERM to the whole process group."""
    runs = tmp_path / "runs_smoke"
    env = dict(ENV, ROADSEG_RUNS_SMOKE=str(runs), SMOKE_STEPS="drills", SMOKE_TILE="128",
               SMOKE_THREADS="4")
    env.pop("ROADSEG_CODE", None)
    r = subprocess.run(["bash", str(SMOKE)], capture_output=True, text=True, timeout=3000, env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-6000:]
    assert out.count("final state ok: 2 history rows") == 2 and "SMOKE: ALL PASSED" in out
    assert out.count("preempted: checkpoint saved at epoch") >= 2
    assert "bf16 autocast on CPU" not in out and "score the test split" not in out   # drills only
    assert "SMOKE_PASSED not written" in out and not (runs / "SMOKE_PASSED").exists()


@pytest.mark.slow
@needs_data
@needs_ram
def test_the_canary_script_rehearsed_on_cpu(tmp_path):
    """G6. scripts/sbatch_canary.sh will run for the first time on a GPU after a day in the
    queue; rehearse the script itself here -- real ResNeXt, real data, tiny sizes, on CPU:
    the preflight table, two miniature runs with test scoring, and the kill-and-resume drill."""
    runs = tmp_path / "runs_canary"
    env = dict(ENV, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CANARY_THREADS="4",
               ROADSEG_RUNS_CANARY=str(runs), CANARY_DEVICE="cpu", CANARY_NAMES="R3 unet",
               CANARY_TRAIN_EXTRA="--preflight --no-amp --tile 128 --batch-size 2",
               CANARY_MINI="--epochs 1 --max-train-batches 3 --limit-train 4 --limit-val 1 --workers 2",
               CANARY_EVAL_EXTRA="--tile 128 --stride 64 --workers 2", CANARY_DRILL_STEPS="2")
    env.pop("ROADSEG_CODE", None)
    r = subprocess.run(["bash", str(ROOT / "scripts/sbatch_canary.sh")], capture_output=True,
                       text=True, env=env, timeout=3000)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-6000:]
    assert r.stdout.rstrip().splitlines()[-1].startswith("CANARY: ALL PASSED"), out[-2000:]
    table = out[out.index("preflight table"):].splitlines()[1:4]
    assert table[0].split() == ["name", "steps/s", "peak", "GB"]
    assert [row.split()[0] for row in table[1:]] == ["R3", "unet"]
    assert all(float(row.split()[1]) > 0 for row in table[1:])              # steps/s parsed
    assert out.count("preflight ok:") == 2 + 2 + 2            # table, 2 mini runs, drill x2
    assert "exit 75" not in out and "preempted: checkpoint saved at epoch" in out
    assert "final state ok: 2 history rows" in out and "2 processes, 1 resume" in out
    for name in ("R1", "R3"):
        assert (runs / f"massachusetts_{name}_s0" / "test_metrics.json").exists()
    assert not list(runs.glob("PREFLIGHT_FAILED.*"))


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
