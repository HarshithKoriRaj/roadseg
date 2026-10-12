"""Training entrypoint. One config + one rung (or baseline) + one seed = one run directory.

    python -m roadseg.train --config configs/<dataset>.yaml (--rung R1 | --model unet) --seed 0 --resume auto

writes <runs_dir>/<dataset>_<name>_s<seed>/{latest.pt, best.pt, history.json, metrics.json}.

Guarantees the experiment relies on:

* The TEST split is never opened here. Only "train" and "val" are ever indexed;
  the checkpoint (best validation IoU at threshold 0.5) and the decision
  threshold are both fixed on validation and frozen into metrics.json, which is
  the only thing `python -m roadseg.eval --split test` will score from.
* Rungs are paired. Every training crop is a pure function of
  (seed, epoch, step, slot in the batch) -- see SeededCrops -- so for a given
  seed R0..R3 and the baselines see the identical crop sequence whatever their
  model construction drew from the RNG, whatever the worker count, and however
  often the job was preempted and resumed.
* Preemption safe. SIGTERM / SIGUSR1 only set a flag; the loop checkpoints after
  the current optimizer step and exits with 75 (EX_TEMPFAIL). latest.pt is also
  written every `ckpt_minutes` inside an epoch. Every file appears by an atomic
  rename of a fsynced temp file, so a kill at any instant leaves the previous
  consistent state, and `--resume auto` continues from it: the remaining steps
  of the interrupted epoch (the per-iteration LR schedule continues exactly),
  the remaining epochs, then the finish phase. metrics.json is published once.
* Every way out ends the process. The DataLoader workers ignore SIGTERM, so they
  are SIGKILLed on every exit path of `main` (stop_workers), and the CLI ends in
  `os._exit` (runtime.run_cli): a crash fails the job at once instead of hanging.

* One writer, one code. The run directory is locked (<run>/.lock) for the life of
  the process, and every temp file is named for its process. The identity of the
  code (the COMMIT file of a frozen snapshot, else git HEAD + dirty flag) is fixed
  at startup, stored in every checkpoint, and a resume under different code is
  refused unless --allow-code-change; metrics.json lists every process of the run.

Exit codes (defined in roadseg/runtime.py; the SLURM wrapper acts on them):

    0   EXIT_OK         done, or the run was already complete
    75  EXIT_REQUEUE    interrupted by SIGTERM/SIGUSR1, checkpoint saved: requeue me
    2   EXIT_REFUSED    refused: resume with a different configuration or different
                        code, run directory locked by another process, an
                        inconsistent run directory, a missing --resume checkpoint
                        (and argparse's usage error)
    3   EXIT_PREFLIGHT  the GPU is not usable for this run
    1   EXIT_CRASHED    anything else: uncaught exception, traceback on stderr

Only 75 is worth running again unchanged; 1, 2 and 3 would repeat themselves.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Sampler

from .data.dataset import RoadFullDataset, RoadTileDataset
from .engine import load_checkpoint, save_checkpoint, train_one_epoch
from .eval import evaluate, predict_full, resolve_device, select_threshold
from .losses import ComboLoss
from .models.segmenter import build_model
from .runtime import (EXIT_CRASHED, EXIT_OK, EXIT_PREFLIGHT, EXIT_REFUSED,  # noqa: F401
                      EXIT_REQUEUE, PreflightFailed, Refused, RunLock, clear_stale_temps,
                      code_identity, describe_code, run_cli, same_code, tmp_name)

PROJECT = Path(__file__).resolve().parents[2]
RUNGS = ("R0", "R1", "R2", "R3")
BASELINES = ("unet", "dlinknet")
DATASETS = ("massachusetts", "deepglobe")
LR_FLOOR = 0.01                       # warmup starts at, and cosine decays to, LR_FLOOR * lr

_REQUIRED = ("dataset", "data_root", "epochs", "tile", "stride", "val_stride", "batch_size",
             "lr", "weight_decay", "warmup_epochs", "road_bias", "num_workers", "amp",
             "preload", "seed", "runs_dir")
_DEFAULTS = {"model": "roadsegmenter", "rung": "R1", "pretrained": True, "hf_init": "exact",
             "ckpt_minutes": 10, "eval_workers": 8,
             "max_train_batches": None, "limit_train": None, "limit_val": None}
# Keys that may differ between the process that wrote a checkpoint and the one resuming it:
# they change how fast the run goes or where it lives, never what it computes.
_RESUME_FREE = frozenset({"num_workers", "eval_workers", "preload", "ckpt_minutes", "runs_dir"})


# --------------------------------------------------------------------------- config

def load_config(path) -> dict:
    """YAML config; `_base: other.yaml` (relative to this file) is loaded first and overridden."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("_base", None)
    if base is not None:
        merged = load_config(path.parent / base)
        merged.update(cfg)
        cfg = merged
    return cfg


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train one (dataset, rung | baseline, seed) run.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--rung", default=None, help="R0..R3 (roadsegmenter only)")
    ap.add_argument("--model", default=None, help="roadsegmenter | unet | dlinknet")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None, help="'auto' (= <run>/latest.pt if present) or a path")
    ap.add_argument("--dry-run", action="store_true",
                    help="build model, loss, optimizer and schedule, print them, touch no data")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-train-batches", type=int, default=None, help="smoke: steps per epoch cap")
    ap.add_argument("--limit-train", type=int, default=None, help="smoke: first N training images")
    ap.add_argument("--limit-val", type=int, default=None, help="smoke: first N validation images")
    ap.add_argument("--tile", type=int, default=None,
                    help="training/inference tile; stride and val_stride scale with it")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader workers; also caps the metric process pool (0 = all in-process)")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--device", default=None,
                    help="default: cuda if available, else cpu. An explicit cuda device never "
                         "falls back to the CPU (exit 3 if CUDA is unavailable)")
    ap.add_argument("--no-preload", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="run the device self-test before the run even on a CPU (on CUDA it "
                         "always runs: 2 optimizer steps + 1 eval batch on a throwaway model)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="no self-test, and ignore PREFLIGHT_FAILED.* files in the runs dir")
    ap.add_argument("--preflight-only", action="store_true",
                    help="run the self-test for this configuration and exit (0, or 3)")
    ap.add_argument("--allow-code-change", action="store_true",
                    help="resume although the code differs from the code that wrote the "
                         "checkpoint (both identities are recorded in metrics.json)")
    return ap


def resolve_config(args, error=None) -> dict:
    """File config + CLI overrides -> the complete config that is recorded with the run.

    `error(msg)` is called on an invalid combination (argparse's `error` in the CLI).
    """
    def fail(msg):
        if error is not None:
            error(msg)
        raise ValueError(msg)

    cfg = {**_DEFAULTS, **load_config(args.config)}
    for key, value in (("seed", args.seed), ("epochs", args.epochs),
                       ("batch_size", args.batch_size), ("num_workers", args.workers),
                       ("runs_dir", args.runs_dir), ("max_train_batches", args.max_train_batches),
                       ("limit_train", args.limit_train), ("limit_val", args.limit_val)):
        if value is not None:
            cfg[key] = value
    if args.no_preload:
        cfg["preload"] = False
    if args.no_amp:
        cfg["amp"] = False
    if args.no_pretrained:
        cfg["pretrained"] = False
    missing = [k for k in _REQUIRED if cfg.get(k) is None]
    if missing:
        fail(f"config {args.config} is missing {missing}")
    if args.workers is not None:
        cfg["eval_workers"] = min(cfg["eval_workers"], args.workers)

    model = str(args.model or cfg["model"]).lower()
    if model == "roadsegmenter":
        rung = args.rung or cfg["rung"]
        if rung not in RUNGS:
            fail(f"unknown rung {rung!r}; expected one of {list(RUNGS)}")
        name = rung
    elif model in BASELINES:
        if args.rung is not None:
            fail(f"--rung applies to roadsegmenter only, not to the {model} baseline")
        rung, name = None, model
    else:
        fail(f"unknown model {model!r}; expected roadsegmenter or one of {list(BASELINES)}")
    cfg.update(model=model, rung=rung, name=name)
    if cfg["dataset"] not in DATASETS:
        fail(f"unknown dataset {cfg['dataset']!r}; expected one of {list(DATASETS)}")

    if args.tile is not None and args.tile != cfg["tile"]:
        # keep the tile/stride geometry of the config (512/256/512 -> 256/128/256)
        for k in ("stride", "val_stride"):
            cfg[k] = max(1, cfg[k] * args.tile // cfg["tile"])
        cfg["tile"] = args.tile
    if cfg["tile"] % 32:
        fail(f"tile must be a multiple of 32 (encoder stride), got {cfg['tile']}")
    for k in ("stride", "val_stride"):
        if not 0 < cfg[k] <= cfg["tile"]:
            fail(f"need 0 < {k} <= tile, got {k}={cfg[k]}, tile={cfg['tile']}")
    for k in ("epochs", "batch_size"):
        if cfg[k] < 1:
            fail(f"{k} must be >= 1, got {cfg[k]}")
    cfg["seed"] = int(cfg["seed"])
    return cfg


def run_name(cfg: dict) -> str:
    """<dataset>_<name>_s<seed>; <name> is the rung, or the model name for a baseline."""
    return f"{cfg['dataset']}_{cfg['name']}_s{cfg['seed']}"


def config_diff(old: dict, new: dict) -> list[str]:
    """Differences that would make `new` a different experiment from `old`."""
    keys = (set(old) | set(new)) - _RESUME_FREE
    return [f"{k}: {old.get(k)!r} -> {new.get(k)!r}" for k in sorted(keys)
            if old.get(k) != new.get(k)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- schedule

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Per-ITERATION schedule: linear warmup from LR_FLOOR*lr to lr over `warmup_steps`,
    then cosine to LR_FLOOR*lr over the remaining steps. Step it once per optimizer step.

    Step k (0-based) of training runs at the LR after k scheduler steps: LR_FLOOR*lr at
    k=0, lr at k=warmup_steps, LR_FLOOR*lr after `total_steps` steps.
    """
    if total_steps < 1 or warmup_steps < 0:
        raise ValueError(f"need total_steps >= 1 and warmup_steps >= 0, got {total_steps}, {warmup_steps}")
    base = optimizer.param_groups[0]["lr"]
    parts = []
    if warmup_steps > 0:
        parts.append(LinearLR(optimizer, start_factor=LR_FLOOR, end_factor=1.0,
                              total_iters=warmup_steps))
    if total_steps > warmup_steps:
        parts.append(CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                                       eta_min=LR_FLOOR * base))
    if len(parts) == 2:
        return SequentialLR(optimizer, parts, milestones=[warmup_steps])
    return parts[0]


# --------------------------------------------------------------------------- data stream

def _mix(seed: int, token: int) -> int:
    """splitmix64 finaliser of (seed, token) -> a 63-bit RNG seed."""
    m = (1 << 64) - 1
    z = ((int(seed) << 40) ^ int(token)) & m
    z = (z + 0x9E3779B97F4A7C15) & m
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
    return (z ^ (z >> 31)) >> 1


class SeededCrops(Dataset):
    """Makes every training crop a pure function of (run seed, token).

    RoadTileDataset ignores its index and seeds each crop with one draw from the
    process-global torch RNG. Left alone, that stream depends on things that
    must not matter: with workers=0 it is the main process's RNG, which each
    rung's model construction has advanced by a different amount (R1/R2 build a
    throwaway stem, R0 does not); with workers>0 it restarts from the top after
    every resume. Here the global generator is seeded from the token for the
    duration of the one call and then put back, so the crop depends on nothing
    else. EpochTokens numbers the tokens by position in training.
    """

    def __init__(self, base: Dataset, seed: int) -> None:
        self.base, self.seed = base, int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, token: int) -> dict:
        g = torch.default_generator
        state = g.get_state()
        try:
            g.manual_seed(_mix(self.seed, int(token)))
            return self.base[int(token) % len(self.base)]
        finally:
            g.set_state(state)


class EpochTokens(Sampler):
    """Tokens of steps [start_step, steps_per_epoch) of one epoch, in order.

    token = (epoch * steps_per_epoch + step) * batch_size + slot: a global
    sample counter, so a resumed epoch continues with exactly the crops the
    interrupted one would have drawn. Call `set` before each pass.
    """

    def __init__(self, steps_per_epoch: int, batch_size: int) -> None:
        self.steps, self.batch_size = int(steps_per_epoch), int(batch_size)
        self.epoch, self.start = 0, 0

    def set(self, epoch: int, start_step: int = 0) -> None:
        self.epoch, self.start = int(epoch), int(start_step)

    def __iter__(self):
        first = self.epoch * self.steps * self.batch_size
        return iter(range(first + self.start * self.batch_size,
                          first + self.steps * self.batch_size))

    def __len__(self) -> int:
        return (self.steps - self.start) * self.batch_size


def _worker_init(worker_id: int) -> None:
    # SLURM signals every process of a preempted job. A worker that died of SIGTERM
    # would make the DataLoader raise in the main process before it could checkpoint.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


def stop_workers(loader) -> None:
    """End the DataLoader's worker processes for certain. Call on EVERY way out of main.

    The workers ignore SIGTERM (above). Everything that would normally end them at
    exit uses SIGTERM: DataLoader's own fallback (`terminate`) and multiprocessing's
    exit handler, which then joins them with no timeout. So after an uncaught
    exception -- the traceback keeps the loader alive until interpreter shutdown --
    the process used to hang for ever, holding its GPU. SIGKILL cannot be ignored,
    and a worker only reads, so nothing is lost by it.
    """
    it = getattr(loader, "_iterator", None)
    if it is None:
        return
    workers = list(getattr(it, "_workers", None) or ())
    try:        # or torch's SIGCHLD handler raises "DataLoader worker killed" in this thread
        if getattr(it, "_worker_pids_set", False):
            torch.utils.data._utils.signal_handling._remove_worker_pids(id(it))
            it._worker_pids_set = False
    except Exception:
        pass
    for w in workers:
        try:
            if w.is_alive():
                w.kill()
        except Exception:
            pass
    for w in workers:
        try:
            w.join(timeout=10)
        except Exception:
            pass
    try:
        it._shutdown_workers()          # queues and the pin-memory thread; the workers are gone
    except Exception:
        pass
    loader._iterator = None


# --------------------------------------------------------------------------- durable writes

def write_json_atomic(path, obj) -> None:
    """Temp file, flush + fsync, atomic rename: readers see the old file or the new one."""
    path = Path(path)
    tmp = tmp_name(path)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def publish_json_once(path, obj) -> bool:
    """Like write_json_atomic, but never replaces an existing file (hard link, which
    fails if the target exists). Returns False if `path` was already there."""
    path = Path(path)
    tmp = tmp_name(path)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        return False
    finally:
        tmp.unlink(missing_ok=True)
    return True


# --------------------------------------------------------------------------- preemption

class Preempt:
    """Truthy once SIGTERM or SIGUSR1 has arrived. The handlers do nothing else."""

    SIGNALS = (signal.SIGTERM, signal.SIGUSR1)

    def __init__(self) -> None:
        self.signum = None
        self._previous = {}

    def _handle(self, signum, frame) -> None:
        self.signum = signum

    def install(self) -> "Preempt":
        for s in self.SIGNALS:
            self._previous[s] = signal.signal(s, self._handle)
        return self

    def restore(self) -> None:
        for s, h in self._previous.items():
            signal.signal(s, h)
        self._previous = {}

    def __bool__(self) -> bool:
        return self.signum is not None


# --------------------------------------------------------------------------- training state

def fresh_state() -> dict:
    """`epoch` is the first epoch that is not complete; `step_in_epoch` the optimizer
    steps of it already taken."""
    return {"epoch": 0, "step_in_epoch": 0, "history": [], "best": -1.0, "best_epoch": None,
            "wall_seconds": 0.0, "resumes": 0, **fresh_provenance()}


def fresh_provenance() -> dict:
    """`code`: identity of the code that wrote the checkpoint (runtime.code_identity).
    `processes`: one entry per process that worked on the run. `code_changes`: every
    resume under different code that --allow-code-change let through."""
    return {"code": None, "processes": [], "code_changes": []}


def save_state(path, model, optimizer, scheduler, cfg: dict, state: dict,
               steps_per_epoch: int) -> None:
    save_checkpoint(path, model, optimizer, scheduler, epoch=state["epoch"], best=state["best"],
                    cfg=cfg, extra={**state, "history": list(state["history"]),
                                    "steps_per_epoch": steps_per_epoch})


def load_state(path, model, optimizer, scheduler, cfg: dict, steps_per_epoch: int,
               map_location="cpu", *, code: dict | None = None,
               allow_code_change: bool = False, log=print) -> dict:
    """Restore model, optimizer, scheduler and the training state from `path`.

    Refuses a checkpoint from a different experiment -- checked BEFORE anything is
    loaded: the scheduler state only means something for the same lr, epochs and
    steps per epoch. With `code` (this process's identity) it also refuses a
    checkpoint written by different code, or by code of unrecorded identity:
    one run, one code. `allow_code_change` lets it through and records both.
    """
    head = torch.load(path, map_location="cpu", weights_only=False)
    extra, old_cfg = head.get("extra", {}), head.get("cfg", {})
    del head
    if "step_in_epoch" not in extra:
        raise Refused(f"refusing to resume: {path} carries no training state")
    diff = config_diff(old_cfg, cfg)
    if extra.get("steps_per_epoch") != steps_per_epoch:
        diff.append(f"steps_per_epoch: {extra.get('steps_per_epoch')!r} -> {steps_per_epoch!r}")
    if diff:
        raise Refused(f"refusing to resume: {path} was written by a different configuration ("
                      + "; ".join(diff) + ")")
    change = None
    if code is not None and not same_code(extra.get("code"), code):
        what = (f"the code changed since {path} was written (checkpoint: "
                f"{describe_code(extra.get('code'))}; now: {describe_code(code)})")
        if not allow_code_change:
            raise Refused(f"refusing to resume: {what}. A run must be one code from start to "
                          f"finish. Start the run again from this code, or pass "
                          f"--allow-code-change to continue (both are recorded).")
        log(f"WARNING: {what}; continuing because of --allow-code-change", flush=True)
        change = {"epoch": extra["epoch"], "step_in_epoch": extra["step_in_epoch"],
                  "from": extra.get("code"), "to": code}
    load_checkpoint(path, model, optimizer, scheduler, map_location=map_location)
    state = {k: extra[k] for k in fresh_state() if k not in fresh_provenance()}
    state["history"] = list(state["history"])
    state["resumes"] += 1
    state["processes"] = [dict(p) for p in extra.get("processes", [])]
    state["code_changes"] = [dict(c) for c in extra.get("code_changes", [])]
    if change is not None:
        state["code_changes"].append(change)
    state["code"] = code if code is not None else extra.get("code")
    return state


def repair_best(run: Path, model, cfg: dict, state: dict, log=print) -> None:
    """Make best.pt agree with the state just loaded from latest.pt.

    At the end of an epoch latest.pt is written first and best.pt second. A kill
    between the two leaves a latest.pt that names the epoch just finished as the
    best while best.pt is still the previous one (or absent). latest.pt then
    holds exactly the weights that were validated, so best.pt is rewritten from
    the model loaded from it. No other state can have the two disagree.
    """
    best_epoch = state["best_epoch"]
    if best_epoch is None:
        return
    path = run / "best.pt"
    on_disk = (torch.load(path, map_location="cpu", weights_only=False).get("epoch")
               if path.exists() else None)
    if on_disk == best_epoch:
        return
    if state["step_in_epoch"] == 0 and best_epoch == state["epoch"] - 1:
        save_checkpoint(path, model, epoch=best_epoch, best=state["best"], cfg=cfg)
        log(f"repaired best.pt from latest.pt (epoch {best_epoch}; was {on_disk})", flush=True)
        return
    raise Refused(f"{path} holds epoch {on_disk} but latest.pt says the best epoch is "
                     f"{best_epoch}: the run directory is inconsistent, refusing to continue")


def fit(*, model, optimizer, scheduler, loss_fn, batches_for, validate, run: Path, cfg: dict,
        state: dict, steps_per_epoch: int, use_amp: bool, device: str, preempt,
        ckpt_seconds: float, saved: bool = False, clock=time.monotonic, log=print) -> bool:
    """Train from `state` until cfg["epochs"] epochs are complete.

    Returns True when they all are, False when `preempt` became truthy (latest.pt
    then holds the exact point reached). `batches_for(epoch, start_step)` yields
    the batches of the remaining steps of `epoch`; `validate(model)` returns the
    validation IoU used for checkpoint selection. `saved` says latest.pt already
    holds `state` (just resumed from it). `state` is updated in place.
    """
    run = Path(run)
    latest = run / "latest.pt"
    t0, wall0 = clock(), state["wall_seconds"]
    last_save = clock()

    def save() -> None:
        nonlocal last_save, saved
        state["wall_seconds"] = wall0 + (clock() - t0)
        if state.get("processes"):          # this process's entry: the last epoch it worked on
            state["processes"][-1]["last_epoch"] = (
                state["epoch"] if state["step_in_epoch"] > 0 else max(state["epoch"] - 1, 0))
        save_state(latest, model, optimizer, scheduler, cfg, state, steps_per_epoch)
        last_save, saved = clock(), True

    while True:
        epoch, start = state["epoch"], state["step_in_epoch"]
        if preempt:
            if not saved:
                save()
            log(f"preempted: checkpoint saved at epoch {epoch} step {start}", flush=True)
            return False
        if epoch >= cfg["epochs"]:
            return True

        t_epoch = clock()
        remaining = steps_per_epoch - start
        ran = 0
        stopped = False

        def on_step(i: int, loss: float) -> bool:
            nonlocal ran, stopped, saved
            ran = i + 1
            state["step_in_epoch"] = start + ran
            saved = False
            if preempt:
                save()
                stopped = True
                return True
            if clock() - last_save >= ckpt_seconds:
                save()
            return False

        train_loss = None
        if remaining > 0:
            train_loss = train_one_epoch(model, batches_for(epoch, start), loss_fn, optimizer,
                                         use_amp, device, scheduler=scheduler,
                                         max_batches=remaining, on_step=on_step)
        if stopped:
            log(f"preempted: checkpoint saved at epoch {epoch} step {start + ran}", flush=True)
            return False
        if ran != remaining:
            raise RuntimeError(f"epoch {epoch}: the loader yielded {ran} batches, expected "
                               f"{remaining}; the LR schedule would fall out of step")

        val_iou = float(validate(model))
        row = {"epoch": epoch, "train_loss": train_loss, "val_iou": val_iou,
               "lr": float(scheduler.get_last_lr()[0]), "seconds": round(clock() - t_epoch, 3)}
        if start > 0:                    # train_loss / seconds cover this process's steps only
            row.update(resumed=True, train_steps=remaining)
        improved = val_iou > state["best"]              # NaN never compares greater
        state["history"].append(row)
        if improved:
            state["best"], state["best_epoch"] = val_iou, epoch
        state["epoch"], state["step_in_epoch"] = epoch + 1, 0
        # Order matters for crash safety: latest.pt (the weights just validated, with the
        # updated record) first, best.pt second -- see repair_best -- history.json last.
        save()
        if improved:
            save_checkpoint(run / "best.pt", model, epoch=epoch, best=val_iou, cfg=cfg)
        write_json_atomic(run / "history.json", state["history"])
        loss_txt = "n/a" if train_loss is None else f"{train_loss:.4f}"
        log(f"epoch {epoch} train_loss {loss_txt} val_iou {val_iou:.4f} lr {row['lr']:.3e} "
            f"{row['seconds']:.1f}s" + (" [best]" if improved else "")
            + (f" [resumed at step {start}]" if start > 0 else ""), flush=True)


# --------------------------------------------------------------------------- finish

def finish(*, model, val_ds, run: Path, cfg: dict, state: dict, device: str, use_amp: bool,
           eval_workers: int, extra: dict, device_name: str = "cpu", log=print) -> bool:
    """Freeze checkpoint and threshold on VALIDATION and publish metrics.json once.

    Pure function of best.pt and the validation set, so a kill anywhere in here is
    repaired by running it again. Returns False if metrics.json already existed.
    Provenance comes from `state` (fixed at startup): the code this process runs,
    every process that worked on the run, and any allowed code change.
    """
    run = Path(run)
    code = state.get("code") or {"commit": "unknown", "dirty": None, "source": "none"}
    if state["best_epoch"] is None:
        raise SystemExit("no epoch produced a finite validation IoU; there is no best.pt to score")
    meta = load_checkpoint(run / "best.pt", model, map_location=device)
    if meta["epoch"] != state["best_epoch"]:
        raise Refused(f"best.pt holds epoch {meta['epoch']}, expected {state['best_epoch']}")
    geometry = dict(tile=cfg["tile"], stride=cfg["stride"], use_amp=use_amp)
    t0 = time.monotonic()
    threshold, threshold_iou = select_threshold(model, val_ds, device, **geometry)
    agg, rows = evaluate(model, val_ds, threshold, device, full_metrics=True, with_apls=True,
                         workers=eval_workers, **geometry)
    payload = {
        "cfg": cfg, "threshold": threshold, "threshold_val_iou": threshold_iou,
        "val": agg, "per_image": rows,
        "best_epoch": state["best_epoch"], "best_val_iou": state["best"],
        "history": state["history"], "resumes": state["resumes"],
        "code_commit": code["commit"], "code_dirty": code["dirty"], "code_source": code["source"],
        "git_commit": code["commit"], "git_dirty": code["dirty"],        # the older names
        "processes": state.get("processes", []), "code_changes": state.get("code_changes", []),
        "torch_version": torch.__version__,
        "device_name": device_name,
        "hostname": socket.gethostname(),
        "wall_seconds": round(state["wall_seconds"] + (time.monotonic() - t0), 3),
        **extra,
    }
    published = publish_json_once(run / "metrics.json", payload)
    log(f"done: best val IoU {state['best']:.4f} (epoch {state['best_epoch']}), threshold "
        f"{threshold:.2f} -> val IoU {agg['dataset_iou']:.4f}; "
        + (f"wrote {run / 'metrics.json'}" if published else "metrics.json already existed"),
        flush=True)
    return published


# --------------------------------------------------------------------------- preflight

PREFLIGHT_STEPS = 2          # optimizer steps at the configured batch size and tile
PREFLIGHT_EVAL_BATCH = 8     # tiles in the one eval-mode forward (predict_full's batch)
SENTINEL = "PREFLIGHT_FAILED."


def preflight(cfg: dict, device: str, use_amp: bool) -> dict:
    """Self-test of exactly what this run will ask of the device, on a THROWAWAY model.

    A throwaway instance of the run's architecture takes PREFLIGHT_STEPS real optimizer
    steps (engine.train_one_epoch: bf16 autocast, the run's loss -- clDice for R3 --, the
    gradient guard, AdamW) at the configured batch size and tile on random tensors, then
    one eval-mode forward of PREFLIGHT_EVAL_BATCH tiles through predict_full. Anything
    that would kill the job at step 0 -- out of memory at this batch size, an operator
    without a bf16 kernel, a non-finite loss -- raises here, in the first minute.

    It cannot change the run: the random tensors come from private generators, the
    throwaway model loads no pretrained weights and is freed, and the real model is seeded
    (set_seed) and built only afterwards. Returns {name, batch, tile, amp, steps_per_s,
    peak_gb, seconds}; peak_gb is torch.cuda.max_memory_allocated (0 on a CPU).
    """
    is_cuda = str(device).startswith("cuda")
    b, t = int(cfg["batch_size"]), int(cfg["tile"])
    g = torch.Generator().manual_seed(0)
    t0 = time.monotonic()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    model = build_model({**cfg, "pretrained": False}).to(device)
    loss_fn = ComboLoss(use_cldice=(cfg["rung"] == "R3"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    batches = [{"image": torch.randn(b, 3, t, t, generator=g),
                "mask": (torch.rand(b, 1, t, t, generator=g) > 0.9).float(),
                "valid": torch.ones(b, 1, t, t)} for _ in range(PREFLIGHT_STEPS)]
    stamps = [time.monotonic()]

    def on_step(i: int, loss: float) -> None:
        if is_cuda:
            torch.cuda.synchronize()
        stamps.append(time.monotonic())

    train_one_epoch(model, batches, loss_fn, optimizer, use_amp, device,
                    max_batches=PREFLIGHT_STEPS, on_step=on_step)
    if len(stamps) != PREFLIGHT_STEPS + 1:
        raise RuntimeError(f"preflight took {len(stamps) - 1} steps, expected {PREFLIGHT_STEPS}")
    # 2 x 4 tiles with stride == tile: exactly one batch of 8 through the inference path
    image = np.random.default_rng(0).standard_normal((2 * t, 4 * t, 3)).astype(np.float32)
    prob = predict_full(model, image, tile=t, stride=t, device=device,
                        batch=PREFLIGHT_EVAL_BATCH, use_amp=use_amp)
    if prob.shape != (2 * t, 4 * t) or not np.isfinite(prob).all():
        raise FloatingPointError(f"preflight inference gave shape {prob.shape}, "
                                 f"finite={bool(np.isfinite(prob).all())}")
    result = {"name": cfg["name"], "batch": b, "tile": t, "amp": bool(use_amp),
              # the first step pays for cuDNN autotuning and allocation; rate the last one
              "steps_per_s": 1.0 / max(stamps[-1] - stamps[-2], 1e-9),
              "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if is_cuda else 0.0,
              "seconds": time.monotonic() - t0}
    del model, loss_fn, optimizer, batches, prob, image
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return result


def run_preflight(cfg: dict, device: str, use_amp: bool, *, code: dict, device_name: str,
                  log=print) -> dict:
    """`preflight`, reported. On ANY failure: write <runs_dir>/PREFLIGHT_FAILED.<run>.txt
    (which makes every other task of the launch stop at startup -- see check_sentinels)
    and raise PreflightFailed (exit code 3, never requeued)."""
    try:
        r = preflight(cfg, device, use_amp)
    except Exception as exc:
        name = run_name(cfg)
        sentinel = Path(cfg["runs_dir"]) / f"{SENTINEL}{name}.txt"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(
            f"run: {name}\nhost: {socket.gethostname()}\ndevice: {device} ({device_name})\n"
            f"commit: {describe_code(code)}\ntime: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
            f"job: {os.environ.get('SLURM_JOB_ID', '-')}\n"
            f"batch_size: {cfg['batch_size']}  tile: {cfg['tile']}  amp: {use_amp}\n\n"
            f"{traceback.format_exc()}\n"
            f"While this file exists every `roadseg.train` in {sentinel.parent} exits 3 at "
            f"startup. Fix the cause, delete the file, and launch again.\n")
        raise PreflightFailed(
            f"{type(exc).__name__}: {exc} -- the self-test of {name} on {device_name} failed "
            f"before the run started; details in {sentinel}") from exc
    log(f"preflight ok: name={r['name']} batch={r['batch']} tile={r['tile']} amp={r['amp']} "
        f"steps_per_s={r['steps_per_s']:.2f} peak_gb={r['peak_gb']:.2f} "
        f"eval_batch={PREFLIGHT_EVAL_BATCH} seconds={r['seconds']:.1f} device={device_name}",
        flush=True)
    return r


def check_sentinels(runs_dir) -> None:
    """Exit 3 if any task's preflight has failed in this runs directory: the other tasks
    of the launch would fail the same way, each after its own day in the queue."""
    found = sorted(Path(runs_dir).glob(f"{SENTINEL}*")) if Path(runs_dir).is_dir() else []
    if found:
        raise PreflightFailed(
            f"{found[0]} exists" + (f" (and {len(found) - 1} more)" if len(found) > 1 else "")
            + ": a preflight failed in this runs directory, so no run starts here. Read it, "
              "fix the cause, delete it, launch again (--skip-preflight overrides).")


# --------------------------------------------------------------------------- entrypoint

def main(argv=None) -> int:
    ap = _parser()
    args = ap.parse_args(argv)
    cfg = resolve_config(args, error=ap.error)
    if args.preflight_only and args.skip_preflight:
        ap.error("--preflight-only and --skip-preflight contradict each other")
    run = Path(cfg["runs_dir"]) / run_name(cfg)
    device = resolve_device(args.device)       # an explicit cuda never falls back to the CPU
    use_amp = bool(cfg["amp"])
    # Provenance is fixed ONCE, here, before any work: what this process imported (never
    # re-read later, when the live tree may have moved on) and what it runs on. Both calls
    # can fail, and must fail now rather than after a day of training.
    code = code_identity(PROJECT)
    device_name = (torch.cuda.get_device_name(torch.device(device))
                   if str(device).startswith("cuda") else "cpu")
    print(f"rung={cfg['rung'] or '-'} seed={cfg['seed']} dataset={cfg['dataset']} "
          f"model={cfg['model']} device={device} amp={use_amp} run={run}", flush=True)
    print(f"code={code['commit'][:12]}{'+dirty' if code['dirty'] else ''} ({code['source']}) "
          f"host={socket.gethostname()} device_name={device_name} pid={os.getpid()}", flush=True)

    def build():
        set_seed(cfg["seed"])                  # immediately before build_model, nothing between
        model = build_model(cfg).to(device)
        loss_fn = ComboLoss(use_cldice=(cfg["rung"] == "R3"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])
        return model, loss_fn, optimizer

    if args.dry_run:                           # touches nothing on disk: no lock, no directory
        model, loss_fn, optimizer = build()
        steps = cfg["max_train_batches"] or 100              # nominal: no data is read
        scheduler = build_scheduler(optimizer, round(cfg["warmup_epochs"] * steps),
                                    cfg["epochs"] * steps)
        n = sum(p.numel() for p in model.parameters())
        print(f"dry-run ok: {n / 1e6:.1f}M params, loss={type(loss_fn).__name__}"
              f"(use_cldice={loss_fn.use_cldice}), {type(optimizer).__name__} "
              f"lr={cfg['lr']} wd={cfg['weight_decay']}, {type(scheduler).__name__} "
              f"warmup_epochs={cfg['warmup_epochs']} epochs={cfg['epochs']}", flush=True)
        return EXIT_OK

    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.backends.cudnn.benchmark = True      # before the preflight: same conv algorithms
    if args.preflight_only:                        # the canary: test this configuration, exit
        run_preflight(cfg, device, use_amp, code=code, device_name=device_name)
        return EXIT_OK
    # Every job on a GPU tests the device before the real model exists (--preflight forces
    # it on a CPU). A failure anywhere stops every task that starts afterwards.
    self_test = None
    if not args.skip_preflight:
        check_sentinels(cfg["runs_dir"])
        if is_cuda or args.preflight:
            def self_test():
                run_preflight(cfg, device, use_amp, code=code, device_name=device_name)

    # One writer per run directory, from before any state is read until the process ends
    # (the kernel drops the lock on any exit, SIGKILL included). Refused -> exit code 2.
    lock = RunLock(run).acquire()
    try:
        # Under the lock every temp file is the debris of a dead process: a half-written
        # checkpoint is never read, but it is 300 MB.
        clear_stale_temps(run)
        return _train(args, cfg, run, device, build, code, device_name, self_test)
    finally:
        lock.release()


def _train(args, cfg: dict, run: Path, device: str, build, code: dict, device_name: str,
           self_test=None) -> int:
    """Everything `main` does to the run directory. The caller holds the run's lock.
    `self_test()` (the preflight, or None) runs before the real model is seeded and built."""
    is_cuda = str(device).startswith("cuda")
    use_amp = bool(cfg["amp"])
    if (run / "metrics.json").exists():
        done = json.loads((run / "metrics.json").read_text())
        diff = config_diff(done.get("cfg", {}), cfg)
        if diff:
            raise Refused(f"refusing: {run / 'metrics.json'} exists for a different "
                          f"configuration ({'; '.join(diff)})")
        print(f"already complete: {run / 'metrics.json'} exists; nothing to do", flush=True)
        return EXIT_OK

    resume = None
    if args.resume == "auto":
        resume = run / "latest.pt" if (run / "latest.pt").exists() else None
    elif args.resume:
        resume = Path(args.resume)
        if not resume.exists():
            raise Refused(f"--resume {resume}: no such checkpoint")
    if resume is None:
        stale = [f for f in ("latest.pt", "best.pt", "history.json") if (run / f).exists()]
        if stale:
            raise Refused(f"refusing to start fresh: {run} already holds {stale}. Pass "
                             f"--resume auto to continue that run, or remove the directory.")

    preempt = Preempt().install()
    loader = None
    try:
        data = dict(root=cfg["data_root"], dataset=cfg["dataset"])
        tile_kw = dict(split="train", tile=cfg["tile"], road_bias=cfg["road_bias"],
                       limit=cfg["limit_train"], preload_workers=max(1, cfg["num_workers"]))
        train_ds = RoadTileDataset(**data, **tile_kw, preload=False)          # index only
        steps_per_epoch = len(train_ds) // cfg["batch_size"]
        if cfg["max_train_batches"] is not None:
            steps_per_epoch = min(steps_per_epoch, cfg["max_train_batches"])
        if steps_per_epoch < 1:
            raise Refused(f"no full batch: {len(train_ds)} crops per epoch, "
                          f"batch_size {cfg['batch_size']}")

        if self_test is not None:
            self_test()                        # raises PreflightFailed (exit 3) with a sentinel
        model, loss_fn, optimizer = build()    # seeded HERE, after the throwaway model is gone
        n_params = sum(p.numel() for p in model.parameters())
        scheduler = build_scheduler(optimizer, round(cfg["warmup_epochs"] * steps_per_epoch),
                                    cfg["epochs"] * steps_per_epoch)

        if resume is not None:
            state = load_state(resume, model, optimizer, scheduler, cfg, steps_per_epoch,
                               map_location=device, code=code,
                               allow_code_change=args.allow_code_change)
            repair_best(run, model, cfg, state)
            write_json_atomic(run / "history.json", state["history"])
            print(f"resumed from {resume} at epoch {state['epoch']} step "
                  f"{state['step_in_epoch']}", flush=True)
        else:
            state = fresh_state()
        # Every checkpoint written from here on names this code and this process.
        state["code"] = code
        state["processes"].append({
            "hostname": socket.gethostname(), "pid": os.getpid(), "device_name": device_name,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "code_commit": code["commit"], "code_dirty": code["dirty"],
            "first_epoch": state["epoch"],      # == cfg["epochs"]: the finish phase only
            "last_epoch": state["epoch"]})
        saved = resume is not None and resume.resolve() == (run / "latest.pt").resolve()
        print(f"{n_params / 1e6:.1f}M params, {steps_per_epoch} steps/epoch x {cfg['epochs']} "
              f"epochs, {len(train_ds.rows)} training images", flush=True)

        # Datasets are decoded BEFORE any DataLoader worker exists, so forked workers share
        # the cache copy-on-write. Only the train and validation splits are ever opened.
        training_left = state["epoch"] < cfg["epochs"]
        if training_left and cfg["preload"]:
            train_ds = RoadTileDataset(**data, **tile_kw, preload=True)
        val_ds = RoadFullDataset(**data, split="val", preload=cfg["preload"],
                                 preload_workers=max(1, cfg["num_workers"]),
                                 limit=cfg["limit_val"])

        batches_for = None
        if training_left:
            tokens = EpochTokens(steps_per_epoch, cfg["batch_size"])
            workers = cfg["num_workers"]
            loader = DataLoader(SeededCrops(train_ds, cfg["seed"]), batch_size=cfg["batch_size"],
                                sampler=tokens, drop_last=True, num_workers=workers,
                                pin_memory=is_cuda, persistent_workers=workers > 0,
                                worker_init_fn=_worker_init if workers > 0 else None,
                                generator=torch.Generator().manual_seed(cfg["seed"]))

            def batches_for(epoch: int, start_step: int):
                tokens.set(epoch, start_step)
                return loader

        def validate(m) -> float:
            agg, _ = evaluate(m, val_ds, threshold=0.5, device=device, tile=cfg["tile"],
                              stride=cfg["val_stride"], full_metrics=False, use_amp=use_amp)
            return agg["dataset_iou"]

        complete = fit(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn,
                       batches_for=batches_for, validate=validate, run=run, cfg=cfg, state=state,
                       steps_per_epoch=steps_per_epoch, use_amp=use_amp, device=device,
                       preempt=preempt, ckpt_seconds=60.0 * cfg["ckpt_minutes"], saved=saved)
        if not complete:
            return EXIT_REQUEUE
        finish(model=model, val_ds=val_ds, run=run, cfg=cfg, state=state, device=device,
               use_amp=use_amp, eval_workers=cfg["eval_workers"],
               device_name=device_name,
               extra={"n_params": n_params, "steps_per_epoch": steps_per_epoch,
                      "n_train_images": len(train_ds.rows), "n_val_images": len(val_ds)})
        return EXIT_OK
    finally:
        # Normal return, exit 75, a refusal and an uncaught exception all pass through here.
        try:
            stop_workers(loader)
        finally:
            preempt.restore()


if __name__ == "__main__":
    run_cli(main)          # ends the process with the exit-code contract; never returns
