"""Training loop and checkpointing.

Checkpoints store the complete state_dict, buffers included. The LoRA-oriented
checkpointer in sam2_road_dac saved only parameters with requires_grad, which
drops BatchNorm running statistics — a model reloaded that way evaluates with
uninitialised statistics and scores badly for reasons unrelated to the method.
"""
from __future__ import annotations

import itertools
import math
import os
from pathlib import Path

import torch

from .runtime import tmp_name


def save_checkpoint(path, model, optimizer=None, scheduler=None, *,
                    epoch: int = 0, best: float = 0.0, cfg: dict | None = None,
                    extra: dict | None = None) -> None:
    """Write a checkpoint atomically: temp file, flush + fsync, then os.replace.

    A process killed at any instant leaves either the previous file or the new
    one at `path`, never a truncated one. `extra` is an opaque dict for the
    caller's own resume state (step within the epoch, history, ...).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "epoch": epoch, "best": best,
               "cfg": cfg or {}, "extra": extra or {}, "torch_version": torch.__version__}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    tmp = tmp_name(path)       # .<name>.<host>.<pid>.tmp: never shared with another process
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())   # data on disk BEFORE the rename makes it visible
    os.replace(tmp, path)      # atomic: a preemption mid-save must not corrupt latest.pt


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    """Load into `model` strictly (RuntimeError on any architecture mismatch).

    `map_location` defaults to CPU so a GPU-written checkpoint loads anywhere.
    Returns {epoch, best, cfg, extra}.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return {"epoch": payload.get("epoch", 0), "best": payload.get("best", 0.0),
            "cfg": payload.get("cfg", {}), "extra": payload.get("extra", {})}


def train_one_epoch(model, loader, loss_fn, optimizer, use_amp: bool, device: str,
                    scheduler=None, max_batches: int | None = None, on_step=None) -> float:
    """One pass over `loader` (at most `max_batches`). Returns mean training loss.

    bf16 autocast needs no GradScaler (unlike fp16), so this takes a plain bool;
    on a CPU device it selects CPU bf16 autocast. The loss is masked by
    batch["valid"] and computed in fp32 by the loss itself. `scheduler` (if
    given) is stepped once per optimizer step. A non-finite loss OR a non-finite
    gradient raises FloatingPointError before the optimizer step, so the weights
    are never updated from NaN. Exactly `max_batches` batches are fetched from
    the loader, never one more. `on_step(i, loss)` is called after the i-th
    (0-based) optimizer + scheduler step of this call; returning True stops the
    epoch there.
    """
    model.train()
    total, n = 0.0, 0
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    geo = getattr(loss_fn, "geo", False)    # rungs GA/GB: aux outputs + geometric targets
    for i, batch in enumerate(itertools.islice(loader, max_batches)):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        v = batch["valid"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=bool(use_amp)):
            logits = model(x, aux=True) if geo else model(x)
        if geo:                            # outside autocast: fp32
            loss = loss_fn(logits, y, v, batch["centre"].to(device, non_blocking=True),
                           batch["width"].to(device, non_blocking=True))
        else:
            loss = loss_fn(logits, y, v)       # outside autocast: fp32
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite loss {value} at batch {i}")
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        except RuntimeError as e:
            if "non-finite" not in str(e):
                raise
            raise FloatingPointError(f"non-finite gradient at batch {i}: {e}") from e
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += value; n += 1
        if on_step is not None and on_step(i, value):
            break
    return total / max(n, 1)
