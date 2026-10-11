"""Training loop and checkpointing.

Checkpoints store the complete state_dict, buffers included. The LoRA-oriented
checkpointer in sam2_road_dac saved only parameters with requires_grad, which
drops BatchNorm running statistics — a model reloaded that way evaluates with
uninitialised statistics and scores badly for reasons unrelated to the method.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, *,
                    epoch: int = 0, best: float = 0.0, cfg: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "epoch": epoch, "best": best,
               "cfg": cfg or {}, "torch_version": torch.__version__}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)          # atomic: a preemption mid-save must not corrupt latest.pt


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    """Load into `model` strictly (RuntimeError on any architecture mismatch).

    `map_location` defaults to CPU so a GPU-written checkpoint loads anywhere.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return {"epoch": payload.get("epoch", 0), "best": payload.get("best", 0.0),
            "cfg": payload.get("cfg", {})}


def train_one_epoch(model, loader, loss_fn, optimizer, use_amp: bool, device: str,
                    scheduler=None, max_batches: int | None = None) -> float:
    """One pass over `loader` (at most `max_batches`). Returns mean training loss.

    bf16 autocast needs no GradScaler (unlike fp16), so this takes a plain bool.
    The loss is masked by batch["valid"] and computed in fp32 by the loss itself.
    `scheduler` (if given) is stepped once per optimizer step. A non-finite loss
    raises FloatingPointError rather than training on NaN.
    """
    model.train()
    total, n = 0.0, 0
    amp = bool(use_amp) and str(device).startswith("cuda")
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        v = batch["valid"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        loss = loss_fn(logits, y, v)       # outside autocast: fp32
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite loss {value} at batch {i}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += value; n += 1
    return total / max(n, 1)
