# DWT–ResNeXt Road Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a road-segmentation codebase that measures, under controlled conditions, whether the paper's DWT-at-input ResNeXt actually beats a stock ResNeXt baseline on Massachusetts Roads and DeepGlobe.

**Architecture:** One configurable encoder–decoder whose downsampling and upsampling operators are swappable, so four "rungs" (R0 control, R1 the paper, R2 wavelet-throughout, R3 R2+clDice) share identical code, recipe, and seeds. Stride parity between R0 and R1 is enforced by an assertion, making the R1−R0 difference attributable to the wavelet alone. Wavelets are fixed-weight conv kernels, so they run on GPU inside the network.

**Tech Stack:** Python 3.11, PyTorch 2.10+cu128, torchvision 0.25 (pretrained `resnext50_32x4d`), numpy, scipy, scikit-image, albumentations, networkx, pytest. SLURM array jobs on H200.

**Spec:** `/scratch/hk4488/roadseg/docs/superpowers/specs/2026-09-25-dwt-resnext-road-segmentation-design.md`

## Global Constraints

- Project root: `/scratch/hk4488/roadseg`. All commands run from there.
- Python interpreter: `/scratch/hk4488/.conda/envs/unsloth_sft/bin/python`. There is **no conda on the login node**; do not `conda activate`.
- Extra packages live in project-local `.deps/`, added via `PYTHONPATH`. Never `pip install` into the shared conda env.
- Canonical test command: `bash scripts/pytest.sh <args>` (Task 1 creates it).
- **This session's `/tmp` is 100% full cluster-wide**, so Claude Code cannot capture shell output. Every Bash command must redirect: `{ ...; } > /scratch/hk4488/_probe.txt 2>&1`, then read that file. This is an environment workaround, not a project requirement.
- Wavelet: orthonormal Haar. `lo = [1/√2, 1/√2]`, `hi = [1/√2, −1/√2]`. Sub-band channel order is **fixed** as `[R-LL, R-LH, R-HL, R-HH, G-LL, …, B-HH]`.
- Model outputs **logits**, never probabilities. Sigmoid is applied inside losses and metrics only.
- Encoder total stride is **32** for every rung. Model output resolution always equals input resolution.
- Cardinality is **32 at every stage** — it is the grouped-conv group count, not a per-stage width.
- All metrics are computed on **stitched full-resolution images**, never on tiles.
- Seeds are explicit and recorded. Test set is evaluated once, after all selection is frozen on validation.
- Commit after every task. Attribution line on every commit:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

Five failure modes the spec implies but that no task's primary deliverable naturally exercises. Each has a test assigned to the task that owns the code.

1. **Input dimensions not divisible by 32.** DWT requires even H,W at every level; R2 applies it five times. A 500×500 tile silently mismatches shapes at IDWT. → test in Task 6.
2. **Empty prediction or empty ground truth mask.** Dice, IoU, and clDice all divide by zero when a tile has no road. Background-only tiles are the *majority* at ~4% road prevalence. → tests in Tasks 7, 8, 9.
3. **Fully-nodata region.** After white-border masking a tile can have zero valid pixels, making every metric denominator zero. → test in Task 8.
4. **A split whose masks do not exist.** DeepGlobe's `valid/` has images but no labels; the existing `sam2_road_dac` loader globs the images and then dies opening masks. `prepare.py` must refuse loudly at index time. → test in Task 11.
5. **Logits passed where probabilities are required.** `soft_cl_dice_loss` requires post-sigmoid input; the model emits logits. Its guard only fires when values escape `[0,1]`, which early-training logits often do not. → test in Task 7.

---

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `scripts/setup_deps.sh`, `scripts/pytest.sh`, `src/roadseg/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`, `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: `bash scripts/pytest.sh` as the canonical test runner for every later task.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
def test_environment_has_required_packages():
    import torch, torchvision, numpy, scipy, skimage, albumentations, networkx
    assert torch.__version__.startswith("2.")
    assert hasattr(skimage.morphology, "skeletonize")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
bash scripts/pytest.sh tests/test_smoke.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `scripts/pytest.sh` does not exist yet.

- [ ] **Step 3: Write the dependency installer**

```bash
# scripts/setup_deps.sh
#!/bin/bash
set -euo pipefail
PROJECT=/scratch/hk4488/roadseg
PY=/scratch/hk4488/.conda/envs/unsloth_sft/bin/python
mkdir -p "$PROJECT/.deps"
"$PY" -m pip install --no-cache-dir --target "$PROJECT/.deps" \
    "scikit-image>=0.24" "opencv-python-headless>=4.10" \
    "albumentations>=1.4" "pytest>=8.0"
echo "deps installed to $PROJECT/.deps"
```

- [ ] **Step 4: Write the test runner**

```bash
# scripts/pytest.sh
#!/bin/bash
set -euo pipefail
PROJECT=/scratch/hk4488/roadseg
PY=/scratch/hk4488/.conda/envs/unsloth_sft/bin/python
export PYTHONPATH="$PROJECT/.deps:$PROJECT/src:${PYTHONPATH:-}"
export TORCH_HOME=/scratch/hk4488/.cache/torch
exec "$PY" -m pytest "$@"
```

```ini
# pytest.ini
[pytest]
testpaths = tests
filterwarnings = ignore::DeprecationWarning
```

Create empty `src/roadseg/__init__.py` and `tests/__init__.py`.

- [ ] **Step 5: Install and run**

```bash
{ bash scripts/setup_deps.sh && chmod +x scripts/*.sh && bash scripts/pytest.sh tests/test_smoke.py -v; } > /scratch/hk4488/_probe.txt 2>&1
```
Expected: PASS. Note `setup_deps.sh` may take several minutes; if the login node OOMs, rerun it as a `cpu_short` SLURM job.

- [ ] **Step 6: Commit**

```bash
git add scripts/ src/ tests/ pytest.ini
git commit -m "chore: project scaffolding and vendored deps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Haar wavelet transform as fixed conv kernels

**Files:**
- Create: `src/roadseg/models/__init__.py`, `src/roadseg/models/wavelet.py`, `tests/test_wavelet.py`

**Interfaces:**
- Produces:
  - `haar_filters(dtype) -> Tensor[4,1,2,2]` — the four 2×2 analysis kernels, order LL,LH,HL,HH.
  - `class DWT2d(nn.Module)`: `forward(x: [B,C,H,W]) -> [B,4C,H/2,W/2]`, channel order `[c0-LL,c0-LH,c0-HL,c0-HH,c1-LL,…]`.
  - `class IDWT2d(nn.Module)`: `forward(x: [B,4C,H,W]) -> [B,C,2H,2W]`, exact inverse of `DWT2d`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wavelet.py
import pytest, torch
from roadseg.models.wavelet import DWT2d, IDWT2d, haar_filters


def test_filters_are_orthonormal():
    f = haar_filters(torch.float64).reshape(4, 4)   # each row a flattened 2x2 kernel
    torch.testing.assert_close(f @ f.T, torch.eye(4, dtype=torch.float64))


def test_perfect_reconstruction():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64, dtype=torch.float64)
    y = IDWT2d(3)(DWT2d(3)(x))
    assert y.shape == x.shape
    torch.testing.assert_close(y, x, atol=1e-10, rtol=0)


def test_shapes_and_channel_order():
    x = torch.zeros(1, 3, 8, 8)
    x[:, 1] = 5.0                       # constant in the green channel only
    out = DWT2d(3)(x)
    assert out.shape == (1, 12, 4, 4)
    # green LL is channel 4; orthonormal Haar gives LL = 2 * mean
    torch.testing.assert_close(out[0, 4], torch.full((4, 4), 10.0))
    # green LH/HL/HH are zero for a constant input
    assert out[0, 5:8].abs().max() == 0


def test_wavelet_weights_are_frozen():
    d = DWT2d(3)
    assert all(not p.requires_grad for p in d.parameters())
    assert len(list(d.parameters())) == 0   # registered as a buffer, not a parameter


def test_gradients_flow_through():
    x = torch.randn(1, 3, 16, 16, requires_grad=True)
    DWT2d(3)(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_wavelet.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.models.wavelet`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/models/wavelet.py
"""Orthonormal Haar DWT/IDWT as fixed-weight convolutions.

Implemented as grouped conv2d / conv_transpose2d with frozen 2x2 kernels so the
transform runs on GPU, batched, differentiable, and exactly invertible. A CPU
library such as pywt cannot be used here: R2 applies the transform *inside* the
network at every downsampling stage.

Because the four 2x2 Haar kernels form an orthonormal basis of each 2x2 block,
conv_transpose2d (the adjoint of conv2d) is an exact inverse.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_S = 1.0 / math.sqrt(2.0)
_LO = (_S, _S)
_HI = (_S, -_S)


def haar_filters(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Four 2x2 analysis kernels, shape [4,1,2,2], order LL, LH, HL, HH."""
    lo = torch.tensor(_LO, dtype=dtype)
    hi = torch.tensor(_HI, dtype=dtype)
    # outer(rows, cols): LL=lo*lo, LH=hi*lo, HL=lo*hi, HH=hi*hi
    bank = torch.stack([
        torch.outer(lo, lo),
        torch.outer(hi, lo),
        torch.outer(lo, hi),
        torch.outer(hi, hi),
    ])
    return bank.unsqueeze(1)


class DWT2d(nn.Module):
    """[B,C,H,W] -> [B,4C,H/2,W/2], channels ordered c0-LL,c0-LH,c0-HL,c0-HH,c1-LL,..."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        w = haar_filters().repeat(channels, 1, 1, 1)   # [4C,1,2,2]
        self.register_buffer("weight", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 or x.shape[-2] % 2:
            raise ValueError(f"DWT2d needs even spatial dims, got {tuple(x.shape[-2:])}")
        return F.conv2d(x, self.weight.to(x.dtype), stride=2, groups=self.channels)


class IDWT2d(nn.Module):
    """[B,4C,H,W] -> [B,C,2H,2W]. Exact inverse of DWT2d(channels)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        w = haar_filters().repeat(channels, 1, 1, 1)
        self.register_buffer("weight", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 4 * self.channels:
            raise ValueError(f"IDWT2d({self.channels}) expects {4*self.channels} channels, got {x.shape[1]}")
        return F.conv_transpose2d(x, self.weight.to(x.dtype), stride=2, groups=self.channels)
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_wavelet.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/models/wavelet.py tests/test_wavelet.py src/roadseg/models/__init__.py
git commit -m "feat: orthonormal Haar DWT/IDWT as frozen conv kernels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Swappable downsample and upsample operators

**Files:**
- Create: `src/roadseg/models/ops.py`, `tests/test_ops.py`

**Interfaces:**
- Consumes: `DWT2d`, `IDWT2d` from Task 2.
- Produces:
  - `class WaveletDown(nn.Module)`: `__init__(in_ch, out_ch)`, halves resolution losslessly (DWT then 1×1 conv `4*in_ch -> out_ch`).
  - `class StridedDown(nn.Module)`: `__init__(in_ch, out_ch)`, 3×3 stride-2 conv. Same signature, for A/B swapping.
  - `class BilinearUp(nn.Module)`: `__init__(in_ch, out_ch)`, 2× bilinear then 3×3 conv.
  - `class WaveletUp(nn.Module)`: `__init__(in_ch, out_ch)`, 1×1 conv `in_ch -> 4*out_ch` then `IDWT2d`.
  - `make_down(kind, in_ch, out_ch)` / `make_up(kind, in_ch, out_ch)` with `kind in {"stride","wavelet"}` / `{"bilinear","wavelet"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ops.py
import pytest, torch
from roadseg.models.ops import make_down, make_up


@pytest.mark.parametrize("kind", ["stride", "wavelet"])
def test_down_halves_resolution(kind):
    op = make_down(kind, 16, 32)
    out = op(torch.randn(2, 16, 32, 32))
    assert out.shape == (2, 32, 16, 16)


@pytest.mark.parametrize("kind", ["bilinear", "wavelet"])
def test_up_doubles_resolution(kind):
    op = make_up(kind, 32, 16)
    out = op(torch.randn(2, 32, 16, 16))
    assert out.shape == (2, 16, 32, 32)


def test_wavelet_down_is_information_preserving_before_projection():
    """The 1x1 projection may lose information, but the DWT stage must not."""
    from roadseg.models.wavelet import DWT2d, IDWT2d
    x = torch.randn(1, 8, 16, 16, dtype=torch.float64)
    torch.testing.assert_close(IDWT2d(8)(DWT2d(8)(x)), x, atol=1e-10, rtol=0)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown"):
        make_down("bogus", 4, 4)
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_ops.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.models.ops`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/models/ops.py
"""Swappable resolution-changing operators.

The whole ablation reduces to which of these the encoder and decoder use, so
they share a signature (in_ch, out_ch) and are constructed by name.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import DWT2d, IDWT2d


class StridedDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class WaveletDown(nn.Module):
    """Lossless halving: keep all four sub-bands, then project to out_ch."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.dwt = DWT2d(in_ch)
        self.proj = nn.Conv2d(4 * in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dwt(x))


class BilinearUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class WaveletUp(nn.Module):
    """Predict four sub-bands, then invert the transform."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 4 * out_ch, 1, bias=False)
        self.idwt = IDWT2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.idwt(self.proj(x))


_DOWN = {"stride": StridedDown, "wavelet": WaveletDown}
_UP = {"bilinear": BilinearUp, "wavelet": WaveletUp}


def make_down(kind: str, in_ch: int, out_ch: int) -> nn.Module:
    if kind not in _DOWN:
        raise ValueError(f"unknown down op {kind!r}; expected one of {sorted(_DOWN)}")
    return _DOWN[kind](in_ch, out_ch)


def make_up(kind: str, in_ch: int, out_ch: int) -> nn.Module:
    if kind not in _UP:
        raise ValueError(f"unknown up op {kind!r}; expected one of {sorted(_UP)}")
    return _UP[kind](in_ch, out_ch)
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_ops.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/models/ops.py tests/test_ops.py
git commit -m "feat: swappable down/up sampling operators

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: ResNeXt encoder with rung config and pretrained-weight adaptation

**Files:**
- Create: `src/roadseg/models/encoder.py`, `tests/test_encoder.py`

**Interfaces:**
- Consumes: `DWT2d` (Task 2), `WaveletDown` (Task 3).
- Produces:
  - `adapt_stem_weights(old_w: Tensor[O,3,k,k], hf_init: str) -> Tensor[O,12,k,k]`
  - `class ResNeXtEncoder(nn.Module)`: `__init__(rung: str, pretrained: bool = True, hf_init: str = "zero")`, `forward(x) -> list[Tensor]` returning `[stem, layer1, layer2, layer3, layer4]`.
  - `ResNeXtEncoder.CHANNELS == (64, 256, 512, 1024, 2048)`

This task carries the spec's central guarantee: **R0 and R1 must produce identical feature-map shapes at every stage.**

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_encoder.py
import pytest, torch
from roadseg.models.encoder import ResNeXtEncoder, adapt_stem_weights


@pytest.mark.parametrize("rung", ["R0", "R1", "R2"])
def test_stage_shapes(rung):
    enc = ResNeXtEncoder(rung, pretrained=False)
    feats = enc(torch.randn(1, 3, 128, 128))
    assert [f.shape[-1] for f in feats] == [64, 32, 16, 8, 4]
    assert [f.shape[1] for f in feats] == list(ResNeXtEncoder.CHANNELS)


def test_stride_parity_R0_R1():
    """The spec's core guarantee: R1 differs from R0 only in HOW it downsamples."""
    x = torch.randn(1, 3, 128, 128)
    s0 = [f.shape for f in ResNeXtEncoder("R0", pretrained=False)(x)]
    s1 = [f.shape for f in ResNeXtEncoder("R1", pretrained=False)(x)]
    assert s0 == s1


def test_adapt_stem_weights_zero_init():
    old = torch.randn(64, 3, 7, 7)
    new = adapt_stem_weights(old, hf_init="zero")
    assert new.shape == (64, 12, 7, 7)
    for c in range(3):
        torch.testing.assert_close(new[:, 4 * c], old[:, c] / 2.0)
        assert new[:, 4 * c + 1: 4 * c + 4].abs().max() == 0


def test_R1_stem_matches_R0_on_constant_input():
    """With zero-init high-frequency weights, R1 starts equivalent to R0 on a
    half-resolution image. Checked on the interior to avoid padding effects."""
    torch.manual_seed(0)
    e0 = ResNeXtEncoder("R0", pretrained=False)
    e1 = ResNeXtEncoder("R1", pretrained=False)
    e1.load_adapted_stem_from(e0)
    x = torch.full((1, 3, 64, 64), 0.7)
    with torch.no_grad():
        a = e0.stem_conv_out(x)[..., 4:-4, 4:-4]
        b = e1.stem_conv_out(x)[..., 4:-4, 4:-4]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_high_frequency_weights_receive_gradient():
    """Zero-initialised is not dead: dL/dW = grad_out * input, both nonzero."""
    enc = ResNeXtEncoder("R1", pretrained=False, hf_init="zero")
    enc(torch.randn(1, 3, 64, 64))[-1].sum().backward()
    g = enc.conv1.weight.grad
    hf = torch.cat([g[:, 4 * c + 1: 4 * c + 4] for c in range(3)], dim=1)
    assert hf.abs().sum() > 0


def test_unknown_rung_raises():
    with pytest.raises(ValueError, match="unknown rung"):
        ResNeXtEncoder("R9", pretrained=False)
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_encoder.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.models.encoder`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/models/encoder.py
"""ResNeXt-50 32x4d encoder with a configurable downsampling scheme.

Rungs
-----
R0  stock: conv7x7 s2 -> maxpool s2 -> layers with strided convs
R1  the paper: DWT at input (3->12 ch), conv7x7 **stride 1**, rest as R0
R2  R1 plus wavelet downsampling replacing every remaining stride-2 op

R1's stem convolution is stride 1 because the DWT has already halved the input.
That is what keeps total stride at 32 for every rung and makes R1 - R0
attributable to the wavelet rather than to a resolution change.

Note a documented confound: R1's 7x7 kernel sees a half-resolution LL band, so
its footprint is 14x14 original pixels versus 7x7 for R0. This is intrinsic to
the paper's design, not an artifact of this implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d

from .ops import WaveletDown
from .wavelet import DWT2d

RUNGS = ("R0", "R1", "R2", "R3")


def adapt_stem_weights(old_w: torch.Tensor, hf_init: str = "zero") -> torch.Tensor:
    """Expand a [O,3,k,k] RGB stem kernel to [O,12,k,k] for DWT sub-band input.

    Sub-band order is [c-LL, c-LH, c-HL, c-HH] per colour channel c.

    The LL band of an orthonormal Haar transform equals 2x the local mean, so
    the pretrained kernel is halved to preserve response magnitude. The three
    high-frequency bands start at zero, which makes the adapted network exactly
    equivalent at initialisation to the pretrained one applied to a
    half-resolution image.
    """
    out_c, in_c, kh, kw = old_w.shape
    new_w = old_w.new_zeros(out_c, in_c * 4, kh, kw)
    for c in range(in_c):
        new_w[:, 4 * c] = old_w[:, c] / 2.0
        if hf_init == "small-random":
            new_w[:, 4 * c + 1: 4 * c + 4].normal_(0.0, float(old_w.std()) * 0.01)
        elif hf_init != "zero":
            raise ValueError(f"unknown hf_init {hf_init!r}; expected 'zero' or 'small-random'")
    return new_w


def _wavelet_ize(block: nn.Module) -> None:
    """Swap a bottleneck block's stride-2 downsampling for a wavelet one.

    The grouped 3x3 conv is KEPT, at stride 1, with the DWT placed in front of
    it. Replacing conv2 outright would drop both its 3x3 spatial mixing and its
    32-way grouping, changing far more than the downsampling operator and
    making R2 - R0 unattributable.
    """
    conv = block.conv2                                   # 3x3, groups=32, stride 2
    conv.stride = (1, 1)
    block.conv2 = nn.Sequential(WaveletDown(conv.in_channels, conv.in_channels), conv)
    ds = block.downsample[0]                             # 1x1 stride-2 projection
    block.downsample[0] = WaveletDown(ds.in_channels, ds.out_channels)


class ResNeXtEncoder(nn.Module):
    CHANNELS = (64, 256, 512, 1024, 2048)

    def __init__(self, rung: str, pretrained: bool = True, hf_init: str = "zero") -> None:
        super().__init__()
        if rung not in RUNGS:
            raise ValueError(f"unknown rung {rung!r}; expected one of {RUNGS}")
        self.rung = "R2" if rung == "R3" else rung   # R3 differs only in loss
        self.hf_init = hf_init

        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnext50_32x4d(weights=weights)

        self.dwt = DWT2d(3) if self.rung in ("R1", "R2") else None
        if self.dwt is None:
            self.conv1 = net.conv1
        else:
            self.conv1 = nn.Conv2d(12, 64, 7, stride=1, padding=3, bias=False)
            with torch.no_grad():
                self.conv1.weight.copy_(adapt_stem_weights(net.conv1.weight, hf_init))

        self.bn1, self.relu = net.bn1, net.relu
        self.maxpool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

        if self.rung == "R2":
            self.maxpool = WaveletDown(64, 64)
            for layer in (self.layer2, self.layer3, self.layer4):
                _wavelet_ize(layer[0])

    def load_adapted_stem_from(self, other: "ResNeXtEncoder") -> None:
        """Copy another encoder's 3-channel stem in, adapted to 12 channels."""
        with torch.no_grad():
            self.conv1.weight.copy_(adapt_stem_weights(other.conv1.weight, self.hf_init))
            self.bn1.load_state_dict(other.bn1.state_dict())

    def stem_conv_out(self, x: torch.Tensor) -> torch.Tensor:
        """Stem convolution output, before BN/ReLU. Used by parity tests."""
        return self.conv1(self.dwt(x) if self.dwt is not None else x)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.relu(self.bn1(self.stem_conv_out(x)))
        f0 = x
        f1 = self.layer1(self.maxpool(x))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_encoder.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 9 passed (3 parametrized + 6).

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/models/encoder.py tests/test_encoder.py
git commit -m "feat: ResNeXt encoder with rung config and stem weight adaptation

Stride parity between R0 and R1 is asserted by test, so the ablation's
central comparison cannot silently drift.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: U-Net decoder with explicit skip connections

**Files:**
- Create: `src/roadseg/models/decoder.py`, `tests/test_decoder.py`

**Interfaces:**
- Consumes: `make_up` (Task 3), `ResNeXtEncoder.CHANNELS` (Task 4).
- Produces: `class UNetDecoder(nn.Module)`: `__init__(enc_channels=(64,256,512,1024,2048), widths=(512,256,128,64,32), up="bilinear")`, `forward(feats: list[Tensor]) -> Tensor[B, 32, H, W]` at full input resolution.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_decoder.py
import pytest, torch
from roadseg.models.decoder import UNetDecoder


def _feats(n=1, size=128):
    ch, res = (64, 256, 512, 1024, 2048), (size, size // 2, size // 4, size // 8, size // 16)
    return [torch.randn(n, c, r, r) for c, r in zip(ch, res)]


@pytest.mark.parametrize("up", ["bilinear", "wavelet"])
def test_returns_full_resolution(up):
    out = UNetDecoder(up=up)(_feats(size=128))
    assert out.shape == (1, 32, 256, 256)   # stem is at H/2, so output is 2x stem


def test_all_skips_are_consumed():
    """A decoder that silently ignores a skip would still produce right shapes."""
    dec = UNetDecoder()
    feats = _feats()
    for f in feats:
        f.requires_grad_(True)
    dec(feats).sum().backward()
    assert all(f.grad is not None and f.grad.abs().sum() > 0 for f in feats)
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_decoder.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.models.decoder`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/models/decoder.py
"""U-Net decoder with explicit encoder skip connections.

The draft paper hedges ("skip connections can be used"); this does not. Skips
are the main determinant of thin-road and boundary quality, so they are wired
unconditionally and a test asserts every one of them carries gradient.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ops import make_up


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoder(nn.Module):
    def __init__(
        self,
        enc_channels: tuple[int, ...] = (64, 256, 512, 1024, 2048),
        widths: tuple[int, ...] = (512, 256, 128, 64, 32),
        up: str = "bilinear",
    ) -> None:
        super().__init__()
        f0, f1, f2, f3, f4 = enc_channels
        skips = [f3, f2, f1, f0, 0]          # d5 has no encoder skip
        ins = [f4] + list(widths[:-1])
        self.ups = nn.ModuleList([make_up(up, i, w) for i, w in zip(ins, widths)])
        self.blocks = nn.ModuleList([_ConvBlock(w + s, w) for w, s in zip(widths, skips)])

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        f0, f1, f2, f3, f4 = feats
        x = f4
        for up, block, skip in zip(self.ups, self.blocks, [f3, f2, f1, f0, None]):
            x = up(x)
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_decoder.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/models/decoder.py tests/test_decoder.py
git commit -m "feat: U-Net decoder with explicit skips

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Segmenter assembly and the resolution invariant

**Files:**
- Create: `src/roadseg/models/segmenter.py`, `tests/test_segmenter.py`

**Interfaces:**
- Consumes: `ResNeXtEncoder` (Task 4), `UNetDecoder` (Task 5).
- Produces: `class RoadSegmenter(nn.Module)`: `__init__(rung="R1", pretrained=True, hf_init="zero")`, `forward(x: [B,3,H,W]) -> [B,1,H,W]` **logits**. Also `build_model(cfg: dict) -> nn.Module`.

Owns **Review Focus item 1** (input dims not divisible by 32).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_segmenter.py
import pytest, torch
from roadseg.models.segmenter import RoadSegmenter


@pytest.mark.parametrize("rung", ["R0", "R1", "R2", "R3"])
def test_output_resolution_equals_input(rung):
    """The draft's architecture emits a half-resolution mask. This must not."""
    m = RoadSegmenter(rung, pretrained=False)
    x = torch.randn(1, 3, 128, 128)
    out = m(x)
    assert out.shape == (1, 1, 128, 128)


def test_output_is_logits_not_probabilities():
    m = RoadSegmenter("R0", pretrained=False)
    with torch.no_grad():
        out = m(torch.randn(2, 3, 64, 64) * 5)
    assert out.min() < 0.0, "logits should span negatives; model must not apply sigmoid"


@pytest.mark.parametrize("bad", [(1, 3, 100, 100), (1, 3, 128, 130), (1, 3, 33, 33)])
def test_rejects_dims_not_divisible_by_32(bad):
    """R2 applies the DWT five times; odd dims at any level corrupt shapes.
    Fail loudly at the entry point rather than mismatching deep in the decoder."""
    m = RoadSegmenter("R2", pretrained=False)
    with pytest.raises(ValueError, match="divisible by 32"):
        m(torch.randn(*bad))


def test_R0_and_R1_have_identical_output_shapes_for_many_sizes():
    for s in (64, 128, 256):
        x = torch.randn(1, 3, s, s)
        a = RoadSegmenter("R0", pretrained=False)(x)
        b = RoadSegmenter("R1", pretrained=False)(x)
        assert a.shape == b.shape == (1, 1, s, s)
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_segmenter.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.models.segmenter`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/models/segmenter.py
"""Full segmentation model: rung-configurable encoder + U-Net decoder + head.

Emits logits at exactly the input resolution. The shape assertion in forward()
is deliberate: the draft paper's architecture silently produced a
half-resolution mask, and that class of bug should be impossible here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import UNetDecoder
from .encoder import ResNeXtEncoder

_UP_FOR_RUNG = {"R0": "bilinear", "R1": "bilinear", "R2": "wavelet", "R3": "wavelet"}


class RoadSegmenter(nn.Module):
    def __init__(self, rung: str = "R1", pretrained: bool = True, hf_init: str = "zero") -> None:
        super().__init__()
        self.rung = rung
        self.encoder = ResNeXtEncoder(rung, pretrained=pretrained, hf_init=hf_init)
        self.decoder = UNetDecoder(
            enc_channels=ResNeXtEncoder.CHANNELS, up=_UP_FOR_RUNG[rung]
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(
                f"input spatial dims must be divisible by 32, got {(h, w)}; "
                "pad or crop the tile before the forward pass"
            )
        out = self.head(self.decoder(self.encoder(x)))
        assert out.shape[-2:] == (h, w), f"resolution invariant violated: {out.shape[-2:]} != {(h, w)}"
        return out


def build_model(cfg: dict) -> nn.Module:
    name = cfg.get("model", "roadsegmenter").lower()
    if name == "roadsegmenter":
        return RoadSegmenter(
            rung=cfg.get("rung", "R1"),
            pretrained=cfg.get("pretrained", True),
            hf_init=cfg.get("hf_init", "zero"),
        )
    from .baselines import build_baseline
    return build_baseline(name)
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_segmenter.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 10 passed. (`build_model`'s baselines branch is exercised in Task 16.)

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/models/segmenter.py tests/test_segmenter.py
git commit -m "feat: segmenter assembly with enforced resolution invariant

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Losses

**Files:**
- Create: `src/roadseg/losses.py`, `tests/test_losses.py`
- Copy from: `/scratch/hk4488/sam2_road_dac/src/losses/cl_dice.py` (verified correct; keep its `[0,1]` input guard verbatim)

**Interfaces:**
- Produces:
  - `soft_skeletonize(x, iters=5)`, `soft_cl_dice_loss(pred_prob, target, iters=6)` — copied.
  - `dice_loss(logits, target, eps=1.0) -> Tensor`
  - `class ComboLoss(nn.Module)`: `__init__(use_cldice: bool = False)`, `forward(logits, target) -> Tensor`.

Owns **Review Focus items 2 and 5** (empty masks; logits-vs-probabilities).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_losses.py
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_losses.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.losses`.

- [ ] **Step 3: Implement**

Copy `soft_dilate`, `soft_erode`, `soft_open`, `soft_skeletonize`, `_soft_dice`, and `soft_cl_dice_loss` verbatim from `/scratch/hk4488/sam2_road_dac/src/losses/cl_dice.py` into the top of `src/roadseg/losses.py`, then append:

```python
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice on logits.

    eps=1.0 (not 1e-6) is deliberate: at ~4% road prevalence most crops are
    entirely background, and a tiny epsilon makes the empty-mask case numerically
    unstable rather than merely near-zero.
    """
    prob = torch.sigmoid(logits)
    dims = (-1, -2, -3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class ComboLoss(nn.Module):
    """BCE + Dice, optionally + soft-clDice (rung R3).

    Takes **logits**. The clDice term needs probabilities, so the sigmoid is
    applied here — callers must never pre-activate.
    """

    def __init__(self, use_cldice: bool = False, cldice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.use_cldice = use_cldice
        self.cldice_weight = cldice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bce(logits, target) + dice_loss(logits, target)
        if self.use_cldice:
            loss = loss + self.cldice_weight * soft_cl_dice_loss(torch.sigmoid(logits), target)
        return loss
```

Add `import torch.nn as nn` to the imports.

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_losses.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/losses.py tests/test_losses.py
git commit -m "feat: BCE+Dice+clDice combo loss

clDice implementation copied from sam2_road_dac, which is correct. Dice uses
eps=1.0 for stability on the background-only crops that dominate at 4% road
prevalence.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Pixel metrics with nodata masking

**Files:**
- Create: `src/roadseg/metrics.py`, `tests/test_metrics.py`
- Copy from: `/scratch/hk4488/sam2_road_dac/src/metrics/cl_dice.py` (`cl_dice_metric`, verified correct)

**Interfaces:**
- Produces:
  - `confusion(pred: ndarray[bool], target: ndarray[bool], valid: ndarray[bool] | None) -> dict` with keys `tp, fp, fn, tn`.
  - `metrics_from_confusion(c: dict) -> dict` with keys `iou, f1, precision, recall`.
  - `relaxed_f1(pred, target, rho=3, valid=None) -> dict` with keys `relaxed_precision, relaxed_recall, relaxed_f1`.
  - `cl_dice_metric(pred, target)` — copied.

Owns **Review Focus items 2 and 3** (empty masks; fully-nodata regions).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import numpy as np, pytest
from roadseg.metrics import cl_dice_metric, confusion, metrics_from_confusion, relaxed_f1


def test_confusion_hand_computed():
    p = np.array([[1, 1], [0, 0]], bool)
    t = np.array([[1, 0], [1, 0]], bool)
    assert confusion(p, t, None) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_iou_and_f1_hand_computed():
    m = metrics_from_confusion({"tp": 1, "fp": 1, "fn": 1, "tn": 1})
    assert m["iou"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(0.5)


def test_both_empty_scores_one_not_nan():
    """Convention: a correct all-background prediction is perfect, not undefined."""
    m = metrics_from_confusion({"tp": 0, "fp": 0, "fn": 0, "tn": 100})
    assert m["iou"] == 1.0 and m["f1"] == 1.0
    assert not np.isnan(m["precision"]) and not np.isnan(m["recall"])


def test_valid_mask_excludes_pixels_from_every_count():
    p = np.array([[1, 1], [0, 0]], bool)
    t = np.zeros((2, 2), bool)
    valid = np.array([[False, True], [True, True]])
    c = confusion(p, t, valid)
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == 3, "masked pixel must not be counted"
    assert c["fp"] == 1


def test_fully_nodata_region_is_not_nan():
    """Nodata borders can make a whole tile invalid; denominators go to zero."""
    p = np.ones((4, 4), bool); t = np.zeros((4, 4), bool)
    m = metrics_from_confusion(confusion(p, t, np.zeros((4, 4), bool)))
    assert all(np.isfinite(v) for v in m.values())


def test_relaxed_f1_credits_a_small_offset():
    t = np.zeros((32, 32), bool); t[16, 4:28] = True
    p = np.zeros((32, 32), bool); p[18, 4:28] = True      # 2 px off, within rho=3
    assert relaxed_f1(p, t, rho=3)["relaxed_f1"] > 0.9
    assert metrics_from_confusion(confusion(p, t, None))["iou"] == 0.0


def test_relaxed_f1_rejects_a_large_offset():
    t = np.zeros((32, 32), bool); t[4, 4:28] = True
    p = np.zeros((32, 32), bool); p[24, 4:28] = True      # 20 px off
    assert relaxed_f1(p, t, rho=3)["relaxed_f1"] < 0.1


def test_cl_dice_on_identical_masks_is_one():
    m = np.zeros((32, 32), bool); m[16, 4:28] = True
    assert cl_dice_metric(m, m) == pytest.approx(1.0, abs=1e-3)
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_metrics.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.metrics`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/metrics.py
"""Pixel and buffered metrics, all nodata-aware.

Every function takes an optional boolean `valid` mask. Massachusetts imagery
carries white no-data borders; counting them inflates true negatives and
therefore every aggregate score. Masking is not optional for honest numbers.

Convention for degenerate cases: when a metric's denominator is zero because
both prediction and target are empty, the score is 1.0 (a correct
all-background prediction is perfect). When the target is empty but the
prediction is not, the score is 0.0.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize


def confusion(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> dict:
    pred = pred.astype(bool); target = target.astype(bool)
    if valid is not None:
        v = valid.astype(bool)
        pred, target = pred & v, target & v
        tn = int((~pred & ~target & v).sum())
    else:
        tn = int((~pred & ~target).sum())
    return {
        "tp": int((pred & target).sum()),
        "fp": int((pred & ~target).sum()),
        "fn": int((~pred & target).sum()),
        "tn": tn,
    }


def _ratio(num: float, den: float) -> float:
    return 1.0 if den == 0 else num / den


def metrics_from_confusion(c: dict) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {
        "iou": _ratio(tp, tp + fp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
    }


def relaxed_f1(pred: np.ndarray, target: np.ndarray, rho: int = 3,
               valid: np.ndarray | None = None) -> dict:
    """Buffered F1 (DeepGlobe convention): a prediction within rho px counts.

    Precision is measured against a dilated target, recall against a dilated
    prediction. Roads are thin, so exact pixel agreement understates quality.
    """
    pred = pred.astype(bool); target = target.astype(bool)
    if valid is not None:
        v = valid.astype(bool); pred, target = pred & v, target & v
    k = np.ones((2 * rho + 1, 2 * rho + 1), bool)
    p = _ratio(float((pred & binary_dilation(target, k)).sum()), float(pred.sum()))
    r = _ratio(float((target & binary_dilation(pred, k)).sum()), float(target.sum()))
    return {"relaxed_precision": p, "relaxed_recall": r,
            "relaxed_f1": 0.0 if p + r == 0 else 2 * p * r / (p + r)}
```

Append `cl_dice_metric` copied verbatim from `sam2_road_dac/src/metrics/cl_dice.py`.

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_metrics.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/metrics.py tests/test_metrics.py
git commit -m "feat: nodata-aware pixel and relaxed-F1 metrics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: APLS, with the snap-distance and complexity defects fixed

**Files:**
- Create: `src/roadseg/apls.py`, `tests/test_apls.py`
- Port from: `/scratch/hk4488/sam2_road_dac/src/metrics/apls.py` (structure is sound; two defects must be fixed)

**Interfaces:**
- Produces:
  - `graph_from_mask(mask: ndarray[bool]) -> nx.Graph` with node attrs `y`, `x` and edge attr `length`.
  - `apls(G_pred, G_true, sample=500, seed=0, max_snap_dist=8.0) -> float`

The two defects, restated so the implementer does not reintroduce them:
1. `_nearest_node` scanned every node per query — O(N²) overall. Skeletonised 1500×1500 masks have thousands of nodes. Use `scipy.spatial.cKDTree`.
2. It snapped to the nearest node at **any** distance, so a prediction whose nearest node was far away still scored well. Real APLS scores an unmatched node **0**. This inflated the metric systematically.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_apls.py
import networkx as nx, numpy as np, pytest
from roadseg.apls import apls, graph_from_mask


def _line_graph(n=10, dx=0.0):
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, y=0.0, x=float(i * 10) + dx)
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G


def test_identical_graphs_score_one():
    G = _line_graph()
    assert apls(G, G, sample=50) == pytest.approx(1.0, abs=1e-6)


def test_small_offset_still_scores_high():
    assert apls(_line_graph(dx=2.0), _line_graph(), sample=50, max_snap_dist=8.0) > 0.9


def test_far_graph_scores_zero_not_high():
    """THE defect in the ported implementation: without max_snap_dist a distant
    prediction still snapped to a nearest node and earned near-full credit."""
    far = _line_graph(dx=10_000.0)
    assert apls(far, _line_graph(), sample=50, max_snap_dist=8.0) < 0.05


def test_empty_prediction_scores_zero():
    assert apls(nx.Graph(), _line_graph(), sample=50) == 0.0


def test_graph_from_mask_recovers_a_line():
    m = np.zeros((64, 64), bool); m[32, 8:56] = True
    G = graph_from_mask(m)
    assert G.number_of_nodes() >= 2
    assert nx.number_connected_components(G) == 1


def test_scales_to_many_nodes():
    """Guards against the O(N^2) nearest-node scan returning."""
    import time
    m = np.zeros((512, 512), bool)
    m[::16, :] = True; m[:, ::16] = True
    G = graph_from_mask(m)
    t0 = time.time(); apls(G, G, sample=100); dt = time.time() - t0
    assert dt < 30.0, f"APLS too slow ({dt:.1f}s) — is cKDTree being used?"
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_apls.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.apls`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/apls.py
"""APLS (Average Path Length Similarity) for rasterised road predictions.

Ported from sam2_road_dac with two corrections:

1. Nearest-node lookup uses a cKDTree. The original scanned every node per
   query, making the metric O(N^2) on graphs with thousands of nodes.
2. A `max_snap_dist` threshold is enforced. The original snapped to the nearest
   node at any distance, so a prediction whose nearest node was far away still
   received credit; the standard metric scores an unmatched node 0. Without
   this the reported APLS is systematically inflated.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def graph_from_mask(mask: np.ndarray, spacing: int = 8) -> nx.Graph:
    """Skeletonise a binary mask into a graph, sampling the skeleton every
    `spacing` pixels and linking samples that are 8-connected neighbours."""
    sk = skeletonize(mask.astype(bool))
    ys, xs = np.nonzero(sk)
    if len(ys) == 0:
        return nx.Graph()
    keep = [i for i, (y, x) in enumerate(zip(ys, xs)) if (y % spacing == 0 or x % spacing == 0)]
    pts = np.stack([ys[keep], xs[keep]], axis=1)
    G = nx.Graph()
    for i, (y, x) in enumerate(pts):
        G.add_node(i, y=float(y), x=float(x))
    tree = cKDTree(pts)
    for i, j in tree.query_pairs(r=spacing * 1.5):
        d = float(np.hypot(*(pts[i] - pts[j])))
        G.add_edge(int(i), int(j), length=d)
    return G


def _pos_array(G: nx.Graph) -> tuple[np.ndarray, list]:
    nodes = list(G.nodes())
    pts = np.array([[G.nodes[n]["y"], G.nodes[n]["x"]] for n in nodes], dtype=float)
    return pts, nodes


def _ensure_edge_lengths(G: nx.Graph) -> None:
    for u, v, data in G.edges(data=True):
        if "length" not in data:
            a, b = G.nodes[u], G.nodes[v]
            data["length"] = math.hypot(a["y"] - b["y"], a["x"] - b["x"])


def _shortest_len(G: nx.Graph, s, t) -> float:
    try:
        return float(nx.shortest_path_length(G, s, t, weight="length"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def _one_way(G_src: nx.Graph, G_dst: nx.Graph, sample: int, rng, max_snap_dist: float) -> float:
    src_nodes = list(G_src.nodes())
    if len(src_nodes) < 2 or G_dst.number_of_nodes() == 0:
        return 0.0
    dst_pts, dst_nodes = _pos_array(G_dst)
    tree = cKDTree(dst_pts)
    diffs = []
    for _ in range(sample):
        i, j = rng.choice(len(src_nodes), size=2, replace=False)
        s, t = src_nodes[int(i)], src_nodes[int(j)]
        L1 = _shortest_len(G_src, s, t)
        if not math.isfinite(L1) or L1 <= 0:
            continue
        sq = [[G_src.nodes[s]["y"], G_src.nodes[s]["x"]],
              [G_src.nodes[t]["y"], G_src.nodes[t]["x"]]]
        dists, idxs = tree.query(np.array(sq), k=1)
        if dists[0] > max_snap_dist or dists[1] > max_snap_dist:
            diffs.append(1.0)          # unmatched -> zero credit
            continue
        L2 = _shortest_len(G_dst, dst_nodes[int(idxs[0])], dst_nodes[int(idxs[1])])
        diffs.append(1.0 if not math.isfinite(L2) else min(1.0, abs(L1 - L2) / max(L1, 1e-6)))
    return 0.0 if not diffs else 1.0 - float(np.mean(diffs))


def apls(G_pred: nx.Graph, G_true: nx.Graph, sample: int = 500, seed: int = 0,
         max_snap_dist: float = 8.0) -> float:
    """Symmetric APLS in [0, 1], higher is better."""
    if G_pred.number_of_nodes() == 0 or G_true.number_of_nodes() == 0:
        return 0.0
    _ensure_edge_lengths(G_pred); _ensure_edge_lengths(G_true)
    rng = np.random.default_rng(seed)
    a = _one_way(G_true, G_pred, sample, rng, max_snap_dist)
    b = _one_way(G_pred, G_true, sample, rng, max_snap_dist)
    return 0.5 * (a + b)
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_apls.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/apls.py tests/test_apls.py
git commit -m "fix: APLS snap-distance threshold and cKDTree lookup

Ported from sam2_road_dac. The original snapped to the nearest node at any
distance, giving credit to predictions with no nearby node and inflating the
metric; and used an O(N^2) scan. Both corrected, both pinned by tests.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Tiling and stitching

**Files:**
- Create: `src/roadseg/data/__init__.py`, `src/roadseg/data/tiling.py`, `tests/test_tiling.py`

**Interfaces:**
- Produces:
  - `tile_positions(h, w, tile, stride) -> list[tuple[int,int]]` — top-left corners covering the image, last row/col flush to the edge.
  - `tile_image(img: ndarray[H,W,...], tile, stride) -> tuple[list[ndarray], list[tuple[int,int]]]`
  - `stitch(tiles, positions, h, w, tile) -> ndarray[h,w]` — overlap-averaged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tiling.py
import numpy as np, pytest
from roadseg.data.tiling import stitch, tile_image, tile_positions


@pytest.mark.parametrize("h,w", [(1500, 1500), (1024, 1024), (600, 513)])
def test_positions_cover_every_pixel(h, w):
    cov = np.zeros((h, w), bool)
    for y, x in tile_positions(h, w, 512, 256):
        cov[y:y + 512, x:x + 512] = True
    assert cov.all(), "tiling must cover the whole image including ragged edges"


@pytest.mark.parametrize("h,w", [(1500, 1500), (600, 513)])
def test_stitch_inverts_tile(h, w):
    rng = np.random.default_rng(0)
    img = rng.random((h, w)).astype(np.float32)
    tiles, pos = tile_image(img, 512, 256)
    np.testing.assert_allclose(stitch(tiles, pos, h, w, 512), img, atol=1e-5)


def test_tiles_are_all_the_requested_size():
    tiles, _ = tile_image(np.zeros((600, 513), np.float32), 512, 256)
    assert all(t.shape == (512, 512) for t in tiles)


def test_overlaps_are_averaged_not_overwritten():
    h = w = 768
    tiles, pos = tile_image(np.ones((h, w), np.float32), 512, 256)
    tiles = [t * (i + 1) for i, t in enumerate(tiles)]
    out = stitch(tiles, pos, h, w, 512)
    assert out.min() > 0 and np.isfinite(out).all()
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_tiling.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.data.tiling`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/data/tiling.py
"""Deterministic sliding-window tiling and overlap-averaged stitching.

Validation and test images are scored at full resolution, so predictions are
made on tiles and reassembled. The last row and column are flush against the
image edge rather than padded, which keeps every tile exactly `tile` pixels and
avoids feeding the network padded borders it never saw in training.
"""
from __future__ import annotations

import numpy as np


def tile_positions(h: int, w: int, tile: int, stride: int) -> list[tuple[int, int]]:
    def starts(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        s = list(range(0, extent - tile + 1, stride))
        if s[-1] != extent - tile:
            s.append(extent - tile)
        return s
    return [(y, x) for y in starts(h) for x in starts(w)]


def _pad_to(img: np.ndarray, tile: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h >= tile and w >= tile:
        return img
    pad = [(0, max(0, tile - h)), (0, max(0, tile - w))] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad, mode="reflect")


def tile_image(img: np.ndarray, tile: int, stride: int):
    img = _pad_to(img, tile)
    h, w = img.shape[:2]
    pos = tile_positions(h, w, tile, stride)
    return [img[y:y + tile, x:x + tile] for y, x in pos], pos


def stitch(tiles, positions, h: int, w: int, tile: int) -> np.ndarray:
    ph, pw = max(h, tile), max(w, tile)
    acc = np.zeros((ph, pw), np.float64)
    cnt = np.zeros((ph, pw), np.float64)
    for t, (y, x) in zip(tiles, positions):
        acc[y:y + tile, x:x + tile] += np.asarray(t, dtype=np.float64)
        cnt[y:y + tile, x:x + tile] += 1.0
    return (acc / np.maximum(cnt, 1.0))[:h, :w].astype(np.float32)
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_tiling.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/data/tiling.py src/roadseg/data/__init__.py tests/test_tiling.py
git commit -m "feat: sliding-window tiling with overlap-averaged stitching

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Dataset acquisition, splits, and index

**Files:**
- Create: `src/roadseg/data/prepare.py`, `scripts/download_data.sh`, `tests/test_prepare.py`

**Interfaces:**
- Produces:
  - `build_index(root: Path, dataset: str, split: str) -> list[dict]` with keys `image`, `mask`, `id`.
  - `deepglobe_splits(root: Path, seed: int = 1337) -> dict[str, list[str]]` — 70/15/15 by image id.
  - `MissingMaskError(Exception)`

Owns **Review Focus item 4** (a split whose masks do not exist).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prepare.py
import pytest
from roadseg.data.prepare import MissingMaskError, build_index, deepglobe_splits


def _make_deepglobe(tmp_path, n=20, masks=True):
    d = tmp_path / "train"; d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i}_sat.jpg").write_bytes(b"x")
        if masks:
            (d / f"{i}_mask.png").write_bytes(b"x")
    return tmp_path


def test_deepglobe_split_is_disjoint_and_covers_everything(tmp_path):
    root = _make_deepglobe(tmp_path, n=100)
    s = deepglobe_splits(root)
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 100
    assert not (set(s["train"]) & set(s["val"]))
    assert not (set(s["train"]) & set(s["test"]))
    assert not (set(s["val"]) & set(s["test"]))


def test_deepglobe_split_is_deterministic(tmp_path):
    root = _make_deepglobe(tmp_path, n=100)
    assert deepglobe_splits(root, seed=1337) == deepglobe_splits(root, seed=1337)
    assert deepglobe_splits(root, seed=1337) != deepglobe_splits(root, seed=7)


def test_missing_masks_raise_at_index_time(tmp_path):
    """DeepGlobe's released valid/ and test/ have images but no labels. The
    sam2_road_dac loader globbed the images and died later opening masks; this
    must fail immediately and say why."""
    root = _make_deepglobe(tmp_path, n=5, masks=False)
    with pytest.raises(MissingMaskError, match="no ground-truth mask"):
        build_index(root, "deepglobe", "train")


def test_index_pairs_images_with_masks(tmp_path):
    root = _make_deepglobe(tmp_path, n=3)
    idx = build_index(root, "deepglobe", "train")
    assert len(idx) == 3
    assert all(r["mask"].exists() and r["image"].exists() for r in idx)
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_prepare.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.data.prepare`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/data/prepare.py
"""Dataset discovery, splitting, and index construction.

DeepGlobe note: only the 6226 images under train/ were ever released with
masks. The challenge's valid/ and test/ directories contain imagery only. Any
index built over them is silently broken, so we refuse at index time and
re-split the labelled 6226 ourselves with a recorded seed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEEPGLOBE_SPLIT_SEED = 1337
DEEPGLOBE_FRACTIONS = (0.70, 0.15, 0.15)


class MissingMaskError(Exception):
    pass


def _deepglobe_ids(root: Path) -> list[str]:
    return sorted(p.name[: -len("_sat.jpg")] for p in (root / "train").glob("*_sat.jpg"))


def deepglobe_splits(root: Path, seed: int = DEEPGLOBE_SPLIT_SEED) -> dict[str, list[str]]:
    ids = _deepglobe_ids(Path(root))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_tr = int(len(ids) * DEEPGLOBE_FRACTIONS[0])
    n_va = int(len(ids) * DEEPGLOBE_FRACTIONS[1])
    picks = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    return {k: sorted(ids[i] for i in v) for k, v in picks.items()}


def build_index(root: Path, dataset: str, split: str) -> list[dict]:
    root = Path(root)
    if dataset == "deepglobe":
        ids = deepglobe_splits(root)[split] if split in ("train", "val", "test") else []
        rows = [{"id": i,
                 "image": root / "train" / f"{i}_sat.jpg",
                 "mask": root / "train" / f"{i}_mask.png"} for i in ids]
    elif dataset == "massachusetts":
        sub = {"train": "train", "val": "valid", "test": "test"}[split]
        rows = [{"id": p.stem, "image": p, "mask": root / f"{sub}_labels" / f"{p.stem}.tif"}
                for p in sorted((root / sub).glob("*.tif*"))]
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    if not rows:
        raise MissingMaskError(f"{dataset}/{split}: no images found under {root}")
    missing = [r["id"] for r in rows if not r["mask"].exists()]
    if missing:
        raise MissingMaskError(
            f"{dataset}/{split}: no ground-truth mask for {len(missing)} of {len(rows)} "
            f"images (e.g. {missing[:3]}). For DeepGlobe, only train/ was released "
            f"with labels — use deepglobe_splits() over train/ instead."
        )
    return rows
```

```bash
# scripts/download_data.sh
#!/bin/bash
set -euo pipefail
DATA=${DATA:-/scratch/hk4488/roadseg/data}
PY=/scratch/hk4488/.conda/envs/unsloth_sft/bin/python
mkdir -p "$DATA"

# Both datasets come from Kaggle, so both need ~/.kaggle/kaggle.json.
if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "ERROR: need a Kaggle API token at ~/.kaggle/kaggle.json" >&2
  echo "  kaggle.com -> Settings -> API -> Create New Token, then chmod 600" >&2
  exit 1
fi
"$PY" -m pip install --quiet --target /scratch/hk4488/roadseg/.deps kaggle

fetch () {  # $1 = kaggle slug, $2 = destination dir name
  local slug=$1 dest=$2
  [[ -d "$DATA/$dest" ]] && { echo "$dest already present, skipping"; return; }
  "$PY" -m kaggle datasets download -d "$slug" -p "$DATA/$dest" --unzip
}

fetch balraj98/deepglobe-road-extraction-dataset deepglobe
fetch balraj98/massachusetts-roads-dataset       massachusetts

echo "data ready under $DATA"
echo "NOTE: verify the extracted directory layout matches build_index() in"
echo "      src/roadseg/data/prepare.py before launching the array job."
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_prepare.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 4 passed. (Tests use synthetic trees; no real download needed.)

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/data/prepare.py scripts/download_data.sh tests/test_prepare.py
git commit -m "feat: dataset index with loud failure on missing masks

DeepGlobe released labels only for train/; indexing valid/ or test/ now raises
instead of failing later at image-open time.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Nodata detection and torch Datasets

**Files:**
- Create: `src/roadseg/data/nodata.py`, `src/roadseg/data/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `build_index` (Task 11), `tile_positions` (Task 10).
- Produces:
  - `valid_mask(img: ndarray[H,W,3]) -> ndarray[H,W] bool` — False on near-white nodata.
  - `class RoadTileDataset(Dataset)` (train): random crops, road-biased sampling. Yields `{"image": [3,T,T] float32, "mask": [1,T,T] float32, "valid": [1,T,T] float32}`.
  - `class RoadFullDataset(Dataset)` (val/test): yields whole images plus their index row.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset.py
import numpy as np, torch
from PIL import Image
from roadseg.data.dataset import RoadTileDataset
from roadseg.data.nodata import valid_mask


def _fake_deepglobe(tmp_path, n=4, size=256):
    d = tmp_path / "train"; d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        m = np.zeros((size, size), np.uint8); m[size // 2, :] = 255
        Image.fromarray(img).save(d / f"{i}_sat.jpg")
        Image.fromarray(m).save(d / f"{i}_mask.png")
    return tmp_path


def test_valid_mask_flags_white_border():
    img = np.full((32, 32, 3), 120, np.uint8)
    img[:, :8] = 255
    v = valid_mask(img)
    assert not v[:, :8].any() and v[:, 8:].all()


def test_item_shapes_and_dtypes(tmp_path):
    ds = RoadTileDataset(_fake_deepglobe(tmp_path), "deepglobe", "train", tile=64)
    it = ds[0]
    assert it["image"].shape == (3, 64, 64) and it["image"].dtype == torch.float32
    assert it["mask"].shape == (1, 64, 64)
    assert set(torch.unique(it["mask"]).tolist()) <= {0.0, 1.0}


def test_road_biased_sampling_beats_uniform(tmp_path):
    """At ~4% prevalence, uniform crops are almost all empty. Biased sampling
    must actually raise the hit rate or training wastes most of its batches."""
    root = _fake_deepglobe(tmp_path, n=4, size=256)
    biased = RoadTileDataset(root, "deepglobe", "train", tile=64, road_bias=1.0)
    hits = sum(biased[i]["mask"].sum().item() > 0 for i in range(40))
    assert hits >= 36, f"only {hits}/40 biased crops contained road"


def test_crops_differ_across_epochs(tmp_path):
    """Guards the np.random-in-workers bug that makes every worker emit the
    same crop. Uses a torch generator instead."""
    ds = RoadTileDataset(_fake_deepglobe(tmp_path), "deepglobe", "train", tile=64, road_bias=0.0)
    a = [ds[0]["image"].clone() for _ in range(8)]
    assert any(not torch.equal(a[0], x) for x in a[1:])


def test_mask_threshold_handles_0_255_encoding(tmp_path):
    ds = RoadTileDataset(_fake_deepglobe(tmp_path), "deepglobe", "train", tile=64, road_bias=1.0)
    assert ds[0]["mask"].max().item() == 1.0, "0/255 masks must map to 0/1, not 0/255"
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_dataset.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.data.nodata`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/data/nodata.py
"""No-data detection for aerial imagery.

Massachusetts tiles carry pure-white borders where no imagery exists. Counted
as background they inflate true negatives and therefore every aggregate score,
so they are excluded from both sampling and metric denominators.
"""
from __future__ import annotations

import numpy as np

WHITE_THRESHOLD = 250


def valid_mask(img: np.ndarray, threshold: int = WHITE_THRESHOLD) -> np.ndarray:
    """False where the pixel is near-white in all three channels."""
    return ~np.all(img >= threshold, axis=-1)
```

```python
# src/roadseg/data/dataset.py
"""Datasets for road segmentation.

Train uses on-the-fly random crops from full images — no duplicated
preprocessed tiles on scratch, and the tiling scheme can change without
re-preprocessing. Crop RNG comes from a torch Generator seeded per __getitem__
call rather than np.random, because np.random is duplicated across DataLoader
workers and would make every worker emit identical crops.

Aerial imagery is never resized: downsampling 1500x1500 to 512 would reduce
2-4 px roads to sub-pixel width, deleting exactly what the model must find.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .nodata import valid_mask
from .prepare import build_index

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _normalize(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def _augment(img, mask, valid, g):
    if torch.rand(1, generator=g).item() < 0.5:
        img, mask, valid = img[:, ::-1], mask[:, ::-1], valid[:, ::-1]
    if torch.rand(1, generator=g).item() < 0.5:
        img, mask, valid = img[::-1], mask[::-1], valid[::-1]
    k = int(torch.randint(0, 4, (1,), generator=g).item())
    if k:
        img = np.rot90(img, k, (0, 1)); mask = np.rot90(mask, k); valid = np.rot90(valid, k)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask), np.ascontiguousarray(valid)


class RoadTileDataset(Dataset):
    """Random crops for training. `road_bias` is the probability that a crop is
    resampled (up to `tries`) until it contains road pixels."""

    def __init__(self, root, dataset: str, split: str, tile: int = 512,
                 road_bias: float = 0.75, tries: int = 12, length: int | None = None,
                 augment: bool = True) -> None:
        self.rows = build_index(Path(root), dataset, split)
        self.tile, self.road_bias, self.tries = tile, road_bias, tries
        self.augment = augment
        self.length = length or len(self.rows) * 8

    def __len__(self) -> int:
        return self.length

    def _load(self, row) -> tuple[np.ndarray, np.ndarray]:
        img = np.array(Image.open(row["image"]).convert("RGB"))
        mask = np.array(Image.open(row["mask"]).convert("L"))
        return img, (mask > 127).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        row = self.rows[int(torch.randint(0, len(self.rows), (1,), generator=g).item())]
        img, mask = self._load(row)
        v = valid_mask(img)
        h, w = mask.shape
        t = self.tile
        want_road = torch.rand(1, generator=g).item() < self.road_bias
        for attempt in range(self.tries):
            y = int(torch.randint(0, max(1, h - t + 1), (1,), generator=g).item())
            x = int(torch.randint(0, max(1, w - t + 1), (1,), generator=g).item())
            m = mask[y:y + t, x:x + t]
            if not want_road or m.sum() > 0 or attempt == self.tries - 1:
                break
        i, m, vv = img[y:y + t, x:x + t], m, v[y:y + t, x:x + t]
        if self.augment:
            i, m, vv = _augment(i, m, vv, g)
        return {
            "image": torch.from_numpy(_normalize(i)).permute(2, 0, 1),
            "mask": torch.from_numpy(m).unsqueeze(0).float(),
            "valid": torch.from_numpy(vv.astype(np.float32)).unsqueeze(0),
        }


class RoadFullDataset(Dataset):
    """Whole images for validation and test; tiling happens at inference time."""

    def __init__(self, root, dataset: str, split: str) -> None:
        self.rows = build_index(Path(root), dataset, split)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = np.array(Image.open(row["image"]).convert("RGB"))
        mask = (np.array(Image.open(row["mask"]).convert("L")) > 127)
        return {"image": _normalize(img), "mask": mask,
                "valid": valid_mask(img), "id": row["id"]}
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_dataset.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/data/nodata.py src/roadseg/data/dataset.py tests/test_dataset.py
git commit -m "feat: road-biased tile dataset with nodata masking

Crop RNG uses a torch Generator, not np.random, which duplicates across
DataLoader workers.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Training engine and checkpointing

**Files:**
- Create: `src/roadseg/engine.py`, `tests/test_engine.py`

**Interfaces:**
- Produces:
  - `save_checkpoint(path, model, optimizer, scheduler, epoch, best, cfg)` — **full `state_dict()` including buffers.**
  - `load_checkpoint(path, model, optimizer=None, scheduler=None) -> dict`
  - `train_one_epoch(model, loader, loss_fn, optimizer, scaler, device) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine.py
import pytest, torch, torch.nn as nn
from roadseg.engine import load_checkpoint, save_checkpoint, train_one_epoch
from roadseg.losses import ComboLoss


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8),
                                 nn.ReLU(), nn.Conv2d(8, 1, 1))

    def forward(self, x):
        return self.net(x)


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


def test_overfit_one_batch():
    """Integration check: the whole train step actually reduces loss."""
    torch.manual_seed(0)
    m = _Tiny()
    x = torch.randn(2, 3, 32, 32)
    y = (torch.rand(2, 1, 32, 32) > 0.5).float()
    batch = [{"image": x, "mask": y, "valid": torch.ones_like(y)}]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    first = train_one_epoch(m, batch, ComboLoss(), opt, use_amp=False, device="cpu")
    for _ in range(60):
        last = train_one_epoch(m, batch, ComboLoss(), opt, use_amp=False, device="cpu")
    assert last < first * 0.5, f"loss did not fall: {first:.4f} -> {last:.4f}"
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_engine.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.engine`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/engine.py
"""Training loop and checkpointing.

Checkpoints store the complete state_dict, buffers included. The LoRA-oriented
checkpointer in sam2_road_dac saved only parameters with requires_grad, which
drops BatchNorm running statistics — a model reloaded that way evaluates with
uninitialised statistics and scores badly for reasons unrelated to the method.
"""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, *,
                    epoch: int = 0, best: float = 0.0, cfg: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "epoch": epoch, "best": best, "cfg": cfg or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)          # atomic: a preemption mid-save must not corrupt latest.pt


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return {"epoch": payload.get("epoch", 0), "best": payload.get("best", 0.0),
            "cfg": payload.get("cfg", {})}


def train_one_epoch(model, loader, loss_fn, optimizer, use_amp: bool, device: str) -> float:
    """One pass over `loader`. Returns mean training loss.

    bf16 autocast needs no GradScaler (unlike fp16), so this takes a plain bool
    rather than a scaler object.
    """
    model.train()
    total, n = 0.0, 0
    amp = bool(use_amp) and str(device).startswith("cuda")
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach()); n += 1
    return total / max(n, 1)
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_engine.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/engine.py tests/test_engine.py
git commit -m "feat: training engine with buffer-preserving atomic checkpoints

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Evaluation, threshold selection, and paired bootstrap

**Files:**
- Create: `src/roadseg/eval.py`, `src/roadseg/stats.py`, `tests/test_stats.py`

**Interfaces:**
- Consumes: `tile_image`/`stitch` (Task 10), metrics (Task 8), `apls`/`graph_from_mask` (Task 9).
- Produces:
  - `predict_full(model, img: ndarray, tile=512, stride=256, device="cuda") -> ndarray[H,W] float32` probabilities.
  - `evaluate(model, dataset, threshold, device, with_apls=False) -> tuple[dict, list[dict]]` — aggregate metrics and per-image rows.
  - `select_threshold(model, val_dataset, device) -> float`
  - `paired_bootstrap(a: list[float], b: list[float], n=10000, seed=0) -> dict` with keys `mean_diff`, `ci_low`, `ci_high`, `p_value`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stats.py
import numpy as np, pytest
from roadseg.stats import paired_bootstrap


def test_identical_series_have_ci_containing_zero():
    x = list(np.random.default_rng(0).random(50))
    r = paired_bootstrap(x, x)
    assert r["mean_diff"] == pytest.approx(0.0)
    assert r["ci_low"] <= 0 <= r["ci_high"]


def test_clear_improvement_excludes_zero():
    rng = np.random.default_rng(0)
    base = rng.random(100) * 0.1 + 0.5
    better = base + 0.05
    r = paired_bootstrap(list(better), list(base))
    assert r["ci_low"] > 0 and r["p_value"] < 0.01


def test_noise_does_not_produce_significance():
    rng = np.random.default_rng(1)
    a, b = rng.random(100), rng.random(100)
    r = paired_bootstrap(list(a), list(b))
    assert r["ci_low"] < 0 < r["ci_high"]


def test_is_paired_not_unpaired():
    """Pairing is the point: same-image differences are far less noisy than
    comparing marginal distributions."""
    rng = np.random.default_rng(2)
    shared = rng.random(100) * 10          # huge between-image variance
    r = paired_bootstrap(list(shared + 0.1), list(shared))
    assert r["ci_low"] > 0, "paired test should detect a consistent +0.1 shift"


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap([1.0, 2.0], [1.0])
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_stats.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.stats`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/stats.py
"""Paired bootstrap over per-image scores.

Images are the resampling unit. Between-image variance in road segmentation is
large (a dense urban tile and an empty rural one differ by more than any method
does), so an unpaired comparison would drown the effect being measured.
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap(a: list[float], b: list[float], n: int = 10_000,
                     seed: int = 0, alpha: float = 0.05) -> dict:
    """Bootstrap the mean of per-image differences a[i] - b[i]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"a and b must be the same length, got {a.shape} and {b.shape}")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, p)), "n": int(len(d))}
```

```python
# src/roadseg/eval.py
"""Full-image evaluation.

Predictions are tiled, stitched back to native resolution, and only then
scored. Scoring on tiles would change the metric: overlapping borders get
double-counted and nodata handling differs at tile edges.
"""
from __future__ import annotations

import numpy as np
import torch

from .apls import apls, graph_from_mask
from .data.tiling import stitch, tile_image
from .metrics import cl_dice_metric, confusion, metrics_from_confusion, relaxed_f1


@torch.no_grad()
def predict_full(model, img: np.ndarray, tile: int = 512, stride: int = 256,
                 device: str = "cuda", batch: int = 8) -> np.ndarray:
    model.eval()
    h, w = img.shape[:2]
    tiles, pos = tile_image(img, tile, stride)
    probs = []
    for i in range(0, len(tiles), batch):
        chunk = np.stack(tiles[i:i + batch])                       # [B,T,T,3]
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            out = torch.sigmoid(model(x)).float()
        probs.extend(out[:, 0].cpu().numpy())
    return stitch(probs, pos, h, w, tile)


@torch.no_grad()
def evaluate(model, dataset, threshold: float = 0.5, device: str = "cuda",
             with_apls: bool = False) -> tuple[dict, list[dict]]:
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        prob = predict_full(model, item["image"], device=device)
        pred = prob >= threshold
        gt, valid = item["mask"], item["valid"]
        c = confusion(pred, gt, valid)
        for k in agg:
            agg[k] += c[k]
        row = {"id": item["id"], **metrics_from_confusion(c),
               **relaxed_f1(pred, gt, rho=3, valid=valid),
               "cl_dice": cl_dice_metric(pred & valid, gt & valid)}
        if with_apls:
            row["apls"] = apls(graph_from_mask(pred & valid), graph_from_mask(gt & valid))
        rows.append(row)
    keys = [k for k in rows[0] if k != "id"]
    out = {f"dataset_{k}": v for k, v in metrics_from_confusion(agg).items()}
    out.update({f"mean_{k}": float(np.mean([r[k] for r in rows])) for k in keys})
    return out, rows


@torch.no_grad()
def select_threshold(model, dataset, device: str = "cuda",
                     grid=np.arange(0.20, 0.81, 0.05)) -> float:
    """Pick the decision threshold on VALIDATION data only."""
    cached = [(predict_full(model, d["image"], device=device), d["mask"], d["valid"])
              for d in (dataset[i] for i in range(len(dataset)))]
    best_t, best_iou = 0.5, -1.0
    for t in grid:
        agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for prob, gt, valid in cached:
            for k, v in confusion(prob >= t, gt, valid).items():
                agg[k] += v
        iou = metrics_from_confusion(agg)["iou"]
        if iou > best_iou:
            best_t, best_iou = float(t), iou
    return best_t
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_stats.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/roadseg/eval.py src/roadseg/stats.py tests/test_stats.py
git commit -m "feat: full-image evaluation and paired bootstrap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Training entrypoint, configs, and SLURM array job

**Files:**
- Create: `src/roadseg/train.py`, `configs/base.yaml`, `configs/massachusetts.yaml`, `configs/deepglobe.yaml`, `scripts/sbatch_train_array.sh`, `tests/test_train_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `python -m roadseg.train --config <yaml> --rung R1 --seed 0 [--resume auto]`, writing `runs/<dataset>_<rung>_s<seed>/{latest.pt,best.pt,metrics.json}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_cli.py
import subprocess, sys, os
from pathlib import Path

ROOT = Path("/scratch/hk4488/roadseg")


def test_config_merge_and_dry_run(tmp_path):
    """--dry-run builds model, dataset, loss and optimizer without training,
    so a config typo fails in seconds rather than after queueing on an H200."""
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/.deps:{ROOT}/src")
    r = subprocess.run(
        [sys.executable, "-m", "roadseg.train", "--config", str(ROOT / "configs/deepglobe.yaml"),
         "--rung", "R2", "--seed", "3", "--dry-run"],
        capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "rung=R2" in r.stdout and "seed=3" in r.stdout


def test_rejects_unknown_rung(tmp_path):
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/.deps:{ROOT}/src")
    r = subprocess.run(
        [sys.executable, "-m", "roadseg.train", "--config", str(ROOT / "configs/deepglobe.yaml"),
         "--rung", "R9", "--dry-run"],
        capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode != 0
```

- [ ] **Step 2: Run to verify it fails**

```bash
bash scripts/pytest.sh tests/test_train_cli.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — no module `roadseg.train`.

- [ ] **Step 3: Write the configs**

```yaml
# configs/base.yaml
model: roadsegmenter
rung: R1
pretrained: true
hf_init: zero
tile: 512
stride: 256
batch_size: 16
lr: 3.0e-4
weight_decay: 1.0e-4
warmup_epochs: 2
road_bias: 0.75
num_workers: 8
amp: true
seed: 0
runs_dir: /scratch/hk4488/roadseg/runs
```

```yaml
# configs/massachusetts.yaml
_base: base.yaml
dataset: massachusetts
data_root: /scratch/hk4488/roadseg/data/massachusetts
epochs: 100
```

```yaml
# configs/deepglobe.yaml
_base: base.yaml
dataset: deepglobe
data_root: /scratch/hk4488/roadseg/data/deepglobe
epochs: 40
```

- [ ] **Step 4: Write the training entrypoint**

```python
# src/roadseg/train.py
"""Training entrypoint. One config + one rung + one seed = one run directory."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import RoadFullDataset, RoadTileDataset
from .engine import load_checkpoint, save_checkpoint, train_one_epoch
from .eval import evaluate, select_threshold
from .losses import ComboLoss
from .models.segmenter import build_model


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if "_base" in cfg:
        base = yaml.safe_load((Path(path).parent / cfg.pop("_base")).read_text())
        base.update(cfg)
        cfg = base
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rung", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto'")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.rung:
        cfg["rung"] = args.rung
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])

    run = Path(cfg["runs_dir"]) / f"{cfg['dataset']}_{cfg['rung']}_s{cfg['seed']}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg).to(device)
    loss_fn = ComboLoss(use_cldice=(cfg["rung"] == "R3"))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    print(f"rung={cfg['rung']} seed={cfg['seed']} dataset={cfg['dataset']} device={device}")
    if args.dry_run:
        print(f"dry-run ok: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
        return

    train_ds = RoadTileDataset(cfg["data_root"], cfg["dataset"], "train",
                               tile=cfg["tile"], road_bias=cfg["road_bias"])
    val_ds = RoadFullDataset(cfg["data_root"], cfg["dataset"], "val")
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)

    start_epoch, best = 0, -1.0
    resume = run / "latest.pt" if args.resume == "auto" else args.resume
    if resume and Path(resume).exists():
        meta = load_checkpoint(resume, model, opt, sched)
        start_epoch, best = meta["epoch"] + 1, meta["best"]
        print(f"resumed from {resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["epochs"]):
        tr = train_one_epoch(model, loader, loss_fn, opt, use_amp=cfg["amp"], device=device)
        sched.step()
        agg, _ = evaluate(model, val_ds, threshold=0.5, device=device)
        iou = agg["dataset_iou"]
        print(f"epoch {epoch} train_loss {tr:.4f} val_iou {iou:.4f}", flush=True)
        save_checkpoint(run / "latest.pt", model, opt, sched, epoch=epoch, best=best, cfg=cfg)
        if iou > best:
            best = iou
            save_checkpoint(run / "best.pt", model, opt, sched, epoch=epoch, best=best, cfg=cfg)

    load_checkpoint(run / "best.pt", model)
    thr = select_threshold(model, val_ds, device=device)
    agg, rows = evaluate(model, val_ds, threshold=thr, device=device, with_apls=True)
    (run / "metrics.json").write_text(json.dumps(
        {"cfg": cfg, "threshold": thr, "val": agg, "per_image": rows}, indent=2))
    print(f"done: best val IoU {best:.4f} at threshold {thr:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the SLURM array job**

```bash
# scripts/sbatch_train_array.sh
#!/bin/bash
#SBATCH --job-name=roadseg
#SBATCH --output=/scratch/hk4488/logs/roadseg/%x_%A_%a.out
#SBATCH --error=/scratch/hk4488/logs/roadseg/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_627_general
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-23

set -euo pipefail
PROJECT=/scratch/hk4488/roadseg
mkdir -p /scratch/hk4488/logs/roadseg "$PROJECT/runs"

RUNGS=(R0 R1 R2 R3)
DATASETS=(massachusetts deepglobe)
SEEDS=(0 1 2)

i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$(( i % 3 ))]}
RUNG=${RUNGS[$(( (i / 3) % 4 ))]}
DATASET=${DATASETS[$(( i / 12 ))]}

export PYTHONPATH="$PROJECT/.deps:$PROJECT/src:${PYTHONPATH:-}"
export TORCH_HOME=/scratch/hk4488/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/scratch/hk4488/.conda/envs/unsloth_sft/bin/python

echo "task $i -> dataset=$DATASET rung=$RUNG seed=$SEED on $(hostname)"
nvidia-smi -L || true

cd "$PROJECT"
exec $PY -m roadseg.train --config "configs/${DATASET}.yaml" \
     --rung "$RUNG" --seed "$SEED" --resume auto
```

- [ ] **Step 6: Run to verify tests pass**

```bash
bash scripts/pytest.sh tests/test_train_cli.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/roadseg/train.py configs/ scripts/sbatch_train_array.sh tests/test_train_cli.py
git commit -m "feat: training entrypoint, configs, and 24-run SLURM array

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: Baselines (U-Net, D-LinkNet)

**Files:**
- Create: `src/roadseg/models/baselines/__init__.py`, `src/roadseg/models/baselines/unet.py`, `src/roadseg/models/baselines/dlinknet.py`, `tests/test_baselines.py`

**Interfaces:**
- Produces: `build_baseline(name: str) -> nn.Module` for `name in {"unet", "dlinknet"}`; both emit `[B,1,H,W]` logits at input resolution, same contract as `RoadSegmenter`.

This is the cuttable tail (spec §8). Everything above stands without it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_baselines.py
import pytest, torch
from roadseg.models.baselines import build_baseline


@pytest.mark.parametrize("name", ["unet", "dlinknet"])
def test_output_contract_matches_segmenter(name):
    m = build_baseline(name)
    out = m(torch.randn(1, 3, 128, 128))
    assert out.shape == (1, 1, 128, 128)


@pytest.mark.parametrize("name", ["unet", "dlinknet"])
def test_emits_logits(name):
    with torch.no_grad():
        out = build_baseline(name)(torch.randn(2, 3, 64, 64) * 5)
    assert out.min() < 0.0


def test_unknown_baseline_raises():
    with pytest.raises(ValueError, match="unknown baseline"):
        build_baseline("nope")
```

- [ ] **Step 2: Run to verify they fail**

```bash
bash scripts/pytest.sh tests/test_baselines.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: FAIL — `ModuleNotFoundError: roadseg.models.baselines`.

- [ ] **Step 3: Implement**

```python
# src/roadseg/models/baselines/unet.py
"""Plain U-Net, no pretrained encoder. Reference point for the comparison table."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, widths=(64, 128, 256, 512), bottleneck=1024) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        c = 3
        for w in widths:
            self.downs.append(_block(c, w)); c = w
        self.bottleneck = _block(c, bottleneck)
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.up_blocks.append(_block(w * 2, w)); c = w
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, blk, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            x = blk(torch.cat([up(x), skip], dim=1))
        return self.head(x)
```

```python
# src/roadseg/models/baselines/dlinknet.py
"""D-LinkNet: ResNet-34 encoder + dilated centre block + LinkNet decoder.

Winner of the DeepGlobe 2018 road extraction challenge. The dilated block
widens the receptive field without losing resolution, which is what makes it
strong on long thin roads.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


class _Dilated(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(ch, ch, 3, padding=d, dilation=d) for d in (1, 2, 4, 8)]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for b in self.branches:
            out = out + self.relu(b(x))
        return out


class _Up(nn.Module):
    def __init__(self, i: int, o: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(i, i // 4, 1, bias=False), nn.BatchNorm2d(i // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(i // 4, i // 4, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(i // 4), nn.ReLU(inplace=True),
            nn.Conv2d(i // 4, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DLinkNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.e1, self.e2, self.e3, self.e4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.center = _Dilated(512)
        self.d4, self.d3, self.d2, self.d1 = _Up(512, 256), _Up(256, 128), _Up(128, 64), _Up(64, 64)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stem(x)
        e1 = self.e1(self.pool(s))
        e2, e3 = self.e2(e1), self.e3(self.e2(e1))
        e4 = self.center(self.e4(e3))
        d = self.d4(e4) + e3
        d = self.d3(d) + e2
        d = self.d2(d) + e1
        d = self.d1(d)
        return self.head(d)
```

```python
# src/roadseg/models/baselines/__init__.py
from __future__ import annotations

import torch.nn as nn

from .dlinknet import DLinkNet
from .unet import UNet


def build_baseline(name: str, pretrained: bool = True) -> nn.Module:
    if name == "unet":
        return UNet()
    if name == "dlinknet":
        return DLinkNet(pretrained=pretrained)
    raise ValueError(f"unknown baseline {name!r}; expected 'unet' or 'dlinknet'")
```

- [ ] **Step 4: Run to verify they pass**

```bash
bash scripts/pytest.sh tests/test_baselines.py -v > /scratch/hk4488/_probe.txt 2>&1
```
Expected: 5 passed.

- [ ] **Step 5: Run the whole suite**

```bash
bash scripts/pytest.sh -q > /scratch/hk4488/_probe.txt 2>&1
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/roadseg/models/baselines/ tests/test_baselines.py
git commit -m "feat: U-Net and D-LinkNet baselines

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Post-implementation: running the experiments

These are not implementation tasks; they are what the code is for. Run in order.

1. `bash scripts/download_data.sh` — needs a Kaggle token for DeepGlobe (spec §14).
2. Smoke test one run end-to-end before queueing 24:
   `sbatch --array=0-0 scripts/sbatch_train_array.sh`, then check `seff <JOBID>` for >70% GPU utilization.
3. `sbatch scripts/sbatch_train_array.sh` for the full ladder.
4. Aggregate `runs/*/metrics.json`, compute `paired_bootstrap(R1_per_image_iou, R0_per_image_iou)`, and report the interval **whichever side of zero it lands on**.
5. Only after all selection is frozen, evaluate the test split once.
