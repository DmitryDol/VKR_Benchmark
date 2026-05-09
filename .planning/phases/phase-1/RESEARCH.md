# Phase 1: RT-DETR Adapter & ONNX Pipeline - Research

**Researched:** 2026-05-10
**Domain:** HuggingFace Transformers RT-DETR, PyTorch ONNX export, COCO benchmarking
**Confidence:** HIGH (codebase confirmed by direct inspection; HF API confirmed via Context7)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**RT-DETR Source Library:** HuggingFace `transformers.RTDetrForObjectDetection`
- `pixel_values: (1,3,H,W)` tensor input matches `PyTorchEngine.infer()` directly
- Model ID: `PekingU/rtdetr-r50`
- `transformers` and `huggingface_hub` must be added to project dependencies

**Model Variant:** `PekingU/rtdetr-r50` (ResNet-50 backbone, 42M params, ~280MB, mAP 53.1)
- Input size: 640×640 (fixed)
- Expected VRAM FP32: ~2.5 GB (safe on RTX 3070 8GB)

**Weight Management:** Download script → local `weights/rtdetr-r50/` directory
- New script: `scripts/download_weights.py`
- Uses `huggingface_hub.snapshot_download("PekingU/rtdetr-r50", local_dir="weights/rtdetr-r50")`
- `ModelAdapter.load(weights_path)` calls `from_pretrained(str(weights_path))`
- `weights/` added to `.gitignore`

**Bug Fix (FIX-01) is first task:**
- Location: `src/benchmark/engines/base.py:96-97`
- Fix: store `raw = self.infer(inputs)` then `self.postprocess(raw, sample)`

### Architecture Decisions (Claude's Discretion)
- Adapter file: `src/benchmark/models/rtdetr_adapter.py` (new `models/` package)
- Package init: `src/benchmark/models/__init__.py` exporting `RTDETRAdapter`
- Score threshold: 0.01 (existing default in `PyTorchEngine`)
- Preprocessing: 640×640 resize + ImageNet normalization (existing constants)
- ONNX export: thin `nn.Module` wrapper accepting positional tensor; use existing `export_and_simplify()`
- Input/output names: `pixel_values` → `logits` + `pred_boxes`

### Deferred Ideas (OUT OF SCOPE)
- RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26 adapters (Phase v2)
- OnnxRuntimeEngine / ONNX inference engine (Phase 2)
- Metrics/logging expansion LOG-* (Phase 2)
- CLI interface CLI-* (Phase 2)
- Strategy C / Sensitivity Analysis ADV-01 (Phase v2)
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FIX-01 | Fix double `infer()` call in warm-up loop (`base.py:96-97`) | Bug confirmed in source; exact fix documented below |
| ADPT-01 | Load RT-DETR pretrained weights and run inference | HF `RTDetrForObjectDetection.from_pretrained()` API verified |
| ADPT-02 | Parse RT-DETR outputs into Detection format (boxes, scores, labels) | Output format (`logits`, `pred_boxes`) documented with conversion logic |
| ADPT-03 | Download and manage pretrained weights | `huggingface_hub.snapshot_download()` pattern documented |
| ONNX-01 | Export to ONNX with opset 17 | `export_to_onnx()` exists; RT-DETR wrapper pattern documented |
| ONNX-02 | Apply onnxsim automatically | `simplify_onnx()` / `export_and_simplify()` exist and work |
| ONNX-03 | Validate ONNX after export | `validate_onnx()` exists; output name mismatch risk documented |
| BENCH-01 | 50 warm-up iterations before measurement | `WARMUP_RUNS=50` constant confirmed in `base.py`; FIX-01 unblocks this |
| BENCH-02 | 1000 measured iterations averaged | `MEASURE_RUNS=1000` constant confirmed in `base.py` |
| BENCH-03 | Batch size 1 for all inference | Enforced by `COCODataLoader` + dummy input shape `(1,3,H,W)` |
| BENCH-04 | Disable TF32 for FP32 baseline | Already implemented in `PyTorchEngine.load_model()` |
| BENCH-05 | Reset VRAM and clear CUDA cache between engines | `reset_vram_tracking()` implemented in `BaseEngine` |
| BENCH-06 | CUDA sync between timing points | `torch.cuda.synchronize()` already in `benchmark_latency()` |
</phase_requirements>

---

## Summary

Phase 1 is primarily an adapter + plumbing phase. The core benchmarking infrastructure (`BaseEngine`, `COCODataLoader`, `ResultLogger`, `onnx_export`) is fully implemented and correct except for one confirmed bug. The bug (FIX-01) causes the warm-up to run inference twice per iteration, inflating the GPU kernel cache state in a non-representative way — it must be fixed first.

The main new code is `RTDETRAdapter`, a concrete implementation of the existing `ModelAdapter` Protocol. It bridges HuggingFace `RTDetrForObjectDetection` into the `PyTorchEngine`. The adapter handles: (1) loading weights from a local HF snapshot, (2) parsing `logits (1,300,92)` + `pred_boxes (1,300,4)` into `Detection` with COCO-91 label IDs and x1y1x2y2 pixel coordinates, (3) wrapping the HF model for `torch.onnx.export` which requires positional args not kwargs.

`transformers` is not yet installed — it must be added to `pyproject.toml`. `huggingface_hub` comes as a transitive dependency of `transformers`.

**Primary recommendation:** Fix FIX-01, add `transformers>=4.43.0` to deps, implement `RTDETRAdapter`, then wire the runner script.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Weight download | Scripts (data layer) | — | Mirrors `data/download_coco.py` pattern; offline reproducibility |
| Model loading | `models/` adapter layer | `engines/` (calls adapter) | ModelAdapter Protocol decouples loading from engine |
| Preprocessing (resize + normalize) | `PyTorchEngine.preprocess()` | Adapter (provides `input_size`) | Already implemented; adapter only declares size |
| Inference | `PyTorchEngine.infer()` | — | `torch.no_grad()` forward pass |
| Output parsing (boxes/scores/labels) | `RTDETRAdapter.parse_outputs()` | — | Model-specific logic isolated in adapter |
| ONNX export | `engines/onnx_export.py` | Adapter (provides wrapper) | Existing `export_and_simplify()` reused with RT-DETR wrapper |
| Benchmarking loop | `BaseEngine.benchmark_latency()` | — | After FIX-01 this is correct |
| mAP evaluation | `BaseEngine.evaluate_accuracy()` | `pycocotools` | Already correct |
| Result persistence | `ResultLogger` | — | Already implemented |

---

## Standard Stack

### Core (new dependencies to add)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | >=4.43.0 | `RTDetrForObjectDetection`, `RTDetrImageProcessor` | Official HF RT-DETR implementation; `RTDetrForObjectDetection` added in 4.43 [CITED: github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/rt_detr.md] |
| huggingface_hub | (transitive) | `snapshot_download()` for weight management | Ships with `transformers` |

### Existing Stack (already installed and locked)
| Library | Version | Purpose |
|---------|---------|---------|
| torch | 2.11.0+cu130 | FP32 inference, ONNX tracing |
| torchvision | 0.26.0+cu130 | Image preprocessing transforms |
| onnx | 1.21.0 | Model interchange format |
| onnxsim | 0.6.3 | Graph simplification |
| pycocotools | 2.0.11 | mAP evaluation |

**CRITICAL FINDING:** RT-DETR is **NOT** present in `torchvision 0.26.0+cu130`. Direct inspection of the installed package confirms `torchvision.models` has no `rtdetr_*` builders and `torchvision.models.detection` has no DETR-family models. [VERIFIED: direct Python inspection of installed package]

**Version verification:**
```bash
uv add "transformers>=4.43.0"
# Check npm view equivalent:
# transformers 4.51.3 is latest stable as of research date [VERIFIED: Context7 library listing]
```

**Installation:**
```bash
uv add "transformers>=4.43.0"
```

---

## Architecture Patterns

### System Architecture Diagram

```
COCO val2017 images
       |
       v
COCODataLoader.yield(COCOSample)
       |
       v
PyTorchEngine.preprocess()     <-- resize 640x640 + ImageNet normalize
       |
       v                         RTDETRAdapter.input_size = (640, 640)
PyTorchEngine.infer()          <-- model(pixel_values) → RTDetrObjectDetectionOutput
       |
       v                         RTDETRAdapter.parse_outputs()
PyTorchEngine.postprocess()    <-- logits sigmoid + box cxcywh→xyxy + threshold
       |
       v
Detection(boxes, scores, labels)
       |
  +-----------+
  |           |
  v           v
benchmark   evaluate_accuracy()
_latency()        |
  |          pycocotools.COCOeval
  v               |
BenchmarkResult   v
  |           mAP metrics
  v
ResultLogger → results/*.csv / *.json


ONNX Export Path (separate):
RTDetrONNXWrapper(nn.Module) — accepts positional pixel_values tensor
       |
export_to_onnx() → rtdetr_r50.onnx
       |
simplify_onnx() → rtdetr_r50_sim.onnx
       |
validate_onnx() → onnx.checker.check_model()
```

### Recommended Project Structure
```
src/benchmark/
├── models/                    # NEW package for model adapters
│   ├── __init__.py            # exports RTDETRAdapter
│   └── rtdetr_adapter.py      # RTDETRAdapter + RTDetrONNXWrapper
├── engines/
│   ├── base.py                # FIX-01 applied here (line 96-97)
│   ├── pytorch_engine.py      # unchanged
│   └── onnx_export.py         # minor: update DEFAULT_DYNAMIC_AXES note
├── data/
│   └── coco_loader.py         # unchanged
└── utils/
    └── logger.py              # unchanged

scripts/
└── download_weights.py        # NEW: huggingface_hub.snapshot_download

weights/
└── rtdetr-r50/                # gitignored; HF snapshot lands here
```

### Pattern 1: RTDETRAdapter implementing ModelAdapter Protocol

```python
# Source: Context7 /huggingface/transformers rt_detr.md + ModelAdapter Protocol (pytorch_engine.py)
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from transformers import RTDetrForObjectDetection

from benchmark.engines.base import Detection
from benchmark.data.coco_loader import COCO_91_TO_80  # for reference; labels come as COCO-91 IDs directly

if TYPE_CHECKING:
    from benchmark.data.coco_loader import COCOSample


class RTDetrONNXWrapper(nn.Module):
    """Thin wrapper making RTDetrForObjectDetection ONNX-traceable.

    torch.onnx.export requires positional args; HF models use kwargs (pixel_values=...).
    This wrapper converts positional input to the expected kwarg call.
    Returns (logits, pred_boxes) tuple — ONNX does not support dataclass outputs.
    """

    def __init__(self, model: RTDetrForObjectDetection) -> None:
        super().__init__()
        self._model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self._model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes


class RTDETRAdapter:
    """ModelAdapter for PekingU/rtdetr-r50 via HuggingFace transformers."""

    @property
    def input_size(self) -> tuple[int, int]:
        return (640, 640)

    def load(self, weights_path: Path, device: torch.device) -> nn.Module:
        model = RTDetrForObjectDetection.from_pretrained(str(weights_path))
        model.eval()
        return model.to(device)

    def parse_outputs(
        self,
        raw_outputs: object,
        original_size: tuple[int, int],
        input_size: tuple[int, int],
        score_threshold: float,
    ) -> Detection:
        # raw_outputs is RTDetrObjectDetectionOutput with .logits and .pred_boxes
        # logits: (1, 300, 92) — 91 COCO classes + 1 background (index 0 = background in HF RT-DETR)
        # pred_boxes: (1, 300, 4) — cx, cy, w, h normalized [0,1]
        from transformers.models.rt_detr.modeling_rt_detr import RTDetrObjectDetectionOutput
        assert isinstance(raw_outputs, RTDetrObjectDetectionOutput)

        logits = raw_outputs.logits[0]    # (300, 92)
        pred_boxes = raw_outputs.pred_boxes[0]  # (300, 4)

        # Sigmoid scores; take max class (exclude background at index 0)
        scores_all = torch.sigmoid(logits[:, 1:])  # (300, 91) — strip background
        scores, class_indices = scores_all.max(dim=-1)  # (300,) each

        # class_indices are 0-indexed over 91 classes (shifted by 1 from COCO-91 IDs)
        # COCO-91 label ID = class_index + 1
        label_ids = class_indices + 1  # (300,) COCO-91 IDs

        # Filter by threshold
        keep = scores >= score_threshold
        scores = scores[keep]
        label_ids = label_ids[keep]
        boxes_norm = pred_boxes[keep]  # (N, 4) cx,cy,w,h normalized

        # Convert cx,cy,w,h → x1,y1,x2,y2 in pixel coords
        orig_h, orig_w = original_size
        cx, cy, w, h = boxes_norm.unbind(-1)
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        return Detection(
            boxes=boxes_xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4),
            scores=scores.cpu().numpy().astype(np.float32),
            labels=label_ids.cpu().numpy().astype(np.int64),
        )
```

### Pattern 2: ONNX export for RT-DETR

```python
# Source: onnx_export.py existing API + RT-DETR wrapper above
# Call site in runner script:
from benchmark.engines.onnx_export import export_and_simplify
from benchmark.models.rtdetr_adapter import RTDetrONNXWrapper

wrapper = RTDetrONNXWrapper(engine.model)  # engine.model is RTDetrForObjectDetection
wrapper.eval()

# RT-DETR has two outputs: logits + pred_boxes — need separate output names
dynamic_axes = {
    "pixel_values": {0: "batch"},
    "logits": {0: "batch"},
    "pred_boxes": {0: "batch"},
}

# Cannot use export_and_simplify() directly because output_names differ from default.
# Must call export_to_onnx() with explicit output_names:
from benchmark.engines.onnx_export import export_to_onnx, simplify_onnx

raw_path = export_to_onnx(
    wrapper,
    output_path=Path("weights/rtdetr-r50/rtdetr_r50.onnx"),
    input_size=(640, 640),
    opset_version=17,
    dynamic_axes=dynamic_axes,
    input_names=["pixel_values"],
    output_names=["logits", "pred_boxes"],
)
sim_path = simplify_onnx(raw_path, output_path=raw_path)
```

**Note:** `export_to_onnx()` currently hardcodes `input_names=["input"]` and `output_names=["output"]` inside the function. The function signature does not accept `input_names`/`output_names` as parameters. **This is a required code change:** add `input_names` and `output_names` parameters to `export_to_onnx()` (with defaults `["input"]` and `["output"]` for backward compatibility).

### Pattern 3: Weight download script

```python
# Source: CONTEXT.md decision + huggingface_hub API [CITED: huggingface.co/docs/huggingface_hub]
from huggingface_hub import snapshot_download
from pathlib import Path

def download_rtdetr_r50(local_dir: Path = Path("weights/rtdetr-r50")) -> Path:
    """Download PekingU/rtdetr-r50 weights to local directory."""
    snapshot_download(
        repo_id="PekingU/rtdetr-r50",
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "*.h5", "flax_*"],  # skip non-PyTorch formats
    )
    return local_dir
```

### Anti-Patterns to Avoid

- **Calling `model(**inputs)` where `inputs` is the dict from `RTDetrImageProcessor`:** In `PyTorchEngine`, the preprocessed input is a plain tensor `(1,3,H,W)`, not the full processor dict. The adapter's `load()` returns a model that must accept `pixel_values=tensor` kwargs. `PyTorchEngine.infer()` calls `model(inputs)` positionally — this will fail unless the HF model is used via the `RTDetrONNXWrapper` approach or `model(pixel_values=inputs)` syntax. **Resolution:** Implement adapter's `load()` to return `RTDetrONNXWrapper` (or override `PyTorchEngine.infer()` to call `model(pixel_values=inputs)`). Simplest: return the bare HF model from `load()` and have `PyTorchEngine.infer()` call `self._model(inputs)` — HF models accept positional `pixel_values` as first argument. [ASSUMED — needs runtime verification]

- **Assuming logits shape is (1,300,91):** HF RT-DETR logits are `(batch, 300, num_labels+1)` where `num_labels=91` for COCO, giving shape `(1,300,92)`. Index 0 is background. [ASSUMED based on DETR family convention — verify after transformers install]

- **Using `DEFAULT_DYNAMIC_AXES` from `onnx_export.py` for RT-DETR:** The default names `"input"` and `"output"` do not match RT-DETR's two-output structure. Always pass explicit `dynamic_axes` with `"pixel_values"`, `"logits"`, `"pred_boxes"`.

- **Calling `torch.onnx.export` directly on `RTDetrForObjectDetection`:** HF models use kwargs in forward; tracing requires positional args. Always wrap in `RTDetrONNXWrapper` first.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Box format conversion (cx,cy,w,h → x1y1x2y2) | Custom converter | PyTorch tensor arithmetic (already shown above) | Trivial math, no library needed |
| mAP computation | Custom AP loop | `pycocotools.COCOeval` (already in codebase) | Handles area ranges, IoU thresholds correctly |
| ONNX graph simplification | Manual op fusion | `onnxsim` (already in codebase) | Removes redundant nodes, cast ops |
| Model download | Custom HTTP download | `huggingface_hub.snapshot_download` | Handles resume, checksums, auth |
| HF model ONNX export pipeline | Custom exporter | `RTDetrONNXWrapper` + existing `export_to_onnx()` | Reuse existing infrastructure |

---

## FIX-01: Warm-up Bug — Exact Diff

**File:** `src/benchmark/engines/base.py`
**Location:** Lines 93–97

```python
# BEFORE (buggy) — double inference per warm-up iteration:
for i in range(WARMUP_RUNS):
    sample = samples[i % n_samples]
    inputs = self.preprocess(sample)
    self.infer(inputs)                           # result discarded
    self.postprocess(self.infer(inputs), sample) # second infer call

# AFTER (correct) — single inference per warm-up iteration:
for i in range(WARMUP_RUNS):
    sample = samples[i % n_samples]
    inputs = self.preprocess(sample)
    raw = self.infer(inputs)
    self.postprocess(raw, sample)
```

**Impact:** Bug doubles GPU work during warm-up. With RTX 3070 and RT-DETR, ~50 extra forward passes (~5s extra wall time, pollutes GPU caches). Does not affect measurement phase (lines 109–133 are correct).

---

## ONNX Export: Required Change to `export_to_onnx()`

The existing function signature:
```python
def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
```

**Missing:** `input_names` and `output_names` parameters. These are currently hardcoded as `["input"]` and `["output"]` inside the function body (lines 79–80).

**Required addition:**
```python
def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    input_names: list[str] | None = None,   # NEW — default ["input"]
    output_names: list[str] | None = None,  # NEW — default ["output"]
) -> Path:
```

This is backward-compatible (existing callers get `["input"]`/`["output"]` by default).

---

## Common Pitfalls

### Pitfall 1: HF Model Forward Pass with Positional Args
**What goes wrong:** `PyTorchEngine.infer()` calls `self._model(inputs)` positionally. `RTDetrForObjectDetection.forward()` signature is `forward(pixel_values=None, ...)`. Passing positionally works if `pixel_values` is the first positional parameter, but the return type is `RTDetrObjectDetectionOutput` (a dataclass), not a plain tensor.
**Why it happens:** HF models return structured output objects, not tensors.
**How to avoid:** `RTDETRAdapter.load()` returns bare HF model (not wrapper). `PyTorchEngine.infer()` returns `RTDetrObjectDetectionOutput`. `RTDETRAdapter.parse_outputs()` accesses `.logits` and `.pred_boxes` attributes.
**Warning signs:** `AttributeError: 'RTDetrObjectDetectionOutput' object has no attribute 'shape'` in postprocess.

### Pitfall 2: ONNX Export Fails on Dataclass Output
**What goes wrong:** `torch.onnx.export` cannot trace a model that returns a dataclass (`RTDetrObjectDetectionOutput`). Tracer expects tensors or tuples of tensors.
**Why it happens:** HF models return named output objects for usability; ONNX tracer needs plain tensor outputs.
**How to avoid:** Always wrap in `RTDetrONNXWrapper` which returns `(logits, pred_boxes)` tuple before calling `export_to_onnx()`.
**Warning signs:** `TypeError: outputs must be a Tensor or a tuple of Tensors` during export.

### Pitfall 3: Class Index Off-by-One (COCO-91 vs 0-indexed)
**What goes wrong:** HF RT-DETR `logits` last dimension has `num_classes + 1` entries (background at index 0). If background is not stripped, `argmax` will sometimes predict background class, producing label ID 0 which does not exist in COCO-91 mapping.
**Why it happens:** DETR-family models include explicit background class in logit head.
**How to avoid:** `scores_all = torch.sigmoid(logits[:, 1:])` — slice off background. Then `label_id = class_idx + 1`.
**Warning signs:** Detections with `category_id=0` appearing in COCO results; `pycocotools` throws KeyError or ignores them → mAP = 0.

### Pitfall 4: `onnxsim` Fails Silently
**What goes wrong:** `simplify_onnx()` logs a warning but continues if `check_ok=False`. The saved model may still work but is not simplified.
**Why it happens:** `onnxsim.simplify()` returns `(simplified_model, check_ok)` where `check_ok` can be False for complex models with dynamic control flow.
**How to avoid:** RT-DETR with fixed 640×640 input and `RTDetrONNXWrapper` (no dynamic shapes except batch) should simplify cleanly. If not, verify the wrapper returns plain tensors.
**Warning signs:** Log line `"onnxsim validation failed for..."` — investigate rather than ignore for TensorRT compatibility.

### Pitfall 5: `transformers` Not Installed
**What goes wrong:** `from transformers import RTDetrForObjectDetection` raises `ModuleNotFoundError`.
**Why it happens:** `transformers` is not in `pyproject.toml` and not in `uv.lock`. [VERIFIED: uv pip show transformers returns "not found"]
**How to avoid:** Add `transformers>=4.43.0` to `pyproject.toml` dependencies and run `uv sync`.
**Warning signs:** Import error on first run of any code importing `RTDETRAdapter`.

### Pitfall 6: `model_size_mb` Measures Parameters, Not File
**What goes wrong:** `PyTorchEngine.model_size_mb` computes in-memory parameter footprint (FP32: 4 bytes/param × 42M = ~161MB), while the `.safetensors` file on disk is ~161MB. These will roughly match for FP32 but diverge in quantized stages.
**Why it happens:** Existing implementation: `sum(p.numel() * p.element_size() for p in self._model.parameters()) / (1024*1024)`. This is correct for Phase 1 FP32 measurements. [VERIFIED: pytorch_engine.py line 152]
**How to avoid:** No change needed for Phase 1. Document that subsequent phases should use file size for ONNX/TRT engines.

---

## Code Examples

### Loading RT-DETR with ModelAdapter protocol (verified pattern)
```python
# Source: Context7 /huggingface/transformers rt_detr.md
from transformers import RTDetrForObjectDetection

model = RTDetrForObjectDetection.from_pretrained("weights/rtdetr-r50")
model.eval().to(torch.device("cuda"))

# Forward pass — pixel_values is first positional arg
with torch.no_grad():
    outputs = model(pixel_values)       # returns RTDetrObjectDetectionOutput
    # outputs.logits: (1, 300, 92)
    # outputs.pred_boxes: (1, 300, 4)  cx,cy,w,h normalized
```

### Converting boxes to x1y1x2y2 pixel coords
```python
# Standard DETR box conversion — [ASSUMED: standard DETR convention]
orig_h, orig_w = original_size   # from COCOSample.original_size
cx, cy, w, h = pred_boxes.unbind(-1)   # all normalized [0,1]
x1 = (cx - w / 2) * orig_w
y1 = (cy - h / 2) * orig_h
x2 = (cx + w / 2) * orig_w
y2 = (cy + h / 2) * orig_h
```

### snapshot_download for weight management
```python
# Source: huggingface_hub API [CITED: huggingface.co/docs/huggingface_hub/package_reference/file_download]
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="PekingU/rtdetr-r50",
    local_dir="weights/rtdetr-r50",
    ignore_patterns=["*.msgpack", "*.h5", "flax_*"],
)
```

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.13 | All | ✓ | 3.13.5 | — |
| CUDA / RTX 3070 | Inference, VRAM tracking | ✓ | sm_86, 8GB | — |
| torch 2.11.0+cu130 | Engines | ✓ | 2.11.0+cu130 | — |
| torchvision 0.26.0+cu130 | Preprocessing | ✓ | 0.26.0+cu130 | — |
| onnx 1.21.0 | ONNX export | ✓ | 1.21.0 | — |
| onnxsim 0.6.3 | Simplification | ✓ | 0.6.3 | — |
| pycocotools 2.0.11 | mAP eval | ✓ | 2.0.11 | — |
| transformers | RTDETRAdapter | ✗ | not installed | None — must install |
| huggingface_hub | Weight download | ✗ | transitive via transformers | None — must install |
| COCO val2017 data | mAP evaluation | [ASSUMED ✓] | ~1GB on disk | Cannot evaluate mAP without it |
| Internet access | First weight download | [ASSUMED ✓] | — | Pre-download manually |

**Missing dependencies with no fallback:**
- `transformers>=4.43.0` — blocks all of ADPT-01, ADPT-02, ADPT-03. Must `uv add transformers` before any adapter code runs.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (not yet configured) |
| Config file | none — Wave 0 gap |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FIX-01 | Warm-up calls `infer()` exactly once per iteration | unit | `uv run pytest tests/test_base_engine.py::test_warmup_single_infer -x` | ❌ Wave 0 |
| ADPT-01 | `RTDETRAdapter.load()` returns nn.Module on CUDA | integration | `uv run pytest tests/test_rtdetr_adapter.py::test_load -x` | ❌ Wave 0 |
| ADPT-02 | `parse_outputs()` returns Detection with boxes in x1y1x2y2, scores in [0,1], COCO-91 labels | unit | `uv run pytest tests/test_rtdetr_adapter.py::test_parse_outputs -x` | ❌ Wave 0 |
| ADPT-03 | `download_weights.py` creates `weights/rtdetr-r50/config.json` | smoke | `uv run pytest tests/test_download_weights.py::test_local_weights_exist -x` | ❌ Wave 0 |
| ONNX-01 | Exported `.onnx` file exists and is >10MB | smoke | `uv run pytest tests/test_onnx_export.py::test_export_creates_file -x` | ❌ Wave 0 |
| ONNX-02 | `simplify_onnx()` reduces node count vs original | unit | `uv run pytest tests/test_onnx_export.py::test_simplify_reduces_nodes -x` | ❌ Wave 0 |
| ONNX-03 | `validate_onnx()` passes without exception | unit | `uv run pytest tests/test_onnx_export.py::test_validate_passes -x` | ❌ Wave 0 |
| BENCH-01-06 | `benchmark_latency()` returns dict with all 6 keys, values > 0 | integration | `uv run pytest tests/test_benchmark.py::test_latency_returns_complete_dict -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q --ignore=tests/test_benchmark.py` (skip slow GPU tests)
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/conftest.py` — shared fixtures (mock COCOSample, mock model)
- [ ] `tests/test_base_engine.py` — covers FIX-01, BENCH-01–06
- [ ] `tests/test_rtdetr_adapter.py` — covers ADPT-01, ADPT-02
- [ ] `tests/test_download_weights.py` — covers ADPT-03
- [ ] `tests/test_onnx_export.py` — covers ONNX-01, ONNX-02, ONNX-03
- [ ] Framework install: `uv add --dev pytest` — pytest not in pyproject.toml

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (limited) | Validate `weights_path` exists before `from_pretrained()`; guard empty detections (already in `BaseEngine`) |
| V6 Cryptography | no | — |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Pickle deserialization via `.pth` weights | Tampering | Use HF `safetensors` format (default for `PekingU/rtdetr-r50` on Hub); `from_pretrained()` prefers safetensors |
| Malicious ONNX model loading | Tampering | `onnx.checker.check_model()` in `validate_onnx()` — already implemented |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | HF RT-DETR `logits` shape is `(batch, 300, 92)` — 91 COCO classes + background at index 0 | Output Parsing pattern | parse_outputs() produces wrong label IDs; mAP = 0. Verify after `transformers` install with `print(outputs.logits.shape)` |
| A2 | `RTDetrForObjectDetection.forward()` accepts positional `pixel_values` as first arg (no wrapper needed for `PyTorchEngine.infer()`) | Pitfall 1, Pattern 1 | `infer()` fails with TypeError; need alternative calling convention |
| A3 | `PekingU/rtdetr-r50` HF Hub repo ID matches the model discussed (not `rtdetr_r50vd`) | Standard Stack | Download fetches wrong checkpoint; mAP differs from expectation |
| A4 | COCO val2017 images are present at `data/val2017/` | Environment Availability | `evaluate_accuracy()` raises `FileNotFoundError` |

**If this table is empty:** N/A — four assumptions require runtime verification post-install.

---

## Open Questions

1. **Exact `logits` shape after `transformers` install**
   - What we know: DETR family uses background class; HF docs show `id2label` has 91 entries for COCO
   - What's unclear: Is background at index 0 or index 91? Is shape `(1,300,92)` or `(1,300,91)`?
   - Recommendation: After `uv add transformers`, run one inference and `print(outputs.logits.shape)` to confirm before implementing `parse_outputs()`

2. **`PekingU/rtdetr-r50` vs `PekingU/rtdetr_r50vd` Hub ID**
   - What we know: Context7 docs show `PekingU/rtdetr_r50vd`; CONTEXT.md specifies `PekingU/rtdetr-r50`
   - What's unclear: Both may exist; `rtdetr-r50` vs `rtdetr_r50vd` is different backbone (plain ResNet vs VD variant)
   - Recommendation: Verify Hub ID exists with `huggingface_hub.list_models(author="PekingU")` before download

3. **`export_to_onnx()` signature extension — is it needed or can `export_and_simplify()` be bypassed?**
   - What we know: Current signature lacks `input_names`/`output_names` params
   - What's unclear: Whether the planner wants to modify `onnx_export.py` or create a separate RT-DETR-specific export function
   - Recommendation: Modify `export_to_onnx()` with default-compatible params (backward safe)

---

## Sources

### Primary (HIGH confidence)
- Direct Python inspection of `torchvision 0.26.0+cu130` package — confirms RT-DETR **absent**
- Direct source read: `src/benchmark/engines/base.py` — FIX-01 bug confirmed at lines 96–97
- Direct source read: `src/benchmark/engines/pytorch_engine.py` — ModelAdapter protocol, `model_size_mb` implementation
- Direct source read: `src/benchmark/engines/onnx_export.py` — `export_to_onnx()` signature, missing `input_names`/`output_names`
- Context7 `/huggingface/transformers` — `RTDetrForObjectDetection` usage pattern, `RTDetrImageProcessor`, `post_process_object_detection()`

### Secondary (MEDIUM confidence)
- Context7 transformers docs — ONNX export via `optimum-cli` (not used in this project; we use `torch.onnx.export` directly)
- CONTEXT.md `01-CONTEXT.md` — locked decisions including HF model ID, weight path, adapter structure

### Tertiary (LOW confidence)
- [ASSUMED] logits shape `(1,300,92)` with background at index 0 — DETR family convention, not confirmed for this specific HF RT-DETR version
- [ASSUMED] `RTDetrForObjectDetection.forward()` accepts positional `pixel_values` — standard HF convention

---

## Metadata

**Confidence breakdown:**
- FIX-01 bug: HIGH — confirmed by direct source read
- RT-DETR absent from torchvision: HIGH — confirmed by direct package inspection
- HF API (`RTDetrForObjectDetection`, `from_pretrained`, usage pattern): HIGH — Context7 verified
- RT-DETR output tensor shapes: MEDIUM — ASSUMED from DETR family convention, needs runtime confirmation
- ONNX wrapper pattern: HIGH — standard solution for HF model ONNX export

**Research date:** 2026-05-10
**Valid until:** 2026-06-10 (30 days — transformers API is stable)
