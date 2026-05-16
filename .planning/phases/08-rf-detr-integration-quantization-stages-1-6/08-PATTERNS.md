# Phase 8: RF-DETR Integration & Quantization (Stages 1-6) - Pattern Map

**Mapped:** 2026-05-16
**Files analyzed:** 5 (1 CREATE adapter, 1 CREATE script, 3 MODIFY in 2 files)
**Analogs found:** 5 / 5

## File Classification

| New / Modified File                                                       | Role                         | Data Flow                              | Closest Analog                                                                    | Match Quality |
|---------------------------------------------------------------------------|------------------------------|----------------------------------------|-----------------------------------------------------------------------------------|---------------|
| `src/benchmark/models/rfdetr_adapter.py` (CREATE)                         | model adapter (Protocol impl) | request-response (tensor in → Detection out) | `src/benchmark/models/rtdetr_adapter.py`                                          | **exact** (same DETR family, same Protocol, same `(N,4)` + `(N,classes)` output shape) |
| `scripts/export_rfdetr_onnx.py` (CREATE)                                  | script / export utility       | batch transform (PyTorch → ONNX file)  | `scripts/export_rtdetr_onnx.py`                                                   | **exact** (same opset=18 + `simplify_onnx` + `validate_onnx` pipeline) |
| `src/benchmark/cli.py` — `MODEL_REGISTRY` (MODIFY +5 lines)               | config / registry             | static lookup                          | existing `"rt-detr"`, `"yolo11l"`, `"yolo26l"` dict entries (cli.py:74-88)        | **exact** |
| `src/benchmark/cli.py` — `_get_adapter` (MODIFY +3 lines)                 | factory / dispatch            | request-response                       | existing `if model_name == "rt-detr": ...` branch (cli.py:109-116)                | **exact** |
| `src/benchmark/cli.py` — Stage-1 `compute_macs` call (MODIFY +2 lines)    | inline parameter swap         | request-response                       | current hardcoded site (cli.py:149-154)                                           | self-reference (bug fix in place) |
| `src/benchmark/engines/mixed_precision.py` — `apply_strategy_b` (MODIFY +2 lines) | TRT layer-precision marker    | graph transform                        | existing predicate in `apply_strategy_b` (mixed_precision.py:60)                  | self-reference (extension to existing function) |

---

## Adapter (`rfdetr_adapter.py`)

**Analog:** `src/benchmark/models/rtdetr_adapter.py` (288 lines, full file in context)
**Secondary reference:** `src/benchmark/models/yolo_adapter.py:1-100` (for the `preprocess()` method ownership pattern — RT-DETR has no `preprocess()`; YOLO does; **RF-DETR will**, per D-RF-04).

### Module Header + Imports — copy from `rtdetr_adapter.py:1-22`

```python
"""RF-DETR ModelAdapter for Roboflow's `rfdetr` package (RFDETRLarge).

Implements the ModelAdapter protocol defined in pytorch_engine.py.
Model: rfdetr.RFDETRLarge (DINOv2 backbone + DETR decoder, 704x704 input, COCO-91 classes).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchvision.transforms.functional as tvf   # NEW (RT-DETR doesn't preprocess in-adapter; RF-DETR does)
from PIL import Image                              # NEW (same reason)
from torch import nn
from rfdetr import RFDETRLarge                     # replaces `from transformers import RTDetrForObjectDetection`

from benchmark.engines.base import Detection

if TYPE_CHECKING:
    from pathlib import Path
    from benchmark.data.coco_loader import COCOSample   # NEW — needed for preprocess() signature

logger = logging.getLogger(__name__)
```

### Module Constants — REPLACE the COCO-80 LUT block, ADD preprocessing constants

RT-DETR carries a 115-line `_COCO80_LUT` array (`rtdetr_adapter.py:24-115`). **RF-DETR does NOT need this** — per RESEARCH § "RF-DETR Model Loading & Inference Path" + `rfdetr/datasets/coco.py:109`, "output slot k corresponds directly to COCO category ID k". The block to copy instead, from the **research adapter sketch (RESEARCH lines 326-330)** and the YOLO `input_size` property pattern (yolo_adapter.py:68-75):

```python
# Module-level constants (D-RF-04, source: rfdetr/detr.py:373-375)
_INPUT_SIZE: tuple[int, int] = (704, 704)         # native RFDETRLarge resolution
_IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]

# RF-DETR ONNX output names (per rfdetr/export/main.py:120): ["dets", "labels"]
# Logits shape: (B, 300, 91). Slot 0 = N/A (no COCO id=0), slots 1..89 = COCO-91 ids, slot 90 = background.
_EXPECTED_NUM_CLASSES: int = 91                    # NOT 80 (unlike RT-DETR)
_NUM_QUERIES: int = 300                            # ModelConfig.num_queries == num_select
_BG_INDEX: int = 90                                # DETR-convention background slot
```

### Class Skeleton — copy from `rtdetr_adapter.py:152-163`

```python
class RFDETRAdapter:
    """ModelAdapter for rfdetr.RFDETRLarge.

    Implements the ModelAdapter protocol (pytorch_engine.py:29-72).
    Handles: weight loading, preprocessing (vendor ImageNet + direct resize),
    inference, output parsing (sigmoid + top-k over flattened (queries × classes)).
    """

    @property
    def input_size(self) -> tuple[int, int]:
        """Model input resolution (height, width)."""
        return _INPUT_SIZE
```

### `load()` — adapt from `rtdetr_adapter.py:164-203` (RF-DETR loader is *different*)

The RT-DETR `load()` calls `RTDetrForObjectDetection.from_pretrained()`. **RF-DETR's `load()` does NOT take `weights_path`** in the usual sense — vendor downloads `rf-detr-large-2026.pth` (~150 MB) into its own cache directory on `RFDETRLarge()` instantiation. From RESEARCH § "Loading" (lines 245-251) + Landmine #7 (model is CPU-by-default until first use):

```python
def load(self, weights_path: Path, device: torch.device) -> nn.Module:
    """Load RF-DETR-L weights and return model ready for inference.

    Parameters
    ----------
    weights_path : Path
        Conventionally `weights/rfdetr-l/`. The rfdetr package manages its own
        checkpoint cache; this directory exists only to keep the project's
        per-model layout uniform with rtdetr-r50vd/, yolo11l/, yolo26l/.
        First call triggers a ~150 MB download (~30-60 s) into vendor cache.
    device : torch.device
        Target device. RFDETRLarge() initialises on CPU by design — adapter
        MUST call `.to(device)` explicitly (vendor's _ensure_model_on_device
        at detr.py:345 only fires on first .predict() call).
    """
    logger.info("Loading RFDETRLarge (weights dir: %s)", weights_path)
    m = RFDETRLarge()                       # downloads rf-detr-large-2026.pth on first call
    nn_model = m.model.model                # unwrap: RFDETR → ModelContext → LWDETR (nn.Module)
    nn_model.eval()
    nn_model = nn_model.to(device)

    # Verify forward shape on a probe — keeps Landmine #8 (output ordering)
    # explicit in logs and catches checkpoint corruption early.
    with torch.no_grad():
        probe = torch.zeros(1, 3, _INPUT_SIZE[0], _INPUT_SIZE[1], device=device)
        probe_out = nn_model(probe)
        logger.info(
            "RF-DETR pred_logits shape: %s  pred_boxes shape: %s",
            tuple(probe_out["pred_logits"].shape),
            tuple(probe_out["pred_boxes"].shape),
        )

    return nn_model
```

### `preprocess()` — NEW method (RT-DETR has no preprocess; RF-DETR mirrors YOLO ownership pattern)

Copy the YOLO preprocess **signature** (`yolo_adapter.py:84-103`) and the RESEARCH adapter sketch **body** (RESEARCH lines 336-345):

```python
def preprocess(
    self, sample: COCOSample, device: torch.device | None = None
) -> torch.Tensor:
    """Direct-resize + ImageNet-normalize to (1, 3, 704, 704) per D-RF-04.

    Vendor pipeline from rfdetr/detr.py:1180-1183:
        F.to_tensor (HWC RGB uint8 → CHW float32 / 255 → [0,1])
        F.resize    (direct stretch to 704×704; NOT letterbox — DINOv2 patch alignment)
        F.normalize (ImageNet mean/std — DINOv2 inherits)
    """
    img = Image.fromarray(sample.image)                          # RGB (project convention)
    t = tvf.to_tensor(img)                                       # (3, H, W) float32 [0,1]
    t = tvf.resize(t, [_INPUT_SIZE[0], _INPUT_SIZE[1]])          # direct stretch (no letterbox)
    t = tvf.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)
    t = t.unsqueeze(0)                                            # (1, 3, 704, 704)
    if device is not None:
        t = t.to(device)
    return t
```

### `infer()` — copy from `rtdetr_adapter.py:205-220`, change **only** the call signature

RT-DETR uses HuggingFace's kwarg-only `model(pixel_values=...)`. RF-DETR's LWDETR accepts a single positional tensor (RESEARCH line 268):

```python
def infer(self, model: nn.Module, inputs: torch.Tensor) -> object:
    """Run forward pass. RF-DETR's LWDETR takes a single positional tensor.

    Parameters
    ----------
    inputs : torch.Tensor
        (1, 3, 704, 704) preprocessed input.

    Returns
    -------
    dict with keys {pred_logits, pred_boxes, aux_outputs, enc_outputs}.
    Only pred_logits (1,300,91) and pred_boxes (1,300,4) are consumed by parse_outputs.
    """
    return model(inputs)            # POSITIONAL, not model(pixel_values=inputs)
```

### `parse_outputs()` — DIVERGES from `rtdetr_adapter.py:222-287` (different math + Landmine #8)

The RT-DETR `parse_outputs` does **per-query argmax**. RF-DETR's `PostProcess.forward` (rfdetr/models/postprocess.py:27-80) does **top-k over flattened (queries × classes)**, decomposes back to `(query_idx, class_idx)`, and filters background. **Use the algorithm verbatim from RESEARCH lines 363-415** (full code body already in RESEARCH; do not re-derive).

**Three structural changes** vs `rtdetr_adapter.py:251-260`:

1. **Output-order detection by shape, not index** (Landmine #8 — RF-DETR ONNX outputs `[dets, labels]` in that order, OPPOSITE of RT-DETR's `[logits, pred_boxes]`):
   ```python
   if isinstance(raw_outputs, (list, tuple)):
       # RT-DETR-safe detect-by-shape: (N, 4) is boxes, (N, 91) is logits.
       a, b = raw_outputs[0][0], raw_outputs[1][0]
       if a.shape[-1] == 4:
           boxes_norm, logits = torch.from_numpy(a), torch.from_numpy(b)
       else:
           logits, boxes_norm = torch.from_numpy(a), torch.from_numpy(b)
   elif isinstance(raw_outputs, dict):
       logits = raw_outputs["pred_logits"][0]
       boxes_norm = raw_outputs["pred_boxes"][0]
   else:
       msg = f"Unsupported raw_outputs type: {type(raw_outputs)}"
       raise TypeError(msg)
   ```

2. **NO COCO-80 LUT** (delete that section entirely): `labels = class_idx[keep].cpu().numpy().astype(np.int64)` is already COCO-91.

3. **Background filter** `(class_idx != 0) & (class_idx != _BG_INDEX)` — drops the no-object slot AND the never-trained id=0 slot.

The xyxy box conversion at the end (`rtdetr_adapter.py:274-281`) and the `Detection(...)` construction (`rtdetr_adapter.py:283-287`) **are reusable verbatim** — same `(cx, cy, w, h) * (orig_w/orig_h)` math, same `.reshape(-1, 4)` safety.

---

## ONNX Export Script (`export_rfdetr_onnx.py`)

**Analog:** `scripts/export_rtdetr_onnx.py` (107 lines, full file in context).

### Strategy — D-RF-02 = vendor `rfdetr.export()` + project `simplify_onnx()`

Per RESEARCH § "D-RF-02 Investigation" (recommendation a), RF-DETR uses the **vendor exporter** (already produces opset=18, clean LayerNormalization nodes, BatchNorm-free check, etc.). Project `simplify_onnx()` is still mandatory (C-10). This means the script is **structurally identical** to `export_rtdetr_onnx.py` but skips the manual `export_to_onnx()` step.

### Module Header + Imports — copy from `export_rtdetr_onnx.py:1-26`, ADAPT

```python
"""Export RF-DETR-Large to ONNX via the vendor `rfdetr.export()` API,
then run the mandatory project simplification + validation (C-10).

WARNING: vendor RFDETR.export() is DESTRUCTIVE on the model object —
LWDETR.export() swaps self.forward = self.forward_export in-place. After this
script runs, the same `m.model.model` instance can NO LONGER be used for
training-mode forward passes. This script instantiates → exports → exits.
The Stage-1 PyTorch baseline (cli.py) instantiates a separate RFDETRLarge();
the two model instances never overlap.

Usage:
    uv run python scripts/export_rfdetr_onnx.py
    uv run python scripts/export_rfdetr_onnx.py --weights-dir weights/rfdetr-l

Outputs:
    weights/rfdetr-l/inference_model.onnx       — vendor ONNX (opset 18)
    weights/rfdetr-l/rfdetr_l_sim.onnx          — onnxsim-simplified ONNX
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from rfdetr import RFDETRLarge

from benchmark.engines.onnx_export import simplify_onnx, validate_onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
```

**DELETE** the `_DYNAMIC_AXES` constant from the RT-DETR script (lines 29-33) — vendor exporter handles its own dynamic axes / opset settings.

### `parse_args()` — copy verbatim from `export_rtdetr_onnx.py:36-52`

Same `--weights-dir` (default `weights/rfdetr-l`) and `--device cuda` flags. Same argparse style.

### `main()` — adapt from `export_rtdetr_onnx.py:55-102`

```python
def main() -> int:
    """Export, simplify, and validate RF-DETR-Large ONNX."""
    args = parse_args()
    args.weights_dir.mkdir(parents=True, exist_ok=True)   # vendor downloads .pth here

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Step 0: Instantiate (downloads ~150 MB rf-detr-large-2026.pth on first run)
    logger.info("Instantiating RFDETRLarge (vendor downloads weights on first call)")
    m = RFDETRLarge()

    # Step 1: Vendor ONNX export at opset=18, shape=(704,704). DESTRUCTIVE — see docstring.
    raw_onnx = args.weights_dir / "inference_model.onnx"
    sim_onnx = args.weights_dir / "rfdetr_l_sim.onnx"
    logger.info("Exporting via vendor m.export() → %s", raw_onnx)
    m.export(opset_version=18, shape=(704, 704), output_dir=str(args.weights_dir))

    # Step 2: Mandatory project simplification (C-10)
    simplify_onnx(raw_onnx, output_path=sim_onnx)

    # Step 3: Validate simplified model
    validate_onnx(sim_onnx)

    logger.info("Export complete:")
    logger.info("  Raw:        %s (%.1f MB)", raw_onnx, raw_onnx.stat().st_size / 1024 / 1024)
    logger.info("  Simplified: %s (%.1f MB)", sim_onnx, sim_onnx.stat().st_size / 1024 / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Expected output sizes** (verified in RESEARCH § "ONNX Graph Inspection"): raw 123.2 MB → simplified 120.1 MB; 918 nodes; 51 LayerNormalization; 20 Softmax.

---

## CLI Integration

### `MODEL_REGISTRY` extension — `src/benchmark/cli.py:73-89`

**Exact line range to touch:** insert a new dict entry **after line 88** (closing brace of `yolo26l` entry), **before line 89** (the closing brace of `MODEL_REGISTRY`).

**Pattern to copy** — the existing `yolo26l` block at cli.py:84-88:

```python
    "yolo26l": {
        "weights": "weights/yolo26l/yolo26l.pt",
        "onnx": "weights/yolo26l/yolo26l_sim.onnx",
        "family": "yolo",
    },
```

**New entry to add** (per RESEARCH lines 510-521):

```python
    "rfdetr-l": {
        "weights": "weights/rfdetr-l/",
        "onnx": "weights/rfdetr-l/rfdetr_l_sim.onnx",
        "family": "rfdetr",
    },
```

Note `"weights"` is a **directory** (matches `"rt-detr"`'s `"weights/rtdetr-r50vd/"`), not a `.pt` file like YOLO — vendor manages the checkpoint inside that dir.

### `_get_adapter` extension — `src/benchmark/cli.py:101-125`

**Exact line range to touch:** insert a new `if`-branch **between line 122** (closing `)` of YOLO `return YOLOAdapter(...)`) **and line 124** (the fallthrough `msg = ...`).

**Pattern to copy** — the existing YOLO branch at cli.py:118-122 (cleanest lazy-import style):

```python
    if model_name in ("yolo11l", "yolo26l"):
        from benchmark.models.yolo_adapter import YOLOAdapter  # noqa: PLC0415

        is_nms_free = model_name == "yolo26l"
        return YOLOAdapter(is_nms_free=is_nms_free)
```

**New branch to add** (per RESEARCH lines 525-537):

```python
    if model_name == "rfdetr-l":
        from benchmark.models.rfdetr_adapter import RFDETRAdapter  # noqa: PLC0415

        return RFDETRAdapter()
```

### Stage-1 `compute_macs` fix — `src/benchmark/cli.py:143-154`

**Exact line range to touch:** the `compute_macs` call at lines 149-154 (in the `if stage == "1_pytorch_fp32":` branch).

**Current code (line 149-154):**

```python
        if macs is None:
            macs, flops = compute_macs(
                engine.model,
                model_name,
                input_shape=(1, 3, 640, 640),       # ← HARDCODED (wrong for RF-DETR @ 704)
            )
```

**Modified code** (per RESEARCH lines 540-549):

```python
        if macs is None:
            h, w = adapter.input_size               # adapter is already in scope (cli.py:141)
            macs, flops = compute_macs(
                engine.model,
                model_name,
                input_shape=(1, 3, h, w),
            )
```

The `adapter` local variable is already in scope from cli.py:141 (`adapter = _get_adapter(model_name)`). **No new imports needed.** The cli.py:188 `OnnxRuntimeEngine(... input_size=(640, 640))` call site is **NOT** touched (RESEARCH line 551 — the field is stored-but-not-read; preprocess is adapter-driven).

---

## Mixed Precision Extension

### `apply_strategy_b` — `src/benchmark/engines/mixed_precision.py:50-65`

**Exact line to touch:** line **60** — the `if` predicate inside the `for` loop.

**Current code (line 60):**

```python
        if layer.type == trt.LayerType.SOFTMAX or "norm" in layer.name.lower():
```

**Modified code** (D-RF-03 = B2, per RESEARCH lines 177-209):

```python
        if (
            layer.type == trt.LayerType.SOFTMAX
            or layer.type == trt.LayerType.NORMALIZATION   # NEW — catches INormalizationLayer (TRT 8.6+)
            or "norm" in layer.name.lower()
        ):
```

**Why the belt-and-braces approach** (RESEARCH line 446): the existing `"norm"` substring check already fires on all 51 LayerNormalization nodes because PyTorch opset=18 emits LayerNorm as a single `onnx::LayerNormalization` node and the name string contains `LayerNormalization`. The new `LayerType.NORMALIZATION` clause is hardening for future graphs (D-FINE / DEIMv2 in Phase 10) where the layer might be named differently but still be an `INormalizationLayer`. **Decomposed-LayerNorm pitfall is verified NOT PRESENT** for RF-DETR (RESEARCH § ONNX Graph Inspection: 0 ReduceMean, 0 Pow after `simplify_onnx()`).

**No other changes** to `mixed_precision.py`. `apply_strategy_a` (lines 14-48) is architecture-agnostic and untouched. `is_constant_or_shape` helper (lines 10-12) unchanged.

---

## Cross-cutting Conventions (Phase 7 carry-forward)

These conventions apply to all RF-DETR files and are validated by the Phase 6 + 7 production run. Pulled from CONTEXT § "Carried Forward From Phase 7":

| Convention                        | Value / Pattern                                                       | Source                                                                  |
|-----------------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------|
| Adapter file location             | `src/benchmark/models/<model>_adapter.py`                             | `rtdetr_adapter.py`, `yolo_adapter.py`                                  |
| Adapter class naming              | `<MODEL>Adapter` (PascalCase, acronym uppercase)                      | `RTDETRAdapter`, `YOLOAdapter` → `RFDETRAdapter`                        |
| Adapter Protocol methods          | `input_size` (property), `load`, `preprocess` (optional), `infer`, `parse_outputs` | `pytorch_engine.py:29-72` Protocol definition                           |
| ONNX export script location       | `scripts/export_<model>_onnx.py`                                      | `scripts/export_rtdetr_onnx.py`                                         |
| ONNX simplification (C-10)        | **Mandatory** `simplify_onnx()` after any export, vendor or in-project | `onnx_export.py`                                                        |
| ONNX opset for transformers       | `opset_version=18` (project default in `onnx_export.py`)              | `export_rtdetr_onnx.py:86`                                              |
| Weights directory layout          | `weights/<model>/...` (one dir per model)                             | `weights/rtdetr-r50vd/`, `weights/yolo11l/`, `weights/yolo26l/` → `weights/rfdetr-l/` |
| TRT engine layout                 | `engines/<model>/<model>_<precision>.engine`                          | Phase 7 (07-01-PLAN.md) → `engines/rfdetr-l/rfdetr-l_tf32.engine` etc. |
| INT8 cache layout                 | `engines/<model>/cal_<method>.cache`                                  | Phase 7 → `engines/rfdetr-l/cal_minmax.cache` etc.                      |
| Calibration set                   | **500 COCO val2017 images** via `_build_calibration_dataloader()` — `COCODataLoader(limit=500)`, deterministic sorted order; **no seed needed** | `cli.py:43-54`                                                          |
| Batch size                        | Strictly **1** (`run_full_benchmark` loop processes one `COCOSample` at a time) | `base.py` constants `WARMUP_RUNS=50`, `MEASURE_RUNS=1000`               |
| TRT workspace                     | Strictly **2 GB** (`config.set_memory_pool_limit(MemoryPoolType.WORKSPACE, 2 << 30)`) | `tensorrt_engine.py` (C-02)                                             |
| BF16 verification                 | Check `builder.platform_has_tf32` (Ampere sm_80+ proxy) before setting `trt.BuilderFlag.BF16` | C-04                                                                    |
| Logger usage                      | Module-level `logger = logging.getLogger(__name__)` + `%s` formatting (lazy) | All existing modules                                                    |
| Error pattern                     | `msg = "..."` + `raise <Error>(msg) from exc` (ruff EM101)            | `rtdetr_adapter.py`-style raises; `cli.py:113-115`                      |
| Type annotations                  | `from __future__ import annotations` in every module; `X \| None` unions; `tuple[int, int]` (not `Tuple`) | `rtdetr_adapter.py:7`                                                   |
| TYPE_CHECKING guard               | `if TYPE_CHECKING: from pathlib import Path`                          | `rtdetr_adapter.py:19-20`                                               |
| Lazy import in `_get_adapter`     | `from ... import X  # noqa: PLC0415` inside the if-branch             | `cli.py:112, 119`                                                       |
| Output handling                   | All metrics via `ResultLogger` → per-stage CSV/JSON + unified `results.csv`/`results.json` (`model_name`, `stage` columns) | `utils/logger.py` (C-09)                                                |

---

## Landmines from RESEARCH.md

**Planner MUST surface these as `must_haves.truths` in the relevant plan files** so the executor encodes them in code or docstrings. Verbatim copy from RESEARCH § "Open Risks & Landmines":

1. **Vendor `RFDETR.export()` is destructive on the model object.** `LWDETR.export()` swaps `self.forward = self.forward_export` in-place. After running the export script, the same `m.model.model` instance can NO LONGER be used for training-mode forward passes. **Mitigation:** the export script instantiates → exports → exits. The Stage-1 PyTorch baseline path instantiates a fresh `RFDETRLarge()` separately. They never share a model instance. **Document this in the export script's docstring.**

2. **Vendor `predict()` class_name display is buggy** (`class_names[class_id]` is off-by-one for COCO pretraining because COCO_CLASS_NAMES has only 80 entries while class_id is COCO-91). The *integer* class_id is correct — only the display label is wrong. The adapter never calls `predict()`; it calls the raw model and parses outputs directly. **The vendor bug does not affect our adapter.** Surfacing here so a future reader doesn't get distracted by the warning logs.

3. **Vendor logs PE / patch-size warnings on `RFDETRLarge()` instantiation** ("Using a different number of positional encodings than DINOv2…"). These are EXPECTED — RF-DETR is fine-tuned from DINOv2 with adapted PE and patch_size. The pretrained `rf-detr-large-2026.pth` checkpoint contains the adapted weights; the warning is purely about the underlying DINOv2 backbone weights NOT being loaded (because they would conflict with RF-DETR's adaptations). **Suppress in the adapter logging if noisy; do NOT treat as an error.**

4. **Activation distribution unknown for INT8.** Phase 7 found Entropy catastrophic on YOLO CNN heads but fine on RT-DETR transformer attention. RF-DETR is closer to RT-DETR (transformer), so Entropy is *expected* to be competitive — but this is a hypothesis, not measurement. The per-model `save_int8_best_calibrator(model)` automatically picks the winner; the plan task should not pre-judge.

5. **`compute_macs` for transformer with GridSample.** The project's `compute_macs` helper uses `thop` (or similar) under the hood. `thop` can fail to count FLOPs for ops it doesn't know (GridSample, custom attention). The result might be a partial undercounting. **Mitigation:** if MACs come back suspicious (very low for a 33.9M-param model), log it but proceed — MACs is a secondary diploma metric, not a verification gate.

6. **Vendor weight download is a network-dependent side effect of `RFDETRLarge()` instantiation.** First-run takes ~30-60 s for the 150 MB download. **Mitigation:** the planner should ensure Wave 0 or first Stage 1 task explicitly notes "first run downloads weights"; subsequent runs hit the vendor's local cache (verified: `File rf-detr-large-2026.pth already exists with correct MD5 hash.`).

7. **`m.model.device` is `cuda` by default but the model itself is on CPU until first use.** `_ensure_model_on_device` (detr.py:345) does the deferred placement. The Stage 1 adapter must call `nn_model.to(device)` explicitly (the existing adapter pattern already does this, just don't assume the model is on GPU after `RFDETRLarge()`).

8. **The exported ONNX has `dets` and `labels` in that ORDER (not labels first like RT-DETR).** RT-DETR's ONNX outputs `[logits, pred_boxes]`. RF-DETR's outputs `[dets (=pred_boxes), labels (=pred_logits)]`. **The adapter's `parse_outputs` MUST detect output order from the list, not assume RT-DETR ordering.** The contract is documented in vendor `export/main.py:120`. The recommended pattern: detect by **shape**, not by index — `(N, 4)` is boxes, `(N, 91)` is logits. Then there's no ambiguity.

---

## Metadata

**Analog search scope:** `src/benchmark/models/`, `src/benchmark/engines/`, `src/benchmark/cli.py`, `scripts/`
**Files scanned (full reads):** `rtdetr_adapter.py` (288 lines), `export_rtdetr_onnx.py` (107 lines), `mixed_precision.py` (66 lines), `cli.py:1-200`, `yolo_adapter.py:1-100`
**Pattern extraction date:** 2026-05-16
**Researcher coverage:** RESEARCH § "Files to Create / Modify" already enumerates the file list — this PATTERNS map extracts the concrete code excerpts the planner can hand to the executor.
