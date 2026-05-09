# Phase 1 Context: RT-DETR Adapter & ONNX Pipeline

**Phase:** 1 of 5 — RT-DETR Adapter & ONNX Pipeline
**Goal:** RT-DETR FP32 baseline runs correctly end-to-end and produces a verified ONNX model
**Status:** Ready for planning

---

## Decisions

### RT-DETR Source Library
**Decision:** HuggingFace `transformers.RTDetrForObjectDetection`

**Rationale:** `pixel_values: (1,3,H,W)` tensor input matches `PyTorchEngine.infer()` directly without wrapping. ONNX export is cleaner than torchvision detection models. Research-standard implementation.

**Model ID:** `PekingU/rtdetr-r50`

**Implications for implementation:**
- `ModelAdapter.load()` calls `RTDetrForObjectDetection.from_pretrained(weights_path)`
- `PyTorchEngine.preprocess()` produces `pixel_values` tensor `(1, 3, 640, 640)`
- `ModelAdapter.parse_outputs()` receives `RTDetrObjectDetectionOutput`:
  - `logits`: `(1, 300, 92)` → apply sigmoid → threshold at `score_threshold`
  - `pred_boxes`: `(1, 300, 4)` cx,cy,w,h normalized → convert to x1y1x2y2 in pixel coords
  - Class IDs are COCO 91-class (compatible with existing `COCO_91_TO_80` mapping)
- ONNX export: wrap model to accept positional tensor input (HF models use kwargs)
- Add `transformers` and `huggingface_hub` to project dependencies

### Model Variant
**Decision:** `PekingU/rtdetr-r50` — ResNet-50 backbone

**Rationale:** 42M params, ~280MB, mAP 53.1 COCO. Comfortable VRAM budget (~2.5GB FP32 inference) leaves headroom for TRT INT8 calibration buffers in later phases. RTX 3070 8GB safe.

**Input size:** 640×640 (fixed, not dynamic)

### Weight Management (ADPT-03)
**Decision:** Download script → local `weights/rtdetr-r50/` directory

**Rationale:** Reproducible offline benchmarking. Mirrors existing `data/download_coco.py` pattern.

**Implementation:**
- New script: `data/download_weights.py` (or `scripts/download_weights.py`)
- Uses `huggingface_hub.snapshot_download("PekingU/rtdetr-r50", local_dir="weights/rtdetr-r50")`
- `ModelAdapter.load(weights_path: Path)` uses `from_pretrained(str(weights_path))`
- `weights/` directory goes in `.gitignore`

### Bug Fix (FIX-01) — First Task
**Decision:** Fix double `infer()` call in warm-up loop before any other work.

**Location:** `src/benchmark/engines/base.py:96-97`

**Fix:** Store result before postprocessing:
```python
# BEFORE (buggy):
self.infer(inputs)
self.postprocess(self.infer(inputs), sample)

# AFTER (correct):
raw = self.infer(inputs)
self.postprocess(raw, sample)
```

---

## Architecture Decisions (Claude's Discretion)

These are implementation details that do not require user input:

- **Adapter file location:** `src/benchmark/models/rtdetr_adapter.py` (new `models/` package)
- **Package init:** `src/benchmark/models/__init__.py` exporting `RTDETRAdapter`
- **Score threshold:** Keep 0.01 (existing default in `PyTorchEngine`)
- **Preprocessing:** Resize to 640×640 + ImageNet normalization (existing constants in `pytorch_engine.py`)
- **ONNX export:** Wrap HF model in a thin `nn.Module` that accepts positional tensor → use existing `export_and_simplify()` with `input_size=(640,640)`
- **Input/output names:** `pixel_values` → `logits` + `pred_boxes` (update `export_to_onnx` output names for RT-DETR)

---

## Scope Boundary

**In Phase 1:**
- Fix FIX-01 (double infer bug)
- `RTDETRAdapter` implementing `ModelAdapter` protocol
- `download_weights.py` script
- FP32 baseline runs end-to-end (ADPT-01, ADPT-02, ADPT-03)
- ONNX export + simplification + validation (ONNX-01, ONNX-02, ONNX-03)
- BENCH-01 through BENCH-06 (already mostly implemented in BaseEngine; FIX-01 covers the gap)

**Deferred to Phase 2:**
- Metrics/logging expansion (LOG-*)
- CLI interface (CLI-*)
- OnnxRuntimeEngine (ONNX inference, not just export)

**Deferred to v2:**
- RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26 adapters (ADPT-04 through ADPT-08)
- Strategy C / Sensitivity Analysis (ADV-01)

---

## Canonical Refs

| Ref | Path | Purpose |
|-----|------|---------|
| Requirements | `.planning/REQUIREMENTS.md` | Phase 1 reqs: FIX-01, ADPT-01–03, ONNX-01–03, BENCH-01–06 |
| Roadmap | `.planning/ROADMAP.md` | Phase 1 success criteria |
| Base engine | `src/benchmark/engines/base.py` | FIX-01 target, BaseEngine abstract interface |
| PyTorch engine | `src/benchmark/engines/pytorch_engine.py` | ModelAdapter protocol, PyTorchEngine |
| ONNX export | `src/benchmark/engines/onnx_export.py` | export_to_onnx, simplify_onnx, export_and_simplify |
| COCO loader | `src/benchmark/data/coco_loader.py` | COCOSample, COCO_91_TO_80 mapping |
| COCO download | `data/download_coco.py` | Pattern for download_weights.py |
| HF model | `PekingU/rtdetr-r50` (HuggingFace Hub) | Pre-trained weights source |

---

## Code Context (Reusable Assets)

| Asset | Location | Reuse |
|-------|----------|-------|
| `ModelAdapter` Protocol | `pytorch_engine.py:29-72` | Implement directly — `input_size`, `load()`, `parse_outputs()` |
| `COCO_91_TO_80` mapping | `coco_loader.py` | Use in `parse_outputs()` for class ID conversion |
| `IMAGENET_MEAN/STD` | `pytorch_engine.py:23-24` | Use in preprocessing |
| `export_and_simplify()` | `onnx_export.py` | Call with wrapped HF model |
| `BaseEngine.benchmark_latency()` | `base.py:80+` | Unchanged after FIX-01 |
| `.gitignore` | root | Add `weights/` entry |
