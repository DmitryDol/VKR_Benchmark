# Phase 07: YOLO Family Quantization - Research

**Researched:** 2026-05-12
**Domain:** YOLO11/YOLO26 Quantization & TensorRT Optimization
**Confidence:** HIGH

## Summary

This research focuses on the quantization of the YOLO family (specifically YOLO11l and YOLO26l) for TensorRT. YOLO11l is a mature model using Distribution Focal Loss (DFL) and NMS post-processing, making it sensitive to INT8 quantization in its attention (C2PSA) and detection head (DFL) layers. YOLO26l is a next-generation "edge-first" model that is natively NMS-free and has removed DFL, significantly improving its robustness to quantization and reducing inference latency variance.

**Primary recommendation:** For YOLO11l, prioritize Mixed Precision (Strategy B: Softmax/LayerNorm in FP16) to mitigate DFL and Attention sensitivity. For YOLO26l, standard INT8 quantization is expected to yield <0.5% mAP drop with superior speedup due to its streamlined architecture.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| ONNX Export | API / Backend | — | Uses PyTorch to export the model graph. |
| INT8 Calibration | API / Backend | — | Requires GPU for TensorRT calibration passes. |
| Inference Execution | API / Backend | — | GPU-accelerated inference via TensorRT engines. |
| Post-processing | API / Backend | — | Scales boxes and maps IDs (NMS for YOLO11, simple thresholding for YOLO26). |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ultralytics | 8.4.48 | Model loading & weights | Industry standard for YOLO models. [VERIFIED: uv pip show] |
| tensorrt | 10.16.1.11 | Optimized inference | Hardware-specific optimization for NVIDIA GPUs. [VERIFIED: uv pip show] |
| onnx | 1.21.0 | Interchange format | Standard intermediate format for TensorRT. [VERIFIED: uv pip show] |
| onnxsim | 0.6.3 | Graph simplification | Mandatory for reducing complex YOLO graphs before TRT build. [CITED: GEMINI.md] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pycocotools | 2.0.11 | mAP evaluation | Standard for COCO dataset metrics. [VERIFIED: uv pip show] |

## Architecture Patterns

### YOLO Output Structures

| Model | Output Shape | Post-processing | Type |
|-------|--------------|-----------------|------|
| **YOLO11l** | `(1, 84, 8400)` | NMS required | Anchor-free (DFL) |
| **YOLO26l** | `(1, 300, 6)` | Thresholding only | NMS-free (End-to-End) |

### Recommended Project Structure
```
src/benchmark/
├── models/
│   └── yolo_adapter.py      # Handles both YOLO11 and YOLO26 parsing
├── engines/
│   ├── tensorrt_engine.py   # Core TRT build/inference logic
│   ├── int8_calibrators.py  # MinMax, Entropy, Percentile implementations
│   └── mixed_precision.py   # Strategy A/B/C logic
```

### Pattern: NMS-Free Post-processing (YOLO26)
**What:** Directly parsing fixed-length detection tensors without NMS.
**When to use:** For YOLO26, YOLOv10, and other end-to-end models.
**Example:**
```python
# From src/benchmark/models/yolo_adapter.py
# raw_outputs shape: (1, 300, 6) -> [x1, y1, x2, y2, conf, cls]
mask = preds[:, 4] > score_threshold
results = preds[mask]
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| INT8 Calibration | Custom histogram logic | `trt.IInt8Calibrator` | Built-in calibrators are highly optimized and integrated with the builder. |
| Graph Simplification | Custom node removal | `onnxsim` | Handles constant folding and fusion specifically for TRT. |
| NMS | Custom IoU loops | `ultralytics.utils.nms` | Optimized PyTorch/C++ implementation of NMS. |

## Common Pitfalls

### Pitfall 1: YOLO11 DFL Sensitivity
**What goes wrong:** Huge mAP drop (>5%) in INT8.
**Why it happens:** Distribution Focal Loss relies on a 16-bin Softmax that is highly sensitive to 8-bit quantization.
**How to avoid:** Use Mixed Precision (Strategy B) to keep the Detection Head in FP16. [VERIFIED: web search]

### Pitfall 2: YOLO26 Output Mismatch
**What goes wrong:** Post-processing failure or "No detections" output.
**Why it happens:** Assuming YOLO26 requires NMS or has the same `(1, 84, 8400)` shape as YOLO11.
**How to avoid:** Use `is_nms_free=True` in `YOLOAdapter` to toggle the correct parser. [VERIFIED: code audit]

### Pitfall 3: TensorRT 10 Calibrator Initialization
**What goes wrong:** "Tried to call pure virtual function" crash.
**Why it happens:** `IInt8LegacyCalibrator` (used for Percentile) requires overriding `read_histogram_cache` and `write_histogram_cache`.
**How to avoid:** Implement dummy overrides returning `None` as seen in `src/benchmark/engines/int8_calibrators.py`. [VERIFIED: code audit]

## Code Examples

### Calibrator Selection (Direct API)
```python
# Source: src/benchmark/engines/int8_calibrators.py
if method == "minmax":
    return MinMaxCalibrator(data, cache_path)
elif method == "entropy":
    return EntropyCalibrator(data, cache_path)
elif method == "percentile":
    return PercentileCalibrator(data, cache_path) # Uses IInt8LegacyCalibrator
```

### BF16 Hardware Verification
```python
# Source: GEMINI.md requirement
# BF16 support check via TF32 availability (Ampere+ proxy)
if builder.platform_has_tf32:
    config.set_flag(trt.BuilderFlag.BF16)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| NMS Post-processing | NMS-Free (E2E) | YOLO26 / YOLOv10 | Zero CPU bottleneck for post-processing. |
| DFL (Distribution) | Simplified Heads | YOLO26 | Better quantization robustness. |
| FP16 Only | BF16 (Ampere+) | TRT 9.x+ | Better range, same speed as FP16 on RTX 3070. |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | YOLO26l corresponds to the Ultralytics 8.4.x YOLO26 version. | Summary | Minimal, since the weights and adapter logic match. |
| A2 | "Percentile" calibration is best implemented via `IInt8LegacyCalibrator` with `quantile`. | Standard Stack | Standard practice in TRT community. |

## Open Questions

1. **RESOLVED: Exact Layer Names for Strategy B/C**: Identifying exact layer strings is unnecessary as `mixed_precision.py` uses heuristic-based selection (`LayerType.SOFTMAX` and "norm" naming) which covers the identified sensitive DFL and Attention layers.
2. **RESOLVED: Optimal Calibration Image Count**: 500 images is the project-standard limit for calibration (D-06), which matches community recommendations for YOLO.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| RTX 3070 | All stages | ✓ | sm_86 | — |
| TensorRT | Engines | ✓ | 10.16.1 | — |
| ultralytics | Loading/Export | ✓ | 8.4.48 | — |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Quick run command | `pytest tests/test_yolo_adapters.py` |
| Full suite command | `pytest tests/` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| YOLO-Q-01 | ONNX Export (YOLO11/26) | unit | `pytest tests/test_onnx_export.py` | ✅ |
| YOLO-Q-02 | TRT Build (INT8) | unit | `pytest tests/test_tensorrt_engine.py` | ✅ |
| YOLO-Q-03 | NMS-Free Parsing | unit | `pytest tests/test_yolo_adapters.py` | ✅ |

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | `torchvision.transforms.functional` handles image tensor sanity. |

## Sources

### Primary (HIGH confidence)
- `/ultralytics/ultralytics` - Official YOLO11/YOLO26 documentation.
- `/nvidia/tensorrt` - TensorRT 10.x Python API documentation.
- `src/benchmark/engines/int8_calibrators.py` - Local implementation verification.

### Secondary (MEDIUM confidence)
- WebSearch for YOLO26 release features and quantization drop benchmarks.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Verified via local env and docs.
- Architecture: HIGH - Verified via code audit and model probing.
- Pitfalls: MEDIUM - Based on community benchmarks for YOLO11 DFL.

**Research date:** 2026-05-12
**Valid until:** 2026-06-11
