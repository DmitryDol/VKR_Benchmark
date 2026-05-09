# Requirements: VKR Benchmark

**Defined:** 2026-05-09
**Core Value:** Scientifically rigorous, per-stage metric logging showing optimization evolution for transformer-based object detectors.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Model Adapters

- [ ] **ADPT-01**: System can load RT-DETR pretrained weights and run inference
- [ ] **ADPT-02**: System can parse RT-DETR outputs into Detection format (boxes, scores, labels)
- [ ] **ADPT-03**: System can download and manage pretrained weights for RT-DETR

### ONNX Pipeline

- [ ] **ONNX-01**: System can export any loaded model to ONNX format with configurable opset
- [ ] **ONNX-02**: System applies onnx-simplifier to exported models automatically
- [ ] **ONNX-03**: System validates ONNX model integrity after export and simplification

### TensorRT Engines

- [ ] **TRT-01**: System can build TensorRT engine from ONNX with TF32 precision flag
- [ ] **TRT-02**: System can build TensorRT engine with FP16 precision
- [ ] **TRT-03**: System can build TensorRT engine with BF16 precision after verifying hardware support
- [ ] **TRT-04**: System enforces 2 GB workspace memory limit for all TensorRT builds
- [ ] **TRT-05**: System can run inference using TensorRT engines with proper CUDA memory management

### INT8 Calibration

- [ ] **CAL-01**: System implements MinMax calibration for INT8 quantization
- [ ] **CAL-02**: System implements Entropy calibration for INT8 quantization
- [ ] **CAL-03**: System implements Percentile calibration for INT8 quantization
- [ ] **CAL-04**: System uses COCO val2017 images as calibration dataset
- [ ] **CAL-05**: System can build INT8 TensorRT engine using any of the three calibrators

### Mixed Precision

- [ ] **MIX-01**: System can build Mixed Precision engine with Strategy A (first/last layers FP16, rest INT8)
- [ ] **MIX-02**: System can build Mixed Precision engine with Strategy B (Softmax/LayerNorm layers FP16, rest INT8)
- [ ] **MIX-03**: System uses the best INT8 calibrator from Stage 5 for mixed precision builds

### Metrics & Logging

- [ ] **LOG-01**: System logs latency split: pre-processing, inference, post-processing (ms)
- [ ] **LOG-02**: System logs throughput (FPS) computed from total latency
- [ ] **LOG-03**: System logs jitter as standard deviation of inference time (ms)
- [ ] **LOG-04**: System logs mAP_50 and mAP_50:95 via COCO evaluation API
- [ ] **LOG-05**: System logs accuracy drop (%) relative to FP32 baseline
- [ ] **LOG-06**: System logs IoU metrics from COCO evaluation
- [ ] **LOG-07**: System logs model size (MB) for each engine format
- [ ] **LOG-08**: System logs peak VRAM usage (MB) strictly via torch.cuda.max_memory_allocated
- [ ] **LOG-09**: System logs MACs and FLOPs for each model
- [ ] **LOG-10**: System generates per-stage CSV and JSON files (one per model per stage)
- [ ] **LOG-11**: System generates unified results CSV/JSON with stage column
- [ ] **LOG-12**: System logs hardware info (GPU name, driver version, CUDA version, TensorRT version)

### CLI Interface

- [ ] **CLI-01**: User can run a single model through a specific stage via CLI
- [ ] **CLI-02**: User can run a single model through all 6 stages sequentially via CLI
- [ ] **CLI-03**: User can export/generate unified results from per-stage files via CLI

### Benchmarking Protocol

- [ ] **BENCH-01**: System performs 50 warm-up iterations before measurement
- [ ] **BENCH-02**: System averages metrics over 1000 measured iterations
- [ ] **BENCH-03**: System uses batch size 1 for all inference
- [ ] **BENCH-04**: System disables TF32 for PyTorch FP32 baseline
- [ ] **BENCH-05**: System resets VRAM tracking and clears CUDA cache between engine runs
- [ ] **BENCH-06**: System synchronizes CUDA between timing measurement points

### Bug Fixes

- [ ] **FIX-01**: Fix double infer() call in warm-up loop (base.py:96-97)

## v2 Requirements

Deferred to future work. Not in current roadmap.

### Additional Model Adapters

- **ADPT-04**: RF-DETR adapter (load weights + parse outputs)
- **ADPT-05**: D-FINE adapter (load weights + parse outputs)
- **ADPT-06**: DEIMv2 adapter (load weights + parse outputs)
- **ADPT-07**: YOLO11 adapter (load weights + parse outputs via Ultralytics)
- **ADPT-08**: YOLO26 adapter (load weights + parse outputs via Ultralytics)

### Extended Models

- **EXT-01**: Additional model variants (e.g., RT-DETR-L vs RT-DETR-X)

### Advanced Features

- **ADV-01**: Strategy C — Sensitivity Analysis for mixed precision (per-layer profiling)
- **ADV-02**: Batch mode CLI to run all models through all stages in one command
- **ADV-03**: Automated graph/chart generation from results
- **ADV-04**: Comparison tables in LaTeX format for diploma

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-GPU support | Single RTX 3070 per spec |
| Model training/fine-tuning | Inference benchmarking only |
| Real-time video pipeline | Static image evaluation sufficient |
| Web UI / dashboard | CSV/JSON output for manual graph creation |
| Multi-batch inference | Batch size strictly 1 per protocol |
| Mobile/edge deployment | Desktop GPU benchmarking only |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| *(populated by roadmapper)* | | |

**Coverage:**
- v1 requirements: 30 total
- Mapped to phases: 0
- Unmapped: 30

---
*Requirements defined: 2026-05-09*
*Last updated: 2026-05-09 after initial definition*
