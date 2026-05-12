# VKR Benchmark — Transformer Object Detection Optimization Pipeline

## What This Is

A production-ready benchmarking framework that conducts six SOTA transformer-based object detectors (RT-DETR, RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26) through a six-stage hardware optimization pipeline on an NVIDIA RTX 3070, logging detailed metrics at every stage. Built for an academic diploma to demonstrate the evolution of inference performance (latency, throughput, mAP, VRAM) from pure PyTorch FP32 through TensorRT INT8 and Mixed Precision quantization.

## Core Value

Scientifically rigorous, per-stage metric logging that produces publication-ready CSV/JSON reports showing how each optimization stage affects every metric — no intermediate results lost.

## Current Milestone: v2.0 v2.0_models_integration

**Goal:** Horizontally scale the optimization pipeline to support RF-DETR, D-FINE, DEIMv2, YOLO11, and YOLO26 by implementing the ModelAdapter protocol and a batch orchestration CLI.

**Target features:**
- Implement ModelAdapter Protocol
- Integrate Transformer-based Family (RF-DETR, D-FINE, DEIMv2)
- Integrate YOLO Family (YOLO11, YOLO26) with efficient NMS
- Batch Orchestration CLI (`run-all`) generating global .csv and summary.md

## Requirements

### Validated

- ✓ COCO val2017 data loading with single-image iteration (batch=1) — existing
- ✓ Abstract BaseEngine with benchmarking protocol (50 warm-up, 1000 measured runs, GPU sync) — existing
- ✓ PyTorch FP32 baseline engine with TF32 explicitly disabled — existing
- ✓ ONNX export with onnx-simplifier optimization — existing
- ✓ BenchmarkResult structured logging to CSV and JSON — existing
- ✓ VRAM peak tracking via max_memory_allocated with cache clearing — existing
- ✓ ModelAdapter protocol for model-specific loading and output parsing — existing (interface only)
- ✓ Latency split measurement (preprocess + inference + postprocess) — existing
- ✓ COCO mAP evaluation (mAP_50, mAP_50:95) via pycocotools — existing
- ✓ RT-DETR Model adapter (ADPT-01, ADPT-02) — v1.0
- ✓ Weight downloading and management for RT-DETR (ADPT-03) — v1.0
- ✓ TensorRT engine builder with TF32 precision flag (TRT-01) — v1.0
- ✓ TensorRT FP16 and BF16 engine builds (TRT-02, TRT-03) — v1.0
- ✓ TensorRT INT8 with MinMax, Entropy, and Percentile calibrators (CAL-01..CAL-05) — v1.0
- ✓ Mixed Precision quantization (Strategy A, Strategy B) (MIX-01..MIX-03) — v1.0
- ✓ Per-stage CSV/JSON output files (LOG-10) — v1.0
- ✓ Unified results file with stage column for cross-stage analysis (LOG-11) — v1.0
- ✓ CLI interface (typer) for single-model and batch execution (CLI-01..CLI-03) — v1.0
- ✓ MACs/FLOPs computation per model (LOG-09) — v1.0
- ✓ Hardware info logging (GPU name, driver, CUDA version, TensorRT version) (LOG-12) — v1.0
- ✓ TensorRT workspace memory limit enforcement (2 GB) (TRT-04) — v1.0

### Active

- [ ] Model adapters for remaining 5 architectures (RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26) (ADPT-04..ADPT-08)
- [ ] Strategy C (Sensitivity Analysis)
- [ ] Automated Output generation (LaTeX tables, auto-charts)

### Out of Scope

- Strategy C (Sensitivity Analysis) — optional bonus, only if time permits after core pipeline is complete
- Multi-GPU support — single RTX 3070 per spec
- Training or fine-tuning — inference benchmarking only
- Real-time video pipeline — static image evaluation only
- Web UI or dashboard — CSV/JSON output is sufficient for diploma graphs
- Multi-batch inference — batch size strictly 1

## Context

**Academic context:** Diploma thesis at SSAU (Samara State Aerospace University). The thesis demonstrates how hardware optimization techniques progressively affect inference quality and speed for transformer-based object detectors. Each chapter of the diploma maps to optimization stages, requiring per-stage metric reports.

**Existing codebase:** ~10 Python source files implementing the data loading, base engine abstraction, PyTorch FP32 engine, ONNX export pipeline, and result logging. The architecture is modular and well-typed. No model adapters, TensorRT code, CLI, or tests exist yet.

**Known bug:** Double `infer()` call in warm-up loop (`base.py:96-97`) — needs fixing.

**Hardware:** NVIDIA RTX 3070 (Ampere, sm_86, 8 GB VRAM). All experiments on this single GPU.

**Data:** COCO val2017 (5000 images). Subset via `--limit` flag during development, full set for final runs.

**Output needs:** Both per-stage files (e.g., `rt-detr_stage3_tf32.csv`) and a unified `results.csv` with stage column. Used to generate diploma graphs showing metric evolution.

## Constraints

- **Timeline**: < 1 month to diploma defense — critical path only
- **Hardware**: RTX 3070 with 8 GB VRAM — all models must fit
- **TRT Workspace**: Strictly 2 GB (`config.set_memory_pool_limit`)
- **Batch Size**: Strictly 1 (real-time inference simulation)
- **Baseline Integrity**: TF32 must be disabled for PyTorch FP32 baseline
- **Scientific Rigor**: 50 warm-up + 1000 measured iterations, CUDA sync between timing points
- **Memory Isolation**: VRAM reset + cache clear between engine runs
- **Code Quality**: ruff strict mode, full type annotations, modular design (open-source ready)
- **Python**: 3.13+ with uv package manager

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| All 6 models in scope | Diploma requires comprehensive comparison | — Pending |
| Strategy C optional | Time constraint — A and B sufficient for thesis | — Pending |
| Both CLI modes (single + batch) | Single for debugging, batch for final runs | — Pending |
| Subset for dev, full val2017 for final | Speed during development, rigor for results | — Pending |
| Per-stage + unified output files | Chapters need per-stage, analysis needs unified | — Pending |
| Open-source after defense | Motivates clean code and documentation | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-09 after initialization*
