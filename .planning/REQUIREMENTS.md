# Requirements: VKR Benchmark

**Defined:** 2026-05-12
**Core Value:** Scientifically rigorous, per-stage metric logging that produces publication-ready CSV/JSON reports showing how each optimization stage affects every metric — no intermediate results lost.

## v2.0 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Model Adapters
- [ ] **ADPT-04**: Implement ModelAdapter for YOLO11 (requires NMS)
- [ ] **ADPT-05**: Implement ModelAdapter for D-FINE (NMS-free)
- [ ] **ADPT-06**: Implement ModelAdapter for DEIMv2 (NMS-free)
- [ ] **ADPT-07**: Implement ModelAdapter for RF-DETR (NMS-free)
- [ ] **ADPT-08**: Implement ModelAdapter for YOLO26 (NMS-free, end2end=True)

### Batch Orchestration CLI
- [ ] **CLI-04**: Implement `run-all` Typer command to sequentially execute all models across all stages
- [ ] **CLI-05**: Enforce explicit CUDA cache clearing (`torch.cuda.empty_cache()`) and garbage collection between batch runs to prevent VRAM leaks
- [ ] **CLI-06**: Implement resilient exception handling within the batch execution loop

### Output & Logging
- [ ] **LOG-13**: Batch CLI correctly appends to the unified `results.csv` with `model_name` and `stage` columns
- [ ] **LOG-14**: Auto-generate `summary.md` with markdown tables comparing Latency, FPS, mAP, and VRAM across models

## Future Requirements

Deferred to future release. Tracked but not in current roadmap.

### Analysis
- **ADV-01**: Strategy C (Sensitivity Analysis)
- **ADV-03**: Automated LaTeX table generation
- **ADV-04**: Automated chart generation

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Multi-GPU Orchestration | Out of scope for this diploma (constrained to single RTX 3070). |
| Live Dashboard / Web UI | Unnecessary complexity for an academic benchmark. |
| Custom Model Training/Fine-tuning | Goal is inference evaluation, not training. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ADPT-04 | | Pending |
| ADPT-05 | | Pending |
| ADPT-06 | | Pending |
| ADPT-07 | | Pending |
| ADPT-08 | | Pending |
| CLI-04 | | Pending |
| CLI-05 | | Pending |
| CLI-06 | | Pending |
| LOG-13 | | Pending |
| LOG-14 | | Pending |

**Coverage:**
- v2.0 requirements: 10 total
- Mapped to phases: 0
- Unmapped: 10 

---
*Requirements defined: 2026-05-12*
*Last updated: 2026-05-12 after initial definition*
