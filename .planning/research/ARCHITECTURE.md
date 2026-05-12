# Architecture Patterns

**Domain:** Computer Vision Optimization / Benchmarking
**Researched:** 2026-05-09

## Recommended Architecture

To horizontally scale the pipeline for 5 new SOTA models (RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26) and add a batch runner, the architecture relies on the existing **Adapter Pattern** and a **CLI Orchestrator**.

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `ModelAdapter` (Protocol) | Decouples model specifics (loading, output parsing) | `PyTorchEngine`, `OnnxRuntimeEngine`, etc. |
| `YOLO*Adapter` | Loads YOLO models, parses outputs, applies NMS | `torchvision.ops.nms`, `PyTorchEngine` |
| `*DETRAdapter` | Loads DETR-based models, directly parses logits | `PyTorchEngine` |
| `run-all` (CLI Command) | Orchestrates sequential execution across all models | `MODEL_REGISTRY`, `_run_stage`, `ResultLogger` |
| `MODEL_REGISTRY` | Maps CLI names to adapter classes and weight paths | `cli.py` |

### Data Flow

1. CLI `run-all` iterates through `MODEL_REGISTRY.keys()`.
2. For each model, it sequentially loops through `STAGE_REGISTRY`.
3. `_run_stage` dynamically instantiates the appropriate adapter via `_get_adapter()`.
4. The Engine uses the adapter to load the model, pre-process, and post-process. YOLO adapters apply `torchvision.ops.nms` during post-processing to filter boxes.
5. `ResultLogger` writes per-model per-stage CSVs. At the end of the batch run, it merges them into a global summary.

## Patterns to Follow

### Pattern 1: Strategy Pattern for Post-Processing
**What:** Each target architecture implements the `ModelAdapter` protocol.
**When:** Integrating new models with different architectures and post-processing needs (e.g., NMS vs no NMS).
**Example:**
```python
class YOLO11Adapter:
    def parse_outputs(self, raw_outputs, original_size, input_size, score_threshold):
        # 1. Apply confidence threshold
        # 2. Apply NMS: torchvision.ops.nms(boxes, scores, iou_threshold)
        # 3. Format and return Detection(...)
```

### Pattern 2: Resilient Batch Orchestration
**What:** Centralized orchestration using Typer, looping over registries with fault tolerance.
**When:** Automating runs across multiple models and stages for unattended benchmarking.
**Implementation:** Wrap `_run_stage` in a `try/except` block inside the `run-all` loop. If a specific model/stage fails, log the error and continue to the next, ensuring one failure doesn't halt a 10-hour benchmark.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Hardcoded Model Logic in Engines
**What:** `if model_name.startswith("yolo"): apply_nms()` inside `BaseEngine` or `PyTorchEngine`.
**Why bad:** Violates Open/Closed Principle. Engines become bloated and tightly coupled to architectures.
**Instead:** Encapsulate all model-specific post-processing strictly within their respective `ModelAdapter.parse_outputs()`.

### Anti-Pattern 2: VRAM State Leakage
**What:** Running multiple models sequentially in `run-all` without resetting the GPU state.
**Why bad:** VRAM tracks will be cumulatively skewed, leading to false peak memory readings or OOM errors.
**Instead:** Ensure the batch CLI cleanly destroys engine instances, and `BaseEngine.reset_vram_tracking()` is called between iterations.

## Scalability Considerations

| Concern | 1 Model | 6 Models |
|---------|--------------|--------------|
| Registry | Simple hardcoded dict | Requires lazy loading of adapters in `_get_adapter` to save memory and import time. |
| VRAM | Single peak measurement | Must explicitly call `torch.cuda.empty_cache()` between models to prevent OOM. |
| Stability | Fail-fast is fine | Requires try/except error boundaries around stages to prevent full batch failure. |

## Sources
- `src/benchmark/engines/pytorch_engine.py`
- `src/benchmark/cli.py`
