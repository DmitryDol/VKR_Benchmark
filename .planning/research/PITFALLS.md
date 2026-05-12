# Domain Pitfalls

**Domain:** Transformer-based Object Detection Benchmarking (v2.0 Models Integration)
**Researched:** 2026-05-09

## Critical Pitfalls

Mistakes that cause rewrites or major issues during the integration of new architectures (RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26) and batch orchestration.

### Pitfall 1: ONNX Export Resolution and Dynamic Axes Limitations
**What goes wrong:** Exporting to ONNX fails or produces a broken graph when trying to enforce dynamic batching or specific resolutions across different model families.
**Why it happens:** 
- **RF-DETR:** Does not support dynamic batch sizes during export because operations like `gen_encoder_output_proposals` rely on hardcoded dimensions. 
- **D-FINE:** Export scripts often have `640x640` hardcoded. Changing this resolution arbitrarily without patching the `valid_mask` and anchor generation logic will cause a `RuntimeError` due to decoder anchor size mismatches.
- **YOLO11:** Exporter unrolls the NMS loop for the default batch size. When `nms=True` and `dynamic=True`, it only produces detections for batch index 0.
**Consequences:** Engine builds fail, or the inference returns empty/incorrect results for non-standard inputs.
**Prevention:**
- For **RF-DETR**, avoid `dynamic_axes` for batch size if it causes tracing errors. Since our project has a strict batch size of 1, hardcoding the batch size to 1 during export is preferred.
- For **D-FINE**, stick to the default training resolution (usually 640x640) for the input shape, or ensure the anchor generator is correctly scaled.
- For **YOLO11**, either export with `nms=False` and apply NMS manually via `EfficientNMS_TRT` plugin, or explicitly pass `batch=1` instead of `dynamic=True`.
**Detection:** `RuntimeError` during `torch.onnx.export` or empty `[0, 0, 0, 0]` bounding boxes during inference.

### Pitfall 2: YOLO Family NMS vs. NMS-Free Heads
**What goes wrong:** Output parsing fails or produces hundreds of overlapping boxes for YOLO models.
**Why it happens:** The YOLO architectures handle Non-Maximum Suppression (NMS) differently.
- **YOLO11:** Requires NMS. If exported without NMS and not added via plugin, the `ModelAdapter` must implement NMS in Python.
- **YOLO26 / YOLOv10:** Uses a One-to-One (O2O) NMS-free head natively. If exported with `end2end=False`, it falls back to the One-to-Many (O2M) head, producing overlapping boxes.
**Consequences:** mAP drops to 0 due to massive false positives, or post-processing logic crashes.
**Prevention:**
- For **YOLO26**, ensure export is done with `end2end=True` to activate the O2O head, simplifying `ModelAdapter.parse_outputs`.
- For **YOLO11**, standardize whether the ONNX graph includes NMS or if `ModelAdapter.parse_outputs` runs it. Given TensorRT's constraints, exporting with `nms=False` and doing it natively or via TRT plugin is safest.

### Pitfall 3: TensorRT FP16 / INT8 Precision Bugs
**What goes wrong:** FP16 or INT8 inference produces wildly incorrect detections or crashes.
**Why it happens:** Specific architectures expose bugs in certain TensorRT versions:
- **DEIMv2:** FP16 is known to be broken on TensorRT 10.4, requiring TensorRT 10.6+ for stability.
- **RF-DETR:** `LayerNorm` layers in FP16 can drastically degrade accuracy.
- **D-FINE:** INT8 Q/DQ node insertion may trigger "Internal Error Code 10" in TensorRT 10.8/10.3 unless `StronglyTyped` mode is configured.
**Consequences:** Silently poor mAP or TensorRT build failures.
**Prevention:** 
- If DEIMv2 produces 0 mAP in FP16, fallback to FP32 or force FP32 on its sensitive layers (Strategy C).
- For RF-DETR, force `LayerNormalization` nodes to FP32 during engine build if accuracy degrades in half precision.

### Pitfall 4: CLI Batch Orchestration VRAM Leaks
**What goes wrong:** The `run-all` batch CLI crashes halfway through execution due to Out of Memory (OOM) errors.
**Why it happens:** Running 6 models sequentially through 6 pipeline stages means creating and tearing down 36 distinct `BaseEngine` instances. PyTorch and TensorRT hold onto CUDA context memory aggressively.
**Consequences:** The automated benchmark dies, invalidating long-running overnight tests.
**Prevention:** 
- The batch orchestrator must explicitly invoke `del engine`, `torch.cuda.empty_cache()`, and `gc.collect()` between each model and stage iteration.
- Enforce the 2GB TRT Workspace limit strictly on every loop iteration, verifying it hasn't leaked.

### Pitfall 5: Unhandled Exceptions Killing Batch Runs
**What goes wrong:** A failure in one model (e.g., DEIMv2 ONNX export fails) aborts the entire `run-all` CLI process.
**Why it happens:** Typer's default error handling (`typer.Exit` or unhandled exceptions like `RuntimeError`) exits the Python process.
**Consequences:** Partial data generation, requiring manual restart and merging of `.csv` files.
**Prevention:** 
- Wrap the main execution loop in the `run-all` command with `try/except Exception as e`.
- Log the error using `logger.error(f"Stage X failed for {model}: {e}")`, append a "FAILED" record to the `results.csv`, and safely `continue` to the next iteration.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| ModelAdapter Implementations | Varying output tensor formats (e.g., YOLO returning `[1, 84, 8400]` vs DETR returning `[1, 300, 4]` and `[1, 300, 80]`). | Ensure the Protocol allows flexible `raw_outputs: object` unpacking specific to the model instance. |
| ONNX Export | D-FINE and RF-DETR hardcoded shape mismatches. | Force `export(batch=1)` and document hardcoded resolutions. Disable `dynamic_axes` if it breaks tracing. |
| TensorRT Quantization | DEIMv2 / D-FINE accuracy drop in FP16/INT8. | Ensure LayerNorm nodes are preserved in FP32. Utilize Mixed Precision Strategy B (preserve Softmax/LayerNorm). |
| CLI Orchestration | Unhandled exceptions and Typer `Exit` stopping the loop. | Strict `try-except` blocks inside the batch loop. Graceful continuation and logging of failures. |

## Sources

- [HIGH] TensorRT/ONNX Official docs via WebSearch for D-FINE and DEIMv2 export constraints.
- [HIGH] Ultralytics GitHub Issues for YOLO11 batch-0 bug and YOLO26 O2O head export.
- [MEDIUM] Typer documentation (via Context7) on exception handling and `typer.Exit()` behavior.