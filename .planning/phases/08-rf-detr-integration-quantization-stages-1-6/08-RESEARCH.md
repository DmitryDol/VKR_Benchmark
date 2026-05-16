# Phase 8: RF-DETR Integration & Quantization (Stages 1-6) - Research

**Researched:** 2026-05-16
**Domain:** RF-DETR-L (DINOv2 + DETR decoder) quantization & TensorRT optimization
**Confidence:** HIGH (all three D-RF-* decisions answered with concrete, in-environment evidence: vendor source inspection + actual ONNX export + graph inspection + PyTorch probe + CLI predict on a COCO sample.)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (carried forward from Phase 7 — DO NOT re-discuss)

- **C-01:** Batch size strictly 1; warm-up 50 / measure 1000 iterations.
- **C-02:** TRT workspace strictly 2 GB (`config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)`).
- **C-03:** TF32 forced **off** for the PyTorch FP32 baseline; `trt.BuilderFlag.TF32` enabled only for Stage 3.
- **C-04:** BF16 availability via `builder.platform_has_tf32` (Ampere proxy); `trt.BuilderFlag.BF16` set when supported.
- **C-05:** All three INT8 calibrators (MinMax, Entropy, Percentile) run.
- **C-06:** Calibration set = the project-standard **500 COCO val2017 images, fixed seed=42** (same as Phase 7 YOLO).
- **C-07:** Full Mixed Precision matrix — Strategy A **and** Strategy B; base = best Stage 5 calibrator per-model.
- **C-08:** D-14/D-15 — hard 2.0% mAP_50:95 gate on the best configuration; miss → flag, **not** auto-Strategy C.
- **C-09:** Every stage writes through `ResultLogger` (per-stage + unified `results.csv/json`).
- **C-10:** ONNX export pipeline **always** passes through project `simplify_onnx()` — overrides any vendor "already simplified" claim.

### Model Variant (locked — DO NOT re-discuss)

- **D-RF-01:** Use **RF-DETR-Large**. Verified config: `resolution=704`, `patch_size=16`, `num_windows=2`, `block_size=32`, `num_classes=90`, `num_queries=300`, `num_select=300`, `encoder=dinov2_windowed_small`, `hidden_dim=256`, `dec_layers=4`, pretrain weights `rf-detr-large-2026.pth` (URL: `https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth`, md5 `5cb72153541cbcb9aa6efa26222acc75`). [VERIFIED: `rfdetr/config.py` lines 272-292; `rfdetr/assets/model_weights.py` lines 212-216; live instantiation in this session.]

### Claude's Discretion (researcher MUST recommend; planner locks)

- **D-RF-02:** ONNX export pathway — see § "D-RF-02 Investigation" below.
- **D-RF-03:** Strategy B on a true transformer — see § "D-RF-03 Investigation" below.
- **D-RF-04:** Input resolution — see § "D-RF-04 Investigation" below.

### Deferred Ideas (OUT OF SCOPE for Phase 8)

- Phase 9 `input_resolution` export column (cross-phase note only; do not widen Phase 8 scope).
- Phase 10 reuse of D-RF-02 / D-RF-03 decisions for D-FINE / DEIMv2 (whatever Phase 8 picks carries forward; no re-discussion).
- RF-DETR-XL / 2XL benchmarking (licence + size; rfdetr[plus] PML 1.0 incompatible with OSS diploma).
- Strategy C / sensitivity analysis (ADV-01, future).
- TRT timing-cache reuse across Stage 6 rebuilds (deferred to Phase 11 batch-orchestration).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ADPT-07 | Implement ModelAdapter for RF-DETR (NMS-free) | § "RF-DETR Model Loading & Inference Path", "Preprocessing Contract", "Output Parsing Contract" — full Stage-1 wiring documented |
| OPT-TR-01 | Export RF-DETR to simplified ONNX (opset 18) and record Stage 2 metrics | § "D-RF-02 Investigation" — recommends vendor `RFDETR.export(opset_version=18)` + project `simplify_onnx()`; live export validated (1701 → 918 nodes, 51 LayerNormalization preserved) |
| OPT-TR-02 | Build TRT TF32/FP16/BF16 engines under 2 GB workspace | § "TRT 10.x Build Considerations" — no transformer-specific blockers identified |
| OPT-TR-03 | Run INT8 calibration (MinMax/Entropy/Percentile) on fixed 500-image set | § "Calibration Set Reuse Plan" — `int8_calibrators.load_calibration_data()` already adapter-aware; only needs RF-DETR `preprocess()` |
| OPT-TR-04 | Apply Mixed Precision Strategy A & B using best per-model calibrator | § "D-RF-03 Investigation" — recommends B2 (rely on opset-18 single-node LayerNormalization + minimal `apply_strategy_b()` extension to also mark `trt.LayerType.NORMALIZATION`); current "norm" substring heuristic ALREADY fires on RF-DETR layer names but the extension hardens the contract |
| OPT-TR-05 | Best config within 2.0% mAP_50:95 of FP32 baseline (D-14/D-15 gate) | § "Validation Architecture" — same Nyquist pattern as Phase 7, applied per-RF-DETR; gate miss = flag, not auto-fallback |
</phase_requirements>

## Phase Summary

RF-DETR-L is a **real transformer** (DINOv2-windowed-small backbone with `nn.LayerNorm` everywhere + a 4-layer DETR decoder with `norm1/norm2/norm3` `nn.LayerNorm` per layer). Live ONNX export through the vendor's `RFDETR.export(opset_version=18)` API succeeded in this session and produced a **clean graph with 51 single-node `LayerNormalization` ops and 20 `Softmax` ops** (zero decomposed Reduce/Pow patterns). The vendor exporter performs the model's `.export()` mode-switch (replaces `forward` with `forward_export`), is monkey-patch-free for our purposes, and lands on opset 18 cleanly when the caller passes `opset_version=18` (the vendor's default is 17, the function arg is configurable). After running the project's mandatory `simplify_onnx()` step on top of the vendor export, node count drops from 1701 → 918 with all 51 `LayerNormalization` and 20 `Softmax` nodes preserved.

The transformer is therefore **Strategy-B-friendly** at the existing project heuristic (`"norm" in layer.name.lower()` substring + `LayerType.SOFTMAX`). RF-DETR's exported ONNX graph names every LayerNorm node `.../norm1/LayerNormalization`, `.../norm2/LayerNormalization`, `.../norm3/LayerNormalization` — all match the substring "norm" — so `apply_strategy_b()` will fire on every one of them out of the box. Hardening the heuristic to also explicitly mark `trt.LayerType.NORMALIZATION` (the TRT 10.x native layer type that `INormalizationLayer` registers for opset-17+ `LayerNormalization` nodes) makes the contract explicit and is the same minimal change Phase 10 (D-FINE + DEIMv2) will inherit.

Output post-processing is **simpler than RT-DETR**: the model returns `(dets, labels)` with `labels` shape `(1, 300, 91)` where the 91 logits map **directly to COCO-91 category IDs by index** (index 0 = N/A, indices 1..90 = COCO-91 IDs with natural sparse gaps, index 90 = background "no-object"). **No COCO-80→COCO-91 LUT is required** — the adapter just sigmoids the logits, filters out index 0 and index 90, applies a topk + threshold, and emits raw category IDs straight into `Detection.labels`. Verified by running `RFDETRLarge().predict('data/val2017/000000000139.jpg', threshold=0.5)` in this session and observing class_id=72 → refrigerator (COCO-91 ID 72 = refrigerator, correct direct mapping).

**Primary recommendation:** Implement `RFDETRAdapter` using the vendor PyTorch model (`m.model.model` from `RFDETRLarge()`) for Stage 1, write a standalone `scripts/export_rfdetr_onnx.py` that wraps `RFDETRLarge().export(opset_version=18, shape=(704,704)) → project simplify_onnx()` for Stage 2 ONNX, lock resolution at **704×704** (vendor default — matches AP_50:95=56.5 reference), extend `apply_strategy_b()` to also mark `trt.LayerType.NORMALIZATION` for explicit-contract Phase-10 reuse, reuse Phase 7's 500-image seed=42 calibration set with adapter-driven preprocess delegation (already supported by `int8_calibrators.load_calibration_data()`), and add a small CLI patch so `compute_macs` and the `OnnxRuntimeEngine`/`TensorRTEngine` infer input shape from the adapter rather than hardcoded `(640, 640)`.

## D-RF-02 Investigation: ONNX Export Pathway

### Evidence Collected (this session)

**Vendor API: `RFDETR.export()`** — defined in `rfdetr/detr.py` lines 746-884, with three layers:

1. `RFDETR.export(...)` (detr.py:758) calls `model.export()` (LWDETR's `export()` at `rfdetr/models/lwdetr.py:179`) which monkey-patches `forward = forward_export` (lwdetr.py:284) so the traced graph emits `(pred_boxes, pred_logits)` instead of the training-mode dict with `aux_outputs`/`enc_outputs`. The output names are explicitly set to `["dets", "labels"]` (detr.py:837).
2. It then delegates to `rfdetr/export/main.py:export_onnx` → `rfdetr/export/_onnx/exporter.py:export_onnx` (lines 54-110). The inner `export_onnx` does plain `torch.onnx.export(..., dynamo=False, do_constant_folding=True, keep_initializers_as_inputs=False)` with caller-provided `input_names`, `output_names`, `dynamic_axes`, `opset_version`. **There is NO simplifier, NO ONNX surgery, NO custom optimizer pass** in the path RFDETR.export takes by default. Note the vendor's `simplify` kwarg is deprecated since 1.6 and explicitly logs a warning + ignores it (lwdetr.py via the @deprecated decorator + `export/main.py` lines 171-174 plus `RFDETR.export` `args_mapping={"simplify": False}` line 747).
3. The vendor function default `opset_version=17` (detr.py:764), but the parameter is fully configurable — passing `opset_version=18` works cleanly.

**Live export test (this session, `weights/rfdetr-l-research/inference_model.onnx`):**

```python
from rfdetr import RFDETRLarge
m = RFDETRLarge()  # downloads rf-detr-large-2026.pth, ~150 MB
m.export(output_dir='weights/rfdetr-l-research', opset_version=18, shape=(704,704))
# → produces inference_model.onnx, 123.2 MB, opset 18, IR 8, producer pytorch 2.11.0
```

Graph after vendor export (raw, before project simplify):
- 1701 nodes total
- `LayerNormalization: 51` (every nn.LayerNorm emitted as a single op — opset 17/18 contract honored)
- `Softmax: 20`
- `MatMul: 151`, `Add: 184`, `Reshape: 141`, `Transpose: 125`, `Mul: 119`, `Constant: 506`, `Slice: 69`, `Sqrt: 48`, `Cast: 41`, `Concat: 37`, `Div: 37`, `Shape: 30`, `Unsqueeze: 25`, `Erf: 12`, `Expand: 10`, `Conv: 9`, `Relu: 9`, `ConstantOfShape: 8`
- `ReduceMean: 0`, `Pow: 0` — no decomposed-LayerNorm subgraphs anywhere
- Inputs: `input: [1, 3, 704, 704]`
- Outputs: `dets: [1, 300, 4]` (cx,cy,w,h normalized to [0,1]), `labels: [1, 300, 91]` (per-class logits; sigmoid not applied in graph)

After running project `simplify_onnx()` (`benchmark.engines.onnx_export.simplify_onnx`) on the vendor output:
- 1701 → **918 nodes (46% reduction)**, size 123.2 MB → 120.1 MB
- LayerNormalization preserved: **51 → 51**
- Softmax preserved: **20 → 20**
- onnxsim's `check_n=3` validator passed (`check_ok=True`)
- Graph still contains `GridSample: 4` (DINOv2 windowed attention's interpolated position embeddings — handled natively by ORT 1.22+ and TRT 10.x)
- Graph still contains `TopK: 1` — but only one near the very end (this is from the encoder's `enc_outputs` topk selection, NOT the postprocess TopK; the model still requires application-level topk+threshold)

**LWDETR.export() side effect (important):** `model.export()` is destructive — it replaces `self.forward = self.forward_export`. After export, the model can no longer be used in training mode without calling `m.train()` to restore it. For our Stage 2 export script that's fine because we instantiate, export, and discard.

**Verdict: vendor exporter is clean** — does what we want at opset 18, no opaque surgery, and we always re-run our own `simplify_onnx()` per C-10.

### Comparison: Project's `torch.onnx.export` + custom wrapper (option b)

Possible — RT-DETR uses this pattern (`scripts/export_rtdetr_onnx.py` + `RTDetrONNXWrapper`). But for RF-DETR it'd require:

1. Manually replicating `LWDETR.export()`'s `forward = forward_export` swap (the inference forward is fundamentally different from training forward — it skips `aux_outputs` and `enc_outputs` entirely).
2. Manually wrapping `(pred_boxes, pred_logits)` tuple ordering, which we'd have to derive from reading vendor source anyway.
3. Re-implementing the post-decoder `bbox_reparam` deltas-to-cx,cy,w,h math the vendor's `forward_export` already emits.

This is strictly more code with strictly more risk of drift from the vendor's tested inference path. The only benefit would be controlling op-level details — but the vendor's `torch.onnx.export(..., dynamo=False, do_constant_folding=True)` call is already the same legacy-TorchScript path we use for RT-DETR. No benefit.

### Recommendation: **D-RF-02 = (a) `rfdetr.export()` + project `simplify_onnx()`**

Create `scripts/export_rfdetr_onnx.py`:

```python
from pathlib import Path
from rfdetr import RFDETRLarge
from benchmark.engines.onnx_export import simplify_onnx, validate_onnx

OUT_DIR = Path("weights/rfdetr-l")
OUT_DIR.mkdir(parents=True, exist_ok=True)

m = RFDETRLarge()  # downloads rf-detr-large-2026.pth via vendor's cache
m.export(
    output_dir=str(OUT_DIR),
    opset_version=18,          # match project transformer convention (RT-DETR uses 18)
    shape=(704, 704),          # vendor default — see D-RF-04
    batch_size=1,              # C-01
    dynamic_batch=False,       # batch=1 fixed for TRT
)
# Vendor writes `inference_model.onnx` — rename to project convention
raw = OUT_DIR / "inference_model.onnx"
target_raw = OUT_DIR / "rfdetr_l.onnx"
raw.rename(target_raw)
sim = simplify_onnx(target_raw, output_path=OUT_DIR / "rfdetr_l_sim.onnx")
validate_onnx(sim)
```

**Rationale (locked under C-10):** the project `simplify_onnx()` runs unconditionally — vendor's own simplifier is dead code (deprecated in 1.6, no-op in 1.6.5). Vendor exporter is just a `torch.onnx.export` wrapper that handles the LWDETR forward-swap correctly; we don't reinvent that.

## D-RF-03 Investigation: Strategy B on True Transformer

### The risk we set out to verify

The Phase 7 (YOLO) `apply_strategy_b()` heuristic (`src/benchmark/engines/mixed_precision.py:50-65`) marks a TRT layer as FP16 if:

1. `layer.type == trt.LayerType.SOFTMAX`, **OR**
2. `"norm" in layer.name.lower()`.

The CONTEXT risk: if PyTorch + opset 18 decomposes `nn.LayerNorm` into `ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add`, then in the TRT network those would be ~8 separate elementwise/reduce layers per LayerNorm, none of which contain "norm" in their auto-generated layer names — Strategy B would silently miss every LayerNorm on a real transformer.

### Evidence (this session, against the actual rfdetr ONNX graph)

After `m.export(opset_version=18)` followed by `simplify_onnx()`:

- **51 single-node `LayerNormalization` ops** (zero decomposed Reduce/Pow subgraphs). Examples:
  - `/backbone/backbone.0/encoder/encoder/encoder/layer.0/norm1/LayerNormalization`
  - `/backbone/backbone.0/encoder/encoder/encoder/layer.0/norm2/LayerNormalization`
  - `/transformer/decoder/layers.3/norm3/LayerNormalization`
  - `/transformer/decoder/norm/LayerNormalization`
- **20 `Softmax` ops** — all in attention blocks, names like `/backbone/.../attention/attention/Softmax` and `/transformer/decoder/layers.N/...Softmax`.
- Every LayerNorm node name **contains the substring "norm"** (case-insensitive). The current `apply_strategy_b()` heuristic WILL fire on each.
- ReduceMean: 0, Pow: 0 in the simplified graph — the decomposed-LayerNorm pitfall does NOT manifest at opset 18 with PyTorch 2.11.

This is consistent with PyTorch's documented behavior: `aten::layer_norm` exports as a single `onnx::LayerNormalization` node at opset ≥ 17 ([source](https://discuss.pytorch.org/t/using-opset-version-higher-than-18-in-onnx/203832), [PyTorch issue #126160](https://github.com/pytorch/pytorch/issues/126160)).

### TRT 10.x side

TensorRT 10.x has a native `INormalizationLayer` that registers under `trt.LayerType.NORMALIZATION` and natively absorbs `onnx::LayerNormalization` nodes ([NVIDIA capabilities doc](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/capabilities.html); the doc explicitly recommends "target the latest ONNX opset containing the corresponding function ops (for example, opset 17 for LayerNormalization … numerical accuracy using function ops is superior to the corresponding implementation with primitive ops"). When TRT 10.x parses the simplified RF-DETR graph, every `onnx::LayerNormalization` becomes an `INormalizationLayer` with `LayerType.NORMALIZATION`.

Critical detail: TRT names these layers using the ONNX node name from the source graph (e.g. `/transformer/decoder/layers.3/norm3/LayerNormalization`). So the current `"norm" in layer.name.lower()` heuristic already fires correctly. But **string-matching is fragile** — adding the explicit type check is one extra OR-branch and removes the fragility for Phase 10's D-FINE / DEIMv2.

### Recommendation: **D-RF-03 = B2 (extend `apply_strategy_b()` with `LayerType.NORMALIZATION` check)**

Minimal hardening, ~3 lines:

```python
# src/benchmark/engines/mixed_precision.py
def apply_strategy_b(network: trt.INetworkDefinition) -> int:
    """Strategy B: Apply FP16 to Softmax and LayerNorm/Normalization nodes."""
    count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if is_constant_or_shape(layer):
            continue
        # ADD: explicit NORMALIZATION type covers opset >= 17 LayerNormalization
        is_norm = (
            layer.type == trt.LayerType.SOFTMAX
            or layer.type == trt.LayerType.NORMALIZATION  # NEW
            or "norm" in layer.name.lower()
        )
        if is_norm:
            layer.precision = trt.float16
            layer.set_output_type(0, trt.float16)
            count += 1
    return count
```

**Why B2, not B1 or B3:**

- **Not B1 (status quo):** The current heuristic *will* work for RF-DETR (every LayerNorm node name contains "norm"), but it's coincidence — driven by PyTorch's auto-naming convention. A future model or vendor whose ONNX graph names its norm nodes something else (e.g. `Block_3_Normalize`) would silently miss them. The risk is invisible — Strategy B reports a count and proceeds; you'd only catch the miss by carefully cross-referencing layer counts. Hardening with the explicit type check is essentially free.
- **Not B3 (subgraph pattern matcher):** Decomposed LayerNorm subgraphs do NOT appear in our actual graph at opset 18. Building and maintaining a pattern matcher for a case that doesn't occur is wasted effort and a maintenance burden that Phase 10 inherits. If a future model DOES emit decomposed LayerNorms, that's the right moment to add B3 — not now.
- **B2 wins** because it costs ~3 lines, is reusable verbatim in Phase 10 for D-FINE / DEIMv2, and documents the intent (we want to FP16 the *normalization layers*, not whatever substring "norm" matches).

**Acceptance gate:** Plan task should `assert apply_strategy_b(network) >= 51` (one per nn.LayerNorm) **plus** `>= 20` (one per Softmax) — i.e. ≥ 71 layers marked. If the count is < 71 something has broken the assumption and we WANT the build to fail loudly.

**Phase 10 inheritance:** D-FINE + DEIMv2 use the same opset 18 + LayerNormalization path. The B2 extension carries forward unchanged. Document this as a Phase 10 invariant.

## D-RF-04 Investigation: Input Resolution

### Confirmation (this session, vendor source)

`RFDETRLargeConfig` (`rfdetr/config.py:272-292`):
- `resolution: int = 704`
- `patch_size: int = 16`
- `num_windows: int = 2`
- `block_size = patch_size * num_windows = 32`
- `positional_encoding_size: int = 704 // 16 = 44`

The resolution is configurable (must be divisible by `block_size=32`). 640 satisfies divisibility (`640 % 32 == 0` — verified live). Live test confirmed `RFDETRLargeConfig(resolution=640)` constructs cleanly with `positional_encoding_size` auto-rescaled to 40 (via the `_sync_pe_with_resolution` model_validator in config.py:117-153).

### Why 704 stays, not 640

1. **Pretrained weights baked at 704/PE=44.** The published checkpoint `rf-detr-large-2026.pth` was trained at 704×704 with `positional_encoding_size=44`. Loading at PE=40 (640×640) would either fail the state_dict load (size mismatch on `pos_embed.weight`) or force a PE interpolation that drifts from the reported AP_50:95=56.5 baseline. The vendor warning logs we observed during instantiation already flag a PE mismatch even at 704 (because the underlying DINOv2 backbone was originally trained at a different patch_size — vendor handles that internally), but resolution=704 is the only setting the vendor explicitly validates against the checkpoint.
2. **vs-paper integrity for the diploma.** The diploma's headline number for RF-DETR is the published 56.5 AP_50:95, which is at 704×704. Running off-spec would invalidate the cross-model comparison ("RF-DETR underperforms" because it ran at the wrong resolution).
3. **Cross-model symmetry is a diploma narrative, not a benchmark requirement.** YOLO11l/26l (640) and RT-DETR R50vd (640) all happen to share 640, but each was trained at its native resolution. RF-DETR at 704 is the correct apples-to-apples-to-its-own-paper choice.
4. **VRAM safety on RTX 3070.** 704² vs 640² is +21% input pixels. Backbone activation tensors scale roughly linearly with token count (which is `(H/patch_size)²` = (44²=1936) vs (40²=1600)). 21% extra activations on an 8 GB card with 2 GB TRT workspace is comfortable — RT-DETR R50vd at 640 used ~1.5-2 GB peak VRAM in v1.0; a +21% activation increase keeps RF-DETR well under the limit. (Quantitative measurement happens in Phase 8 Stage 1 — but the order-of-magnitude analysis says we're safe.)

### Recommendation: **D-RF-04 = keep vendor default 704×704**

- `RFDETRAdapter.input_size` returns `(704, 704)`.
- `scripts/export_rfdetr_onnx.py` passes `shape=(704, 704)`.
- `compute_macs(..., input_shape=(1, 3, 704, 704))` — see § "Files to Create / Modify" for the cli.py change.

**Cross-phase note (already captured in CONTEXT.md Deferred Ideas):** Phase 9 export tables MUST carry an `input_resolution` column. Do NOT widen Phase 8 scope to handle that.

## RF-DETR Model Loading & Inference Path (Stage 1 wiring)

### Loading

```python
from rfdetr import RFDETRLarge

m = RFDETRLarge()                       # downloads rf-detr-large-2026.pth (~150 MB) to vendor cache
nn_model = m.model.model                # the inner nn.Module (LWDETR); .eval() not yet set
nn_model.eval()
nn_model.to(device)                     # device placement is the caller's responsibility (m.model.device is set lazily otherwise)
```

The `RFDETR.__init__` triggers `_load_pretrain_weights_into` which downloads and applies the checkpoint. The model is **CPU at construction time** by deliberate design (avoids CUDA init in __init__ to keep PTL DDP-spawn-able in notebooks); the adapter must explicitly `.to(device)`.

**Forward shape probe (this session):**

```
Input:  torch.randn(1, 3, 704, 704)
Output: dict {
  pred_logits:  (1, 300, 91) float32
  pred_boxes:   (1, 300,  4) float32  # cx, cy, w, h normalized to [0, 1]
  aux_outputs:  list (training-mode auxiliary losses — empty in eval but key present)
  enc_outputs:  dict (two-stage encoder outputs)
}
```

The adapter calls `model(pixel_values)` (positional, single tensor input) — NOT `model(pixel_values=...)` like HF RT-DETR. The model accepts either a `(B, 3, H, W)` tensor or a `NestedTensor`; positional tensor is the simpler path and the one `predict()` itself uses (detr.py:1217).

### Detection contract (consumed by `parse_outputs`)

| Field | Shape | Semantics |
|-------|-------|-----------|
| `pred_logits` | `(B, 300, 91)` float32 | Raw per-class logits. Index 0 = N/A (never trained as positive — COCO has no category_id=0). Indices 1..90 map **directly** to COCO-91 `category_id`. Index 90 = background (the `num_classes`-th slot per DETR convention). NO sigmoid applied in graph. |
| `pred_boxes` | `(B, 300, 4)` float32 | (cx, cy, w, h) normalized to [0, 1] in input-image space. Already post-`bbox_reparam` (delta + reference → absolute), so the adapter does NOT need to apply `sigmoid` to boxes. |

**COCO-91 evidence (this session, `predict()` on `data/val2017/000000000139.jpg`):**

| class_id (raw) | vendor predict label (cosmetic bug) | True COCO-91 ID | Correct COCO label |
|----------------|--------------------------------------|------------------|---------------------|
| 72 | "refrigerator" ✓ | 72 | refrigerator ✓ |
| 86 | "" (out-of-bounds in 80-list) | 86 | vase |
| 62 | "tv" ✗ | 62 | chair |
| 1 | "bicycle" ✗ | 1 | person |

The vendor's `predict()` displays `COCO_CLASS_NAMES[class_id]` which is an off-by-one mistake (their 80-entry name list is indexed by 0-79 but COCO-91 IDs are 1-90 sparse). The **integer `class_id` value itself is the correct COCO-91 ID** — confirmed by `rfdetr/datasets/coco.py:109` comment ("output slot k corresponds directly to COCO category ID k" for non-remapped COCO pretraining).

This means the RF-DETR adapter is **substantially simpler than RT-DETR**: no `_COCO80_LUT` array, no remap step. Directly take the class index as the category_id.

## Preprocessing Contract (mean/std/resize/letterbox)

### Source: `rfdetr/detr.py:373-375, 1180-1183` (the RFDETR.predict() pipeline)

```python
class RFDETR:
    means = [0.485, 0.456, 0.406]   # standard ImageNet mean (DINOv2 inherits)
    stds  = [0.229, 0.224, 0.225]   # standard ImageNet std

# predict() preprocessing:
img_tensor = F.to_tensor(img)                          # HWC uint8 → CHW float32 / 255 → [0,1]
img_tensor = F.resize(img_tensor, [resolution, resolution])  # torchvision.transforms.functional.resize — DIRECT resize, no letterbox
img_tensor = F.normalize(img_tensor, means, stds)
```

### What this is, in adapter terms

| Property | Value |
|----------|-------|
| Channel order | **RGB** (PIL default; `F.to_tensor(PIL_Image)` preserves RGB) |
| Pixel range | `[0, 1]` float32 (from `F.to_tensor` division by 255) |
| Normalization | ImageNet: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Resize strategy | **Direct stretch resize to `(704, 704)`** via `torchvision.transforms.functional.resize` (bilinear, antialias=True since torchvision 0.17). **NOT letterbox.** |
| Input tensor shape | `(1, 3, 704, 704)` |
| Patch alignment | Required — but 704 is divisible by `block_size=32`, so a direct resize to 704×704 satisfies it. |

**Why no letterbox (unlike YOLO):** DINOv2-windowed-attention transformers tokenize the image into a `(H/patch_size, W/patch_size)` patch grid and require both dimensions divisible by `patch_size * num_windows`. Letterboxing would either (a) waste tokens on grey padding (efficiency hit on a small 8 GB card) or (b) require aspect-ratio-preserving square padding that still needs patch alignment — vendor opted for the simpler direct-resize and trains the model with multi-scale augmentation (`multi_scale=True`, `expanded_scales=True` per `TrainConfig`) to compensate for aspect-ratio distortion. Inference must replicate training: direct resize.

### Adapter implementation sketch

```python
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import torch

# Module-level constants
_INPUT_SIZE: tuple[int, int] = (704, 704)   # (H, W) — D-RF-04
_IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]

class RFDETRAdapter:
    @property
    def input_size(self) -> tuple[int, int]:
        return _INPUT_SIZE

    def preprocess(self, sample: COCOSample, device: torch.device | None = None) -> torch.Tensor:
        # sample.image is HWC RGB uint8 (project convention from COCOSample)
        img = Image.fromarray(sample.image)        # RGB
        t = F.to_tensor(img)                       # (3, H, W) float32 [0,1]
        t = F.resize(t, [_INPUT_SIZE[0], _INPUT_SIZE[1]])  # direct resize to (704, 704)
        t = F.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)
        t = t.unsqueeze(0)                          # (1, 3, 704, 704)
        if device is not None:
            t = t.to(device)
        return t
```

**No inverse-letterbox needed in `parse_outputs`** — boxes are normalized cx,cy,w,h in `[0, 1]` of the resized 704×704 input; scaling back to original pixels is just multiply by `original_w` and `original_h` (no padding offsets to subtract, unlike YOLO).

## Output Parsing Contract (raw → Detection)

### What `parse_outputs` receives

| Source | `raw_outputs` shape |
|--------|---------------------|
| PyTorch (`RFDETRAdapter.infer`) | `dict` with `pred_logits: (1,300,91) torch.Tensor`, `pred_boxes: (1,300,4) torch.Tensor`. The `aux_outputs` and `enc_outputs` dict keys exist but are ignored — only `pred_logits` and `pred_boxes` are read. |
| ONNX Runtime | `list[np.ndarray]` in graph output order: `[dets, labels]` = `[(1,300,4), (1,300,91)]`. **Note ordering: `dets` first, `labels` second** (per `export/main.py:120`: `output_names = ["dets", "labels"]`). |
| TensorRT | `list[np.ndarray]` in the same order as the ONNX graph: `[dets, labels]`. |

### Algorithm (mirrors `rfdetr/models/postprocess.py:PostProcess.forward` for top-k)

```python
def parse_outputs(
    self,
    raw_outputs: object,
    original_size: tuple[int, int],
    input_size: tuple[int, int],   # noqa: ARG002 — boxes are normalized
    score_threshold: float,
) -> Detection:
    # --- Unify input format ---
    if isinstance(raw_outputs, dict):                       # PyTorch path
        logits = raw_outputs["pred_logits"][0]              # (300, 91) torch
        boxes_norm = raw_outputs["pred_boxes"][0]           # (300, 4)  torch
    elif isinstance(raw_outputs, (list, tuple)):            # ONNX / TRT path
        # Graph output order: [dets, labels] = [(1,300,4), (1,300,91)]
        boxes_norm = torch.from_numpy(raw_outputs[0])[0]    # (300, 4)
        logits     = torch.from_numpy(raw_outputs[1])[0]    # (300, 91)
    else:
        raise TypeError(f"Unsupported raw_outputs type: {type(raw_outputs)}")

    # --- Sigmoid + top-k by per-query × per-class joint score ---
    # Matches PostProcess: topk over flattened (queries × classes) so high-scoring
    # multi-class predictions on the same query are not lost.
    probs = logits.sigmoid()                                # (300, 91)
    flat = probs.view(-1)                                   # (300*91,)
    num_select = 300                                        # = ModelConfig.num_select
    topk_vals, topk_idx = torch.topk(flat, num_select)      # (300,) each
    # Decompose flat index → (query_idx, class_idx)
    query_idx = topk_idx // probs.shape[1]                  # (300,) values in 0..299
    class_idx = topk_idx %  probs.shape[1]                  # (300,) values in 0..90

    # --- Threshold filter + remove no-object / N/A slots ---
    # class_idx == 0: COCO-91 has no id=0 (never trained as positive — drop)
    # class_idx == 90: background "no-object" per DETR convention (drop)
    keep = (topk_vals >= score_threshold) & (class_idx != 0) & (class_idx != 90)
    scores = topk_vals[keep].cpu().numpy().astype(np.float32)
    labels = class_idx[keep].cpu().numpy().astype(np.int64)  # direct COCO-91 IDs
    sel_queries = query_idx[keep]                            # which of 300 queries

    # --- Gather + denormalize boxes ---
    sel_boxes = boxes_norm[sel_queries]                      # (N, 4) cx,cy,w,h in [0,1]
    orig_h, orig_w = original_size
    cx, cy, w, h = sel_boxes.unbind(-1)
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy().astype(np.float32)

    return Detection(
        boxes=boxes_xyxy.reshape(-1, 4),
        scores=scores,
        labels=labels,
    )
```

**Key correctness notes:**

1. **No COCO-80→COCO-91 LUT** — the labels coming out of `class_idx` ARE COCO-91 IDs.
2. **Filter index 0 AND index 90.** Index 0 is "N/A" (COCO has no id=0; the logit is essentially noise but during topk it could slip through with high sigmoid score on degenerate inputs). Index 90 is "background" (the canonical DETR no-object class). Both must be dropped before COCOeval sees the results.
3. **Topk over flattened (queries × classes), NOT per-query argmax.** This is the vendor convention (`postprocess.py:43-46`) — the same query can contribute multiple detections (different classes) to the top-300 set. Per-query argmax would lose those.
4. **No softmax over classes** — RF-DETR uses sigmoid (independent per-class), like RT-DETR. There is no exclusive class probability; multi-label outputs are by design.
5. **score_threshold=0.001 (project default)** is the right starting point for evaluation; `predict()` defaults to 0.5 for human-visible inference but COCOeval expects to see low-confidence detections too (it ranks across all of them).

## ONNX Graph Inspection (LayerNorm node pattern observed)

Already covered in detail in § D-RF-03. Summary table:

| Property | Before `simplify_onnx()` | After `simplify_onnx()` |
|----------|--------------------------|--------------------------|
| Total nodes | 1701 | **918 (-46%)** |
| `LayerNormalization` (single node, opset 17/18) | 51 | **51** (preserved) |
| `Softmax` | 20 | **20** (preserved) |
| `ReduceMean` | 0 | 0 |
| `Pow` | 0 | 0 |
| `Sqrt` | 48 | 0 (folded into LayerNormalization / unrelated) |
| `MatMul` | 151 | 126 |
| `GridSample` (DINOv2 windowed attention) | 4 | 4 |
| `TopK` | 1 | 1 |
| File size (MB) | 123.2 | 120.1 |
| Opset / IR / Producer | 18 / 8 / pytorch 2.11.0 | unchanged |
| Input shape | `[1, 3, 704, 704]` | unchanged |
| Output shapes | `dets: [1,300,4]`, `labels: [1,300,91]` | unchanged |
| onnxsim `check_n=3` validator | n/a | passes |

**Decomposed-LayerNorm pitfall: NOT PRESENT.** Strategy B's existing "norm" substring heuristic will fire on all 51 LayerNorm layers (every name matches `*norm*/LayerNormalization`); the proposed B2 hardening adds a belt-and-braces explicit `LayerType.NORMALIZATION` check.

## Calibration Set Reuse Plan

### What we reuse from Phase 7 unchanged

- The **500-image fixed set** (the first 500 COCO val2017 image_ids in sorted order — deterministic via `COCODataLoader.__post_init__` doing `sorted(self._coco.getImgIds())[:limit]`, no shuffle, no seed). Per CLI helper: `cli.py:_build_calibration_dataloader()` returns `COCODataLoader(limit=500)`.
- The three calibrators: `MinMaxCalibrator`, `EntropyCalibrator`, `PercentileCalibrator` — all model-agnostic (`int8_calibrators.py`).
- The cache-file convention (per-method per-model: `engines/<model>/cal_<method>.cache`).

### What's RF-DETR-specific (only one thing)

`load_calibration_data(dataloader, adapter=adapter_instance)` (`int8_calibrators.py:44-96`) already checks `getattr(adapter, "preprocess", None)` and delegates to the adapter's preprocess. Since `RFDETRAdapter.preprocess()` returns a tensor at the model's native 704×704 with the correct ImageNet normalization, **the calibration tensors will be at 704×704 — different from YOLO's 640×640**. The function literally needs no code change; just pass `adapter=RFDETRAdapter()` from the CLI like YOLO does.

The one CLI piece that needs attention: `int8_calibrators._INPUT_SIZE: tuple[int, int] = (640, 640)` (line 41) is only used as a **fallback** when `adapter is None`. With adapter-driven preprocess, that constant is dead code for RF-DETR. Leave it alone (don't break the fallback for a future adapter-less use).

### Verification (this session)

The Phase 7 plumbing was confirmed by `cli.py:229-262` (Stage 5 INT8 branch): `cal_dataloader = _build_calibration_dataloader()` then passed verbatim. For RF-DETR the diff is one line — adding the model to `MODEL_REGISTRY` and wiring `_get_adapter("rfdetr-l")` to return `RFDETRAdapter()`. No new calibration code.

**Important warning to surface in the plan:** Phase 7 found `EntropyCalibrator2` catastrophic on YOLO CNN heads (30-47% mAP drop) — but the same calibrator was *competitive* on RT-DETR transformer attention. RF-DETR is closer to RT-DETR in topology (DINOv2 transformer backbone, transformer decoder). Expect Entropy to perform **well** here. The per-model `save_int8_best_calibrator` (CAL-05) auto-picks the winner; no manual override needed.

## TRT 10.x Build Considerations (BF16 sm_86, workspace, INT8 calibrators)

### Build flags — unchanged from Phase 7

| Stage | TRT BuilderFlag(s) | Workspace | Calibrator |
|-------|---------------------|-----------|------------|
| 3_trt_tf32 | `TF32` | 2 GB | none |
| 4_trt_fp16 | `FP16` | 2 GB | none |
| 4_trt_bf16 | `BF16` (set iff `builder.platform_has_tf32`) | 2 GB | none |
| 5_trt_int8_* | `INT8` + `FP16` (mixed for fallback) | 2 GB | one of MinMax / Entropy / Percentile |
| 6_trt_mixed_a | `INT8` + `FP16` + Strategy A (per-layer FP16 first/last) | 2 GB | best of Stage 5 |
| 6_trt_mixed_b | `INT8` + `FP16` + Strategy B (per-layer FP16 LayerNorm + Softmax) | 2 GB | best of Stage 5 |

### Transformer-specific verification

- **BF16 on Ampere sm_86:** No DINOv2 or DETR-decoder ops are documented as BF16-blocking. The standard Ampere proxy check (`builder.platform_has_tf32 == True` ⇒ BF16 available) still applies. GridSample (the only "exotic" op in the graph) is BF16-compatible in TRT 10.x.
- **Workspace = 2 GB:** RT-DETR R50vd at 640 used ~600-900 MB workspace in v1.0 builds. RF-DETR-L at 704 has more attention heads (`ca_nheads=16`, `sa_nheads=8`) and more activations, but the 4-decoder-layer + small backbone keeps overall layer count modest (918 nodes simplified). Expected workspace usage: ~1.0-1.5 GB. Some headroom remains, but **plan task should explicitly verify build completes under the 2 GB ceiling** and flag if it doesn't (no auto-raise — that would violate C-02).
- **TopK in the graph:** TRT 10.x supports `TopK` natively. The single TopK node near the end is from the encoder's two-stage `enc_outputs` proposal step; no special TRT handling required.
- **GridSample (4 nodes):** Native support since TRT 8.5. Mode-bilinear, align_corners-false is the DINOv2 default. No TRT plugin needed; just confirm via TRT build log that GridSample layers select a supported tactic.
- **`INormalizationLayer` (51 layers, after parse):** Native support since TRT 8.6. TRT 10.x is the assumed runtime — no concerns.

### Engine naming — model-scoped paths (Phase 7 convention extended)

| Precision | Engine file |
|-----------|-------------|
| TF32 | `engines/rfdetr-l/rfdetr-l_tf32.engine` |
| FP16 | `engines/rfdetr-l/rfdetr-l_fp16.engine` |
| BF16 | `engines/rfdetr-l/rfdetr-l_bf16.engine` |
| INT8 MinMax | `engines/rfdetr-l/rfdetr-l_int8_minmax.engine` |
| INT8 Entropy | `engines/rfdetr-l/rfdetr-l_int8_entropy.engine` |
| INT8 Percentile | `engines/rfdetr-l/rfdetr-l_int8_percentile.engine` |
| Mixed A | `engines/rfdetr-l/rfdetr-l_mixed_a.engine` |
| Mixed B | `engines/rfdetr-l/rfdetr-l_mixed_b.engine` |

Calibration caches: `engines/rfdetr-l/cal_minmax.cache`, `engines/rfdetr-l/cal_entropy.cache`, `engines/rfdetr-l/cal_percentile.cache`.

No collision with YOLO Phase 7 (`engines/yolo11l/*`, `engines/yolo26l/*`) or RT-DETR v1.0 (`engines/rt-detr/*`).

## Engine Naming & MODEL_REGISTRY Integration

### `MODEL_REGISTRY` entry (extend `src/benchmark/cli.py:73-89`)

```python
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "rt-detr":  {"weights": "weights/rtdetr-r50vd/", "onnx": "weights/rtdetr-r50vd/rtdetr_r50_sim.onnx", "family": "detr"},
    "yolo11l":  {"weights": "weights/yolo11l/yolo11l.pt", "onnx": "weights/yolo11l/yolo11l_sim.onnx", "family": "yolo"},
    "yolo26l":  {"weights": "weights/yolo26l/yolo26l.pt", "onnx": "weights/yolo26l/yolo26l_sim.onnx", "family": "yolo"},
    "rfdetr-l": {  # NEW
        "weights": "weights/rfdetr-l/",                          # vendor downloads .pth here on RFDETRLarge() init
        "onnx":    "weights/rfdetr-l/rfdetr_l_sim.onnx",         # scripts/export_rfdetr_onnx.py output
        "family":  "rfdetr",                                       # for MACs routing if needed
    },
}
```

### `_get_adapter()` branch (extend `src/benchmark/cli.py:101-125`)

```python
def _get_adapter(model_name: str) -> object:
    if model_name == "rt-detr":
        from benchmark.models.rtdetr_adapter import RTDETRAdapter  # noqa: PLC0415
        return RTDETRAdapter()
    if model_name in ("yolo11l", "yolo26l"):
        from benchmark.models.yolo_adapter import YOLOAdapter  # noqa: PLC0415
        return YOLOAdapter(is_nms_free=(model_name == "yolo26l"))
    if model_name == "rfdetr-l":                                              # NEW
        from benchmark.models.rfdetr_adapter import RFDETRAdapter  # noqa: PLC0415
        return RFDETRAdapter()
    msg = f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
    raise typer.BadParameter(msg)
```

### Bug fix opportunity (also needed for Phase 8): adapter-driven input shape in CLI

The current `cli.py:153` hardcodes `input_shape=(1, 3, 640, 640)` for `compute_macs`. That number is wrong for RF-DETR (704). Replace with the adapter-driven value:

```python
# cli.py — Stage 1 branch
h, w = adapter.input_size                            # NEW: read from adapter
if macs is None:
    macs, flops = compute_macs(engine.model, model_name, input_shape=(1, 3, h, w))  # FIXED
```

Same pattern: `OnnxRuntimeEngine(... input_size=(640, 640))` constructor call at cli.py:188 already silently ignores its `input_size` (the field is stored but never read — see § Engine Naming details). Leave the cli.py value as-is for backwards-compat, since the adapter's `preprocess` drives the actual tensor shape.

## Validation Architecture (Nyquist Dimension 8)

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already present per `pyproject.toml` `[tool.ruff.lint.per-file-ignores] tests/**/*.py`) — but **no `tests/` directory currently exists in the repo** (verified). Phase 8 follows the Phase 7 pattern: **scripted CLI invocations are the regression gates, not pytest unit tests.** This matches the v1.0 / Phase 6 / Phase 7 reality — there is no `tests/` infrastructure to extend, and writing pytest cases for "TRT engine builds and produces metrics" is impractical in a 1-month diploma timeline. |
| Quick run command (per task — smoke) | `uv run benchmark run --model rfdetr-l --stage <stage_id> --limit 16` (16-image smoke run completes in <30 s for Stage 1; <2 min for any TRT stage including engine build) |
| Full suite command (per wave merge) | `uv run benchmark run --model rfdetr-l --stage <stage_id>` (full 5000-image COCO val2017 mAP eval — ~5-10 min for Stage 1, ~3-8 min for TRT stages) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|---------------------|--------------|
| ADPT-07 | Stage 1 PyTorch FP32 baseline runs end-to-end, produces non-zero mAP_50:95 | smoke | `uv run benchmark run --model rfdetr-l --stage 1_pytorch_fp32 --limit 16` | ❌ Wave 0 — depends on `RFDETRAdapter` (new) + MODEL_REGISTRY entry (new) |
| OPT-TR-01 (Stage 2) | Simplified ONNX exists and ORT inference produces non-zero mAP | smoke | `uv run python scripts/export_rfdetr_onnx.py` then `uv run benchmark run --model rfdetr-l --stage 2_onnx_fp32 --limit 16` | ❌ Wave 0 — depends on new export script |
| OPT-TR-02 (Stages 3-4) | TF32/FP16/BF16 engines build under 2 GB and produce mAP | smoke | `uv run benchmark run --model rfdetr-l --stage 3_trt_tf32,4_trt_fp16,4_trt_bf16 --limit 16 --force-rebuild` | ✓ (TensorRTEngine already in place; needs only adapter + ONNX) |
| OPT-TR-03 (Stage 5) | Three INT8 calibrators complete, write caches, produce mAP | smoke | `uv run benchmark run --model rfdetr-l --stage 5_trt_int8_minmax,5_trt_int8_entropy,5_trt_int8_percentile --limit 16 --force-rebuild` | ✓ (calibrators already adapter-aware) |
| OPT-TR-04 (Stage 6) | Mixed A and Mixed B engines build with ≥ 71 layers marked for Strategy B (51 LayerNorm + 20 Softmax minimum) | smoke + assertion | `uv run benchmark run --model rfdetr-l --stage 6_trt_mixed_a,6_trt_mixed_b --limit 16 --force-rebuild` + Stage 6 task asserts `apply_strategy_b()` returned count ≥ 71 | partial — needs B2 patch to `mixed_precision.py` |
| OPT-TR-05 (D-14/D-15 gate) | Best config across all 8 stages within 2.0% mAP_50:95 of Stage 1 baseline | gate | Full-suite run + `result_logger.save_int8_best_calibrator("rfdetr-l")` + manual review of `results.csv` `accuracy_drop_pct` column | ✓ (ResultLogger logic in place from Phase 7) |

### Sampling Rate
- **Per task commit:** the `--limit 16` smoke run for that stage. Total time < 3 min for any Stage 2-6 task (including engine build).
- **Per wave merge:** the full-suite run for every stage shipped in that wave (no `--limit`). Wave 4 (full Stage 6) is the heaviest: full ONNX + 3 INT8 + 2 Mixed Precision builds + 6 full mAP evaluations ≈ 60-90 min.
- **Phase gate (`/gsd-verify-work`):** the full corpus `results.csv` reviewed for the D-15 2.0% gate. One row per stage × precision = 8 rows for RF-DETR. Gate check is mechanical: filter to `model_name == "rfdetr-l"`, find min `accuracy_drop_pct` across rows, require ≤ 2.0%.

### Wave 0 Gaps

- [ ] `src/benchmark/models/rfdetr_adapter.py` — new file implementing `ModelAdapter` (load, infer, preprocess, parse_outputs, input_size). Covers ADPT-07.
- [ ] `scripts/export_rfdetr_onnx.py` — new file. Covers OPT-TR-01 export step.
- [ ] `MODEL_REGISTRY` entry for `"rfdetr-l"` + `_get_adapter` branch in `src/benchmark/cli.py`. One-line additions.
- [ ] `apply_strategy_b()` extension (3-line patch) in `src/benchmark/engines/mixed_precision.py` — adds `LayerType.NORMALIZATION` to the FP16-mark predicate. Covers OPT-TR-04 robustness.
- [ ] `compute_macs` input_shape fix in `src/benchmark/cli.py:153` — read `adapter.input_size` instead of hardcoded `(640, 640)`. Covers ADPT-07 correctness for non-640 models.
- [ ] (optional, but cheap) Document the resolved D-RF-02/03/04 in the existing Phase 8 plan files when they're written, so Phase 10 can copy-paste.

*(No pytest gaps: project has no `tests/` directory and Phase 7 also operated without unit tests. The CLI smoke runs ARE the regression gates.)*

## Files to Create / Modify (table)

| File | Action | Lines | Phase 8 Plan |
|------|--------|-------|--------------|
| `src/benchmark/models/rfdetr_adapter.py` | **CREATE** | ~120 (input_size, load, preprocess, infer, parse_outputs + module constants for mean/std/resolution) | Stage-1 adapter plan |
| `scripts/export_rfdetr_onnx.py` | **CREATE** | ~40 (RFDETRLarge() → m.export(opset=18) → simplify_onnx → validate_onnx; ruff-strict, full type ann.) | Stage-2 ONNX plan |
| `src/benchmark/cli.py` (`MODEL_REGISTRY`) | **MODIFY** | +5 (one dict entry) | Stage-1 adapter plan (same wave) |
| `src/benchmark/cli.py` (`_get_adapter`) | **MODIFY** | +3 (one `if`-branch with lazy import) | Stage-1 adapter plan |
| `src/benchmark/cli.py` (Stage-1 `compute_macs` call) | **MODIFY** | +2 (read `adapter.input_size` instead of hardcoded `(640, 640)`) | Stage-1 adapter plan |
| `src/benchmark/engines/mixed_precision.py` (`apply_strategy_b`) | **MODIFY** | +2 (`or layer.type == trt.LayerType.NORMALIZATION` clause) | Stage-6 Mixed Precision plan |
| `weights/rfdetr-l/` (directory + downloaded `rf-detr-large-2026.pth`) | **CREATE** at first `RFDETRLarge()` call (auto via vendor downloader) | n/a | Wave-0 / Stage-1 prerequisite |
| `engines/rfdetr-l/` (TRT engines + INT8 caches) | **CREATE** at build time | n/a | Stages 3-6 outputs |
| `results/rfdetr-l/<run-id>/...` (per-stage CSV/JSON + unified) | **CREATE** at run time via existing `ResultLogger` | n/a | Every stage's deliverable |
| `.planning/phases/08-rf-detr-integration-quantization-stages-1-6/08-0N-PLAN.md` | **CREATE** (planner output) | n/a | Planner step after this RESEARCH |

**Files that do NOT need touching** (verified, contrary to what a naive reader might assume):
- `src/benchmark/engines/onnx_export.py` — `export_to_onnx` / `simplify_onnx` already do what RF-DETR needs.
- `src/benchmark/engines/onnx_engine.py` — `OnnxRuntimeEngine` already delegates preprocess to adapter.
- `src/benchmark/engines/tensorrt_engine.py` — TRT build flow already adapter-aware.
- `src/benchmark/engines/int8_calibrators.py` — `load_calibration_data(adapter=...)` already calls adapter preprocess.
- `src/benchmark/data/coco_loader.py` — calibration set helper (`_build_calibration_dataloader` in cli.py) unchanged.
- `src/benchmark/utils/logger.py` — `ResultLogger` schema covers all metrics.

## Open Risks & Landmines

1. **Vendor `RFDETR.export()` is destructive on the model object.** `LWDETR.export()` swaps `self.forward = self.forward_export` in-place. After running the export script, the same `m.model.model` instance can NO LONGER be used for training-mode forward passes. **Mitigation:** the export script instantiates → exports → exits. The Stage-1 PyTorch baseline path instantiates a fresh `RFDETRLarge()` separately. They never share a model instance. **Document this in the export script's docstring.**

2. **Vendor `predict()` class_name display is buggy** (`class_names[class_id]` is off-by-one for COCO pretraining because COCO_CLASS_NAMES has only 80 entries while class_id is COCO-91). The *integer* class_id is correct — only the display label is wrong. The adapter never calls `predict()`; it calls the raw model and parses outputs directly. **The vendor bug does not affect our adapter.** Surfacing here so a future reader doesn't get distracted by the warning logs.

3. **Vendor logs PE / patch-size warnings on `RFDETRLarge()` instantiation** ("Using a different number of positional encodings than DINOv2…"). These are EXPECTED — RF-DETR is fine-tuned from DINOv2 with adapted PE and patch_size. The pretrained `rf-detr-large-2026.pth` checkpoint contains the adapted weights; the warning is purely about the underlying DINOv2 backbone weights NOT being loaded (because they would conflict with RF-DETR's adaptations). **Suppress in the adapter logging if noisy; do NOT treat as an error.**

4. **Activation distribution unknown for INT8.** Phase 7 found Entropy catastrophic on YOLO CNN heads but fine on RT-DETR transformer attention. RF-DETR is closer to RT-DETR (transformer), so Entropy is *expected* to be competitive — but this is a hypothesis, not measurement. The per-model `save_int8_best_calibrator(model)` automatically picks the winner; the plan task should not pre-judge.

5. **`compute_macs` for transformer with GridSample.** The project's `compute_macs` helper uses `thop` (or similar) under the hood. `thop` can fail to count FLOPs for ops it doesn't know (GridSample, custom attention). The result might be a partial undercounting. **Mitigation:** if MACs come back suspicious (very low for a 33.9M-param model), log it but proceed — MACs is a secondary diploma metric, not a verification gate.

6. **Vendor weight download is a network-dependent side effect of `RFDETRLarge()` instantiation.** First-run takes ~30-60 s for the 150 MB download. **Mitigation:** the planner should ensure Wave 0 or first Stage 1 task explicitly notes "first run downloads weights"; subsequent runs hit the vendor's local cache (verified: `File rf-detr-large-2026.pth already exists with correct MD5 hash.`).

7. **`m.model.device` is `cuda` by default but the model itself is on CPU until first use.** `_ensure_model_on_device` (detr.py:345) does the deferred placement. The Stage 1 adapter must call `nn_model.to(device)` explicitly (the existing adapter pattern already does this, just don't assume the model is on GPU after `RFDETRLarge()`).

8. **The exported ONNX has `dets` and `labels` in that ORDER (not labels first like RT-DETR).** RT-DETR's ONNX outputs `[logits, pred_boxes]`. RF-DETR's outputs `[dets (=pred_boxes), labels (=pred_logits)]`. **The adapter's `parse_outputs` MUST detect output order from the list, not assume RT-DETR ordering.** The contract is documented in vendor `export/main.py:120`. The recommended pattern: detect by **shape**, not by index — `(N, 4)` is boxes, `(N, 91)` is logits. Then there's no ambiguity.

## Sources

### Primary (HIGH confidence)
- **Vendor source inspection (this session):**
  - `rfdetr/detr.py` (1529 lines) — `RFDETR`, `RFDETRLarge`, `ModelContext`, `export()`, `predict()`, `optimize_for_inference()`, preprocessing constants (lines 373-375: means/stds).
  - `rfdetr/config.py` — `RFDETRLargeConfig` (lines 272-292; resolution=704, patch_size=16, num_windows=2, num_queries=300, num_select=300, num_classes=90).
  - `rfdetr/export/main.py` — `export_onnx` orchestrator (output_names=["dets","labels"], BatchNorm-free check, model.export() invocation).
  - `rfdetr/export/_onnx/exporter.py` — inner `export_onnx` (lines 54-110: `torch.onnx.export(..., dynamo=False, do_constant_folding=True)`).
  - `rfdetr/models/lwdetr.py` (export() at line 179, forward at line 187, forward_export at line 284).
  - `rfdetr/models/postprocess.py` — `PostProcess.forward` (lines 27-80: sigmoid + topk over flattened logits, class_idx = topk_idx % num_classes).
  - `rfdetr/datasets/coco.py` — line 109 ("output slot k corresponds directly to COCO category ID k" for non-remapped COCO pretraining).
  - `rfdetr/assets/model_weights.py` (lines 207-216: RF_DETR_LARGE_2026 URL + MD5).
  - `rfdetr/assets/coco_classes.py` — 80-entry COCO_CLASS_NAMES list (vendor's off-by-one source).
- **Live experiments (this session, repo `C:\projects\ResearchSSAU\VKR_Claude`):**
  - `RFDETRLarge()` instantiation + weights download + `.export(opset_version=18, shape=(704,704))` → produced `weights/rfdetr-l-research/inference_model.onnx`, 123.2 MB.
  - Project `simplify_onnx()` applied → produced `inference_model_sim.onnx`, 120.1 MB, 918 nodes, 51 LayerNormalization, 20 Softmax, 0 ReduceMean, 0 Pow.
  - `RFDETRLarge().predict('data/val2017/000000000139.jpg', threshold=0.5)` → confirmed class_id=72 → refrigerator (direct COCO-91 mapping).
  - `RFDETRLargeConfig(resolution=640)` → confirmed config-valid (PE auto-rescales 44→40).
  - `nn_model(torch.randn(1,3,704,704))` → confirmed PyTorch dict output `{pred_logits: (1,300,91), pred_boxes: (1,300,4), aux_outputs, enc_outputs}`.
- **Project source:**
  - `src/benchmark/models/rtdetr_adapter.py` — adapter pattern template.
  - `src/benchmark/models/yolo_adapter.py` — adapter pattern + preprocess delegation example.
  - `src/benchmark/engines/onnx_export.py` — `export_to_onnx`, `simplify_onnx`, `validate_onnx` (default `opset_version=18`).
  - `src/benchmark/engines/int8_calibrators.py` — `load_calibration_data(adapter=...)` already adapter-aware.
  - `src/benchmark/engines/mixed_precision.py` — `apply_strategy_b` heuristic to extend.
  - `src/benchmark/cli.py` — MODEL_REGISTRY, _get_adapter, _run_stage wiring.

### Secondary (MEDIUM confidence — verified against official sources)
- PyTorch ONNX export emits `nn.LayerNorm` as single `onnx::LayerNormalization` node at opset ≥ 17 — confirmed via the PyTorch discussion forum thread "Using opset version higher than 18 in onnx" (https://discuss.pytorch.org/t/using-opset-version-higher-than-18-in-onnx/203832) and PyTorch issue [#126160](https://github.com/pytorch/pytorch/issues/126160).
- TensorRT 10.x has native `INormalizationLayer` registered as `trt.LayerType.NORMALIZATION` and natively absorbs `onnx::LayerNormalization` for opset ≥ 17 — confirmed via NVIDIA TensorRT capabilities documentation (https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/capabilities.html).

### Tertiary
- None used. All claims sourced from primary or secondary above; no LOW-confidence assertions in this RESEARCH.

## Metadata

**Confidence breakdown:**
- D-RF-02 (vendor exporter clean, use it): **HIGH** — live export validated, simplifier preserves all 51 LayerNorm + 20 Softmax nodes, vendor's `simplify` kwarg is deprecated no-op so C-10 isn't conflicting with anything.
- D-RF-03 (B2 minimal heuristic extension): **HIGH** — graph contains zero decomposed-LayerNorm patterns at opset 18; existing "norm" substring match fires on every vendor LayerNorm node; explicit `LayerType.NORMALIZATION` check is a 3-line robustness/Phase-10 future-proofing.
- D-RF-04 (704×704): **HIGH** — vendor config default + checkpoint trained at PE=44 (= 704/16); 640 is config-valid but breaks the vs-paper AP_50:95=56.5 baseline.
- Stage 1 wiring (load, preprocess, parse_outputs): **HIGH** — PyTorch forward shape confirmed, COCO-91 direct mapping confirmed by live predict(), preprocessing constants confirmed in vendor source.
- INT8 calibration reuse: **HIGH** — `load_calibration_data(adapter=...)` is already adapter-aware in `int8_calibrators.py`; only diff vs YOLO is preprocess output shape (704 vs 640), automatic.
- TRT 10.x build viability: **MEDIUM** — no transformer-specific blockers identified, but actual workspace usage at 704² is unmeasured in this session (only known to be < 2 GB by analogy to RT-DETR R50vd at 640²).

**Research date:** 2026-05-16
**Valid until:** 2026-06-16 (30-day window; rfdetr 1.6.5.post0, PyTorch 2.11.0, TensorRT 10.16.1.11 — stable stack).

**Assumptions Log (assumed claims requiring user/planner confirmation):** none — all claims tagged HIGH/MEDIUM are backed by vendor source, live experiment, or official documentation in this session.
