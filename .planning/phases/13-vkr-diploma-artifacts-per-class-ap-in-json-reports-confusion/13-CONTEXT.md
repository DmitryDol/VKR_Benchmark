# Phase 13: VKR Diploma Artifacts — Context

**Gathered:** 2026-05-19
**Status:** Ready for planning
**Source:** PRD Express Path (`prompt4edits/code-agent-vkr-artifacts.md`)

<domain>
## Phase Boundary

Generate the visual and tabular diploma-defense artifacts the advisor requested over the existing benchmark dataset. Strictly post-process the already-built TRT engines and existing JSON reports — DO NOT rebuild engines, DO NOT add new quantization stages, DO NOT touch `report.md`. Outputs land in `media/` (main thesis text) and `results/` (appendix), organized by artifact type. Every artifact must be deterministic: same seed / image_id list / engine → same byte-identical output.

Concretely, the phase delivers:
1. Per-class COCO AP (80 classes) inside every JSON report.
2. Confusion matrices: 12×12 (supercategory aggregate, for main text) and 80×80 (full, for appendix), one of each per valid configuration.
3. Per-class summary tables (CSV + Markdown): top-10 by AP drop, top-10 by frequency, per model.
4. A 3×3 collage of representative COCO val2017 samples with GT boxes.
5. Qualitative detection comparisons: 4 models × 3 scenarios × 8 inference modes = 12 collage PNGs.
6. Real-time demo videos with FPS overlay (≥3 MP4) using a user-provided `data/demo.mp4`.
7. Three console screenshots taken manually during a single end-to-end pipeline run.

Strict scope is 35 valid configurations: 4 models × 10 stages = 40, minus the 5 defective RF-DETR-L runs whose INT8 calibration and Mixed-Precision rebuilds tactic-rolled back to FP16 (`5_trt_int8_entropy`, `5_trt_int8_minmax`, `5_trt_int8_percentile`, `6_trt_mixed_a`, `6_trt_mixed_b`). RF-DETR-L therefore contributes 5 configurations (FP32 → BF16); other models contribute 10 each.

</domain>

<decisions>
## Implementation Decisions

### Layout (locked)
- Logic modules live in `src/benchmark/eval/` (NEW package — does not exist today). Files: `__init__.py`, `per_class.py` (per-class AP postprocessor), `confusion.py` (confusion-matrix builder), and a shared `_coco_utils.py` if needed.
- CLI entry points live in `scripts/` as standalone runnable Python scripts that import from `benchmark.eval`. Files: `build_per_class_ap.py`, `build_confusion.py`, `per_class_summary.py`, `coco_collage.py`, `qualitative_examples.py`, `realtime_demo.py`.
- Pure analytics functions stay in `src/benchmark/eval/`; orchestration / file I/O / argument parsing stays in `scripts/`.

### Configuration scope (locked)
- **35 configurations.** Exclude 5 defective RF-DETR-L stages (`5_trt_int8_*` and `6_trt_mixed_*`) from per-class AP, confusion matrices, and qualitative examples. This overrides the spec's "generate anyway" note for those tasks — RF-DETR-L INT8/Mixed will not appear in any artifact.
- RF-DETR-L's qualitative-examples row uses its 5 valid stages only (PyTorch FP32, ONNX FP32, TRT TF32, TRT FP16, TRT BF16). Skip the "best/worst INT8" and "best Mixed" cells with a clear "n/a (RF-DETR-L tactic rollback)" placeholder, or omit them entirely — to be decided in the qualitative-examples plan.
- The 10 canonical stage IDs (already in the codebase) are: `1_pytorch_fp32`, `2_onnx_fp32`, `3_trt_tf32`, `4_trt_fp16`, `4_trt_bf16` (note: numbered "4" by the code, NOT "5"; do not rename — keep the existing stage IDs untouched), `5_trt_int8_entropy`, `5_trt_int8_minmax`, `5_trt_int8_percentile`, `6_trt_mixed_a`, `6_trt_mixed_b`.

### Reuse vs. rebuild (locked)
- **TRT engines exist on disk** (user confirmed). The first task re-runs `evaluate_accuracy` on each valid configuration to (a) cache the COCO-format prediction list as `coco_dt_<model>_<stage>.json` under a new `cache/predictions/` directory and (b) extract `precision[T, R, K, A, M]` from `COCOeval` to compute per-class AP. All downstream artifacts (confusion matrices, per-class summary tables, qualitative examples) read from the cache, not from the engines.
- **Do not** modify TRT build code, calibration code, or any benchmarking timing logic.
- **Do not** change the existing 40 `results/<model>/<variant>/<stage>.json` files' structure beyond adding the new `per_class_ap` field.

### Per-class AP semantics (locked)
- Use `pycocotools.cocoeval.COCOeval` with `iouType="bbox"`. Average `precision[:, :, k, 0, 2]` (T=all, R=all, K=class k, A=area all, M=maxDets 100) across IoU thresholds and recall points → AP@0.5:0.95. Also extract `precision[0, :, k, 0, 2]` → AP@0.5 directly (T=0 = IoU 0.50).
- Output structure per JSON: a `per_class_ap` field — array of 80 objects sorted by COCO-91 `class_id` ascending. Each object has: `class_id` (1..90, COCO-native), `class_name` (English, from `categories[].name` in `instances_val2017.json`), `ap_50_95` (float), `ap_50` (float), `n_gt` (int — count of GT annotations of this class in val2017).
- `n_gt` is computed once from `instances_val2017.json` and is identical across all 35 reports.

### Confusion-matrix semantics (locked)
- Greedy IoU matching at IoU ≥ 0.5, predictions sorted by descending confidence. Matched pair → `(gt_class, pred_class)` cell (filled even when correct). Unmatched GT → `(gt_class, "background")` column. Unmatched prediction → `("background", pred_class)` row.
- Confidence threshold: 0.25 applied uniformly across all models and all 35 configurations. Hardcoded as a constant in `src/benchmark/eval/confusion.py`.
- Row-normalize (each GT-class row sums to 1.0). Use `viridis` colormap with fixed `vmin=0, vmax=1`. The vmin/vmax is shared across all stages of a single model so the human eye can compare matrices.
- Supercategory mapping (12 groups): read from `categories[].supercategory` in `instances_val2017.json`. Groups: `person, vehicle, outdoor, animal, accessory, sports, kitchen, food, furniture, electronic, appliance, indoor`. Aggregate by summing matched counts per `(gt_supercat, pred_supercat)` cell before row-normalization.

### Per-class summary tables (locked)
- Per model, two tables (drop top-10 and frequency top-10). Columns: `class_name, n_gt, AP_FP32, AP_TF32, AP_FP16, AP_BF16, AP_INT8_Entropy, AP_INT8_MinMax, AP_INT8_Percentile, AP_Mixed1, AP_Mixed2, min_delta_AP` (10 stages — excluding ONNX FP32, see RF-DETR caveat below). All values are AP@0.5:0.95 from the per-class AP field.
- For RF-DETR-L, the 5 excluded stage columns contain `n/a` (string) and are skipped when computing `min_delta_AP`.
- Drop sort: by `max(AP_FP32 − AP_stage)` across stages (worst drop in the row). Frequency sort: by `n_gt` descending.
- CSV uses comma separator and **UTF-8 BOM** (Excel-compatible). Markdown uses pipe-format with `:---:` / `---:` alignment.

### COCO collage (locked)
- 9 image_ids HARDCODED in the script — no random selection. Picked once to cover the 9 supercategory bins listed in the spec. Listed in the script with a one-line justification each (e.g., `391895  # person+vehicle: city street`).
- Each thumbnail: 400×300 px with original-aspect padding. GT boxes in RGB `(0, 200, 0)`, thickness 2 px. Class label uses `cv2.FONT_HERSHEY_SIMPLEX`, scale 0.5, white text on green plate. Output: 3×3 grid in `media/coco_val2017_samples.png` (~1500×1200).

### Qualitative examples (locked)
- 3 image_ids HARDCODED per scenario (dense, occluded, large single object). The same 3 image_ids are used across all 4 models.
- 8 inference modes per cell: PyTorch FP32, ONNX FP32, TRT TF32, TRT FP16, TRT BF16, best INT8 (by mAP@0.5:0.95 among Entropy/MinMax/Percentile for this model), worst INT8 (same set), best Mixed (Strategy 1 vs 2 by mAP).
- "Best/worst" is computed by reading the existing JSON `map_50_95` values — no re-evaluation needed.
- Each subframe annotates `mAP@0.5:0.95 = X.XXX, latency = Y.Y ms` in a small font, plus the mode name on top. Predicted boxes: RGB `(0, 100, 255)` (blue), thickness 2 px, opacity 0.7. Confidence threshold 0.25.
- Layout: 4×2 grid (4 modes × 2 rows). 12 PNGs total = 4 models × 3 scenarios.

### Realtime demo (deferred-input)
- Assume `data/demo.mp4` exists — user will provide. Scripts write to `media/video/<model>_<mode>.mp4` and fail with a clean message if the input is missing.
- Required runs (≥3): RT-DETR PyTorch FP32, RT-DETR best Mixed (Strategy 1 or 2 by mAP@0.5:0.95), and one of `yolo11l` / `yolo26l` (whichever has higher FPS in the existing JSON).
- Per-frame overlay: bounding boxes (same style as qualitative), confidence-threshold 0.25, top-right rolling FPS (30-frame moving average), bottom-left "<model> <mode>". Output: H.264 via `cv2.VideoWriter_fourcc(*'mp4v')`, 30 FPS, source resolution.

### Screenshots (manual — no code)
- Three PNGs taken manually by the user after pipeline runs: `trt_build_log.png`, `benchmark_output.png`, `pycocotools_output.png`. Phase 13 only documents how to capture them and verifies they exist at the end.

### Determinism (locked)
- Every Python script sets `np.random.seed(0)` and `random.seed(0)` at module load if it uses any RNG. Image-id lists are written as sorted hardcoded literals. Per-class AP and confusion-matrix outputs are byte-identical across reruns. Verification command: run any script twice and `sha256sum` the output PNG/JSON/CSV — checksums must match.

### Code-quality (locked)
- Strict ruff conformance (the project's `strict` rule set in `pyproject.toml`).
- All new modules/functions fully type-annotated. `from __future__ import annotations` at the top.
- Module-level logger per module.
- No new runtime dependencies. Use only what's already in `pyproject.toml`: `pycocotools`, `opencv-python`, `pillow`, `matplotlib`, `seaborn`, `numpy`, `pandas`.
- Re-use existing `ResultLogger` / `BenchmarkResult` infrastructure — the per-class field extends `BenchmarkResult` (`per_class_ap: list[dict] = field(default_factory=list)`); the CSV writer ignores the new field (CSV doesn't need it; JSON does).

### What NOT to do (locked)
- No edits to `report.md` (handled by a separate editorial agent).
- No re-build of any TRT engine.
- No new quantization stages.
- No git pushes to remote — repo stays private.
- No Cyrillic text inside PNGs (font compatibility) — all captions in English.
- No edits to the existing 40 stage IDs in code.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Authoritative spec
- `prompt4edits/code-agent-vkr-artifacts.md` — the advisor's written request (Russian). Tasks 1–7 with priorities. CONTEXT.md above re-states the locked decisions, but the spec is the source of authority for any detail not enumerated here.

### Project guardrails
- `CLAUDE.md` — Project Context: hardware constraints, ruff strict mode, type annotations mandatory, no comments unless WHY is non-obvious, batch size strictly 1, BF16 verification rules, scientific rigor on warm-up/measure runs.
- `pyproject.toml` — ruff config (target-version py313, line-length 100, 24 rule categories enabled, ANN401/S101/T201/PLR0913 ignored as documented). All new code must lint clean.

### Existing benchmark code (must read before extending)
- `src/benchmark/data/coco_loader.py` — `COCODataLoader`, `COCOSample`, `COCOAnnotation`, `COCO_91_TO_80`, `COCO_80_TO_91`. The loader's `coco` attribute is a `pycocotools.coco.COCO` object — reuse, do not re-initialize.
- `src/benchmark/engines/base.py` — `BaseEngine`, `Detection`, `WARMUP_RUNS`, `MEASURE_RUNS`, and especially the `evaluate_accuracy()` method (~lines 150–200). This is where per-class AP extraction will be added: after `coco_eval.accumulate()`, `coco_eval.eval["precision"]` is a `[T, R, K, A, M]` ndarray.
- `src/benchmark/utils/logger.py` — `BenchmarkResult` dataclass (21 fields, see Phase 13 context above) and `ResultLogger`. Add `per_class_ap: list[dict[str, int | float | str]] = field(default_factory=list)` and update the JSON writer to serialize it (CSV writer keeps the existing 21-column schema; per-class data goes to JSON only).
- `src/benchmark/cli.py` — typer app. Inspect for the `run` command pattern before adding new CLI entry points (the new scripts will mirror this style).

### Existing data / reports
- `data/annotations/instances_val2017.json` — source of categories, supercategories, n_gt, image ids.
- `data/val2017/` — 5000 COCO val images.
- `results/<model>/<variant>/<stage>.json` — 40 existing reports, structure documented above. Phase 13 will extend 35 of them with `per_class_ap`.

### Roadmap context
- `.planning/ROADMAP.md` Phase 13 section — the brief goal line is auto-generated and does not capture the full scope; this CONTEXT.md is the authoritative scope document.
- `.planning/STATE.md` — `## Accumulated Context` records this phase as the active advisor-revision pass.

</canonical_refs>

<specifics>
## Specific Ideas

### Suggested 7-plan breakdown (one plan per spec task, ordered by spec priority)

The spec prioritizes tasks in this order: 1 → 2 → 3 → 5 → 4 → 6 → 7. The plans below preserve that priority (lower plan-id = higher priority). The planner agent should validate this breakdown and adjust waves/dependencies as needed.

| Plan | Spec task | Deliverable | Wave |
|------|-----------|-------------|------|
| P01 | Task 1 | `src/benchmark/eval/per_class.py` + `scripts/build_per_class_ap.py` + extend `BenchmarkResult.per_class_ap` + extend `BaseEngine.evaluate_accuracy()` to cache `cache/predictions/coco_dt_<model>_<stage>.json` + re-run on 35 configs → 35 updated `results/.../<stage>.json` files with `per_class_ap` field | 1 |
| P02 | Task 2 | `src/benchmark/eval/confusion.py` + `scripts/build_confusion.py` → 35 PNGs in `media/confusion_12/` (12×12, ~5×5 in) and 35 PNGs in `results/confusion_80/` (80×80, ~12×12 in) | 2 (depends P01) |
| P03 | Task 3 | `scripts/per_class_summary.py` → 4 models × 4 files (CSV+MD × drop_top10+freq_top10) = 16 files in `results/per_class/` and `media/per_class_md/` | 2 (depends P01) |
| P04 | Task 5 | `scripts/qualitative_examples.py` → 12 PNGs in `media/qualitative/<model>_<scenario>.png` (8-mode collage each) | 2 (depends P01) |
| P05 | Task 4 | `scripts/coco_collage.py` → `media/coco_val2017_samples.png` (3×3 GT-box collage) | 1 (no dependency on P01) |
| P06 | Task 6 | `scripts/realtime_demo.py` → ≥3 MP4 in `media/video/` (deferred until user supplies `data/demo.mp4`) | 3 (depends P01) |
| P07 | Task 7 | Documentation: `media/screenshots/README.md` instructing the user how to capture the 3 PNGs + verification commands | 3 (no code dependency) |

### Cache directory layout (locked)

```
cache/
└── predictions/
    └── coco_dt_<model>_<stage>.json   # COCO-format predictions, one per valid config (35 files)
```

`cache/` should be added to `.gitignore` to avoid bloating the repo. The user can re-generate it from existing TRT engines at any time by re-running `scripts/build_per_class_ap.py`.

### Verification per the spec's checklist (locked)

P01: 35 JSONs each contain `per_class_ap` of length 80.
P02: 35 PNGs in `media/confusion_12/`, 35 in `results/confusion_80/`.
P03: 16 files (4 models × 4 outputs).
P04: 12 PNGs in `media/qualitative/`.
P05: 1 PNG `media/coco_val2017_samples.png`.
P06: ≥3 MP4 in `media/video/` (deferred verification).
P07: 3 PNGs in `media/screenshots/` (manual — the plan only specifies docs and verification).

Cross-cutting verification: re-running any script produces byte-identical artifacts (sha256 check).

</specifics>

<deferred>
## Deferred Ideas

- `data/demo.mp4` source: user will provide. The realtime-demo script will simply fail-fast with a clear message if absent, and the plan's verification step will skip MP4 existence check (P06 is "deferred" in the plan's verification list).
- The advisor's note that RF-DETR-L INT8/Mixed matrices should be generated "anyway with a caveat" is overridden in this phase per the user's decision — see `## Decisions / Configuration scope`. If the advisor pushes back at defense, a follow-up sub-phase can backfill the 5 missing matrices using the existing FP16 predictions, but this is out of scope here.
- Any extension to ONNX FP32-only artifacts is deferred — Task 3's table omits the ONNX column intentionally (10 columns named TF32/FP16/BF16/INT8×3/Mixed×2 plus FP32 baseline = 9 stages; ONNX FP32 is essentially redundant with PyTorch FP32 for accuracy purposes and is dropped from summary tables).

</deferred>

---

*Phase: 13-vkr-diploma-artifacts-per-class-ap-in-json-reports-confusion*
*Context gathered: 2026-05-19 via PRD Express Path (manual)*
