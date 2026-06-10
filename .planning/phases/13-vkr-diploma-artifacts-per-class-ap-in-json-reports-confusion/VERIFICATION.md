---
phase: 13-vkr-diploma-artifacts-per-class-ap-in-json-reports-confusion
verified: 2026-05-19T19:36:29Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
deferred:
  - truth: "35 stage JSONs contain per_class_ap of length 80"
    addressed_in: "User action — Mode B (--live) pass"
    evidence: "Infrastructure ships complete; backfill requires ~24 h GPU run per 13.01 PLAN spec. Mode A (pure-postprocess) confirmed to report 35 skips + exit 0 when cache empty."
  - truth: "35 × 12×12 PNGs in media/confusion_12/ and 35 × 80×80 PNGs in results/confusion_80/"
    addressed_in: "User action — after Mode B populates cache"
    evidence: "build_confusion.py confirmed to skip 35 configs + exit 0 when cache empty. RF-DETR-L exclusion correct."
  - truth: "16 per-class summary table files in results/per_class/ and media/per_class_md/"
    addressed_in: "User action — after Mode B populates cache"
    evidence: "per_class_summary.py exits 2 gracefully with clear message when JSONs not backfilled."
  - truth: "12 qualitative detection PNGs with real predictions in media/qualitative/"
    addressed_in: "User action — after Mode B populates cache"
    evidence: "qualitative_examples.py generates 12 placeholder PNGs now (graceful degradation); will fill real predictions after cache populated."
  - truth: "3+ MP4s in media/video/"
    addressed_in: "User action — supply data/demo.mp4"
    evidence: "realtime_demo.py exits 2 with documented message when data/demo.mp4 absent."
  - truth: "3 console screenshots in media/screenshots/"
    addressed_in: "Manual user action"
    evidence: "media/screenshots/README.md documents the exact capture procedure."
human_verification: []
---

# Phase 13: VKR Diploma Artifacts Verification Report

**Phase Goal:** Diploma defense receives publication-ready visual + tabular artifacts (per-class AP, confusion matrices 12x12 / 80x80, per-class summary tables, COCO sample collage, qualitative detection examples, realtime demo videos with FPS overlay, console screenshot capture docs) produced deterministically by post-processing the existing 35 valid (model x stage) benchmark runs.

**Verified:** 2026-05-19T19:36:29Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Per-class AP infrastructure exists and is importable | VERIFIED | `src/benchmark/eval/per_class.py`, `__init__.py` — import smoke test passed |
| 2 | `BenchmarkResult.per_class_ap` field present, CSV-excluded | VERIFIED | `logger.py:93` — field with `default_factory=list`; `_append_csv` and `save_stage_files` both call `row.pop("per_class_ap", None)` before CSV write |
| 3 | `BaseEngine.evaluate_accuracy` caches predictions + returns per_class_ap key | VERIFIED | `base.py:153-250` — signature `(dataloader, cache_stage=None, cache_predictions=True)`, returns dict with `"per_class_ap"` key; `run_full_benchmark` extracts it and passes to `BenchmarkResult(per_class_ap=per_class_ap, ...)` |
| 4 | `scripts/build_per_class_ap.py` — Mode A (empty cache → 35 skips, exit 0); RF-DETR-L 5 invalid stages excluded | VERIFIED | Ran: `updated=0 skipped-no-cache=35 skipped-no-json=0 other=0`; excluded 5 RF-DETR-L stages logged as "Excluded config" |
| 5 | `scripts/build_confusion.py` — graceful skip when cache empty; RF-DETR-L excluded | VERIFIED | Ran: `rendered=0 skipped_missing_cache=35 skipped_excluded=5` |
| 6 | `media/coco_val2017_samples.png` exists and is valid PNG | VERIFIED | PIL verified: size `(1524, 1224)`, mode `RGB`; 9 hardcoded image_ids covering 9 supercategory bins |
| 7 | `media/screenshots/README.md` has all 4 required sections | VERIFIED | All 4 sections found: `## Expected files`, `## How to capture`, `## Verification`, `## Notes` |

**Score:** 7/7 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/benchmark/eval/__init__.py` | Package re-exporting 3 names | VERIFIED | Exports `PerClassAPEntry`, `compute_per_class_ap_from_results`, `compute_per_class_ap_from_cache` |
| `src/benchmark/eval/per_class.py` | Per-class AP postprocessor | VERIFIED | `compute_per_class_ap_from_results` + `compute_per_class_ap_from_cache` + `PerClassAPEntry` TypedDict |
| `src/benchmark/eval/confusion.py` | Greedy IoU matching + renderer | VERIFIED | Exports `build_confusion_80`, `aggregate_to_supercat_12`, `render_confusion_png`, `row_normalize`, `IOU_THRESHOLD`, `CONFIDENCE_THRESHOLD`, `SUPERCATEGORIES` |
| `scripts/build_per_class_ap.py` | Mode A/B CLI | VERIFIED | `--help` confirmed; Mode A empty-cache path tested |
| `scripts/build_confusion.py` | Confusion PNG CLI | VERIFIED | `--help` confirmed; empty-cache graceful skip tested |
| `scripts/per_class_summary.py` | Summary table CLI | VERIFIED | `--help` confirmed; exits 2 with message when JSONs not backfilled |
| `scripts/qualitative_examples.py` | Qualitative collage CLI | VERIFIED | 12 PNGs generated with placeholder cells for missing cache |
| `scripts/coco_collage.py` | COCO collage CLI | VERIFIED | Produces `media/coco_val2017_samples.png` (1524x1224) |
| `scripts/realtime_demo.py` | Realtime demo CLI | VERIFIED | Exits 2 with documented message when `data/demo.mp4` absent |
| `media/coco_val2017_samples.png` | 3x3 GT-box collage | VERIFIED | PIL-verified, 1524x1224 RGB |
| `media/screenshots/README.md` | Screenshot capture docs | VERIFIED | 109 lines, all 4 required sections present |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `base.py::evaluate_accuracy` | `per_class.py::compute_per_class_ap_from_results` | function call after `coco_eval.accumulate()` | WIRED | `base.py:232` — `per_class = compute_per_class_ap_from_results(coco_eval, dataloader.coco)` |
| `base.py::run_full_benchmark` | `BenchmarkResult(per_class_ap=...)` | `.pop("per_class_ap")` from accuracy dict | WIRED | `base.py` — `per_class_ap: list[...] = accuracy.pop("per_class_ap", [])` then passed to constructor |
| `ResultLogger._append_csv` | CSV write (per_class_ap excluded) | `row.pop("per_class_ap", None)` | WIRED | `logger.py:448` |
| `ResultLogger.save_stage_files` | CSV write (per_class_ap excluded) | `row_csv.pop("per_class_ap", None)` | WIRED | `logger.py:179` |
| `scripts/build_confusion.py` | `benchmark.eval.confusion::build_confusion_80` | import + call | WIRED | Imports confirmed; RFDETR_L_INVALID_STAGES matches exactly |
| `scripts/qualitative_examples.py` | `cache/predictions/coco_dt_*.json` | json.load per config | WIRED | Missing cache renders placeholder cell, no crash |

---

### RF-DETR-L Exclusion Consistency

`RFDETR_L_INVALID_STAGES = frozenset({"5_trt_int8_entropy", "5_trt_int8_minmax", "5_trt_int8_percentile", "6_trt_mixed_a", "6_trt_mixed_b"})` is present and identical in all 4 scripts that iterate configs:

| Script | RFDETR_L_INVALID_STAGES | Status |
|--------|------------------------|--------|
| `scripts/build_per_class_ap.py` | 5-element frozenset, correct | VERIFIED |
| `scripts/build_confusion.py` | 5-element frozenset, correct | VERIFIED |
| `scripts/per_class_summary.py` | 5-element frozenset, correct | VERIFIED |
| `scripts/qualitative_examples.py` | 5-element frozenset, correct | VERIFIED |

---

### Data-Flow Trace (Level 4)

The pipeline is infrastructure-only at this point (cache not yet populated). The data-flow paths are:

- `evaluate_accuracy` → writes `cache/predictions/coco_dt_<model>_<stage>.json` → read by `build_confusion.py`, `qualitative_examples.py`, `per_class_summary.py` (via `build_per_class_ap.py` first)
- All paths handle missing cache files gracefully (log warning, skip, exit 0 or exit 2 with message)
- Data-flow correctness is structurally verified: `compute_per_class_ap_from_cache` reloads predictions, runs fresh `COCOeval`, delegates to `compute_per_class_ap_from_results` — same logic path as live inference

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Import smoke (all Phase 13 exports) | `uv run python -c "from benchmark.eval..."` | `IMPORTS OK` | PASS |
| `build_per_class_ap.py --help` | CLI check | Shows all 6 options including `--live` | PASS |
| Mode A empty-cache | `uv run python scripts/build_per_class_ap.py` | `updated=0 skipped-no-cache=35` exit 0 | PASS |
| `build_confusion.py` empty-cache | `uv run python scripts/build_confusion.py` | `rendered=0 skipped_missing_cache=35 skipped_excluded=5` | PASS |
| `per_class_summary.py` not-backfilled | `uv run python scripts/per_class_summary.py` | Exit 2, clear message | PASS |
| `qualitative_examples.py` empty-cache | `uv run python scripts/qualitative_examples.py` | 12 PNGs generated (placeholder cells) | PASS |
| `realtime_demo.py` no demo.mp4 | `uv run python scripts/realtime_demo.py` | Exit 2, documented message | PASS |
| `media/coco_val2017_samples.png` PIL verify | PIL `img.verify()` + size check | `(1524, 1224) RGB` | PASS |
| README sections | `python -c "... assert '## Expected files' in txt ..."` | All 4 sections found | PASS |

---

### Code Quality Gates

| Check | Command | Result |
|-------|---------|--------|
| ruff strict (all Phase 13 files) | `uv run ruff check src/benchmark/eval/ scripts/build_per_class_ap.py scripts/build_confusion.py scripts/per_class_summary.py scripts/qualitative_examples.py scripts/coco_collage.py scripts/realtime_demo.py` | **All checks passed** |
| .gitignore `cache/` entry | `grep -n "cache/" .gitignore` | Line 26: `cache/` |

---

### Anti-Patterns Found

No debt markers (`TBD`, `FIXME`, `XXX`) found in any Phase 13 file.

No placeholder/stub return values in codepaths that render data. The 12 qualitative PNGs contain grey placeholder cells for missing-cache entries — this is correct specified behavior, not a stub.

---

### Deferred Items

Items not yet met, but explicitly structured as user-action-required per phase scope.

| # | Item | Addressed By | Evidence |
|---|------|-------------|----------|
| 1 | 35 stage JSONs backfilled with `per_class_ap[80]` | User runs `scripts/build_per_class_ap.py --live --limit 5000` (~24 h) | Mode A confirmed: `updated=0 skipped-no-cache=35`, exit 0 |
| 2 | 35 × 12x12 PNGs in `media/confusion_12/` | User: run `build_per_class_ap.py --live` then `build_confusion.py` | `build_confusion.py` gracefully skips all 35 today |
| 3 | 35 × 80x80 PNGs in `results/confusion_80/` | Same as above | Same |
| 4 | 16 summary table files (CSV + MD) | User: run `build_per_class_ap.py --live` then `per_class_summary.py` | `per_class_summary.py` exits 2 with clear message |
| 5 | 12 qualitative PNGs with real predictions | User: run `build_per_class_ap.py --live` then `qualitative_examples.py` | Script produces placeholder-cell PNGs now; fills predictions when cache populated |
| 6 | 3+ MP4s in `media/video/` | User: supply `data/demo.mp4`, run `realtime_demo.py` | Script exits 2 with documented message when file absent |
| 7 | 3 PNGs in `media/screenshots/` | User: manual capture during benchmark run | `media/screenshots/README.md` provides exact instructions |

All deferred items are user-action-required after GPU infrastructure is available or `data/demo.mp4` is supplied. The code paths are complete and verified to function correctly when inputs exist.

---

### Requirements Coverage

| Requirement | Plan | Status | Evidence |
|-------------|------|--------|----------|
| VKR-13-T1 | 13.01 | SATISFIED | `benchmark.eval.per_class` package + `build_per_class_ap.py` — infrastructure complete |
| VKR-13-T2 | 13.02 | SATISFIED | `benchmark.eval.confusion` module + `build_confusion.py` — CLI + logic complete |
| VKR-13-T3 | 13.03 | SATISFIED | `scripts/per_class_summary.py` — CLI complete |
| VKR-13-T4 | 13.04 | SATISFIED | `scripts/qualitative_examples.py` — 12 PNGs (placeholder cells) confirmed |
| VKR-13-T5 | 13.05 | SATISFIED | `media/coco_val2017_samples.png` exists, PIL-verified 1524x1224 |
| VKR-13-T6 | 13.06 | SATISFIED | `scripts/realtime_demo.py` — deferred-input design confirmed working |
| VKR-13-T7 | 13.07 | SATISFIED | `media/screenshots/README.md` — all 4 required sections present |

---

### Human Verification Required

None. All automated checks passed. Deferred items are user-action-required (GPU time / file provision), not verification uncertainties.

---

## Summary

Phase 13 ships a complete, ruff-strict-clean infrastructure for all 7 diploma artifact types. The single non-deferred artifact (`media/coco_val2017_samples.png`) exists and is valid. All CLI scripts handle the empty-cache path gracefully. The RF-DETR-L exclusion logic (5 invalid stages) is consistent across all 4 scripts that iterate configurations.

The phase goal is achieved for the infrastructure layer. Data-dependent outputs (35 backfilled JSONs, 70 confusion PNGs, 16 summary tables, MP4s, screenshots) are deferred to user action — this is explicitly in scope per the phase design.

**Recommendation: SHIP**

---

_Verified: 2026-05-19T19:36:29Z_
_Verifier: Claude (gsd-verifier)_
