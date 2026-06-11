"""Backfill per-class COCO AP into the 35 valid stage JSON reports.

Two operating modes
-------------------
Mode A — ``pure-postprocess`` (default, fast, deterministic):
    For each of the 35 valid (model, stage) configurations, locate
    ``cache/predictions/coco_dt_<model>_<stage>.json`` written by
    :meth:`~benchmark.engines.base.BaseEngine.evaluate_accuracy`.
    If the cache file is present, recompute per-class AP from the cached
    predictions and rewrite the stage JSON in-place, adding ``per_class_ap``.
    If the cache file is missing, log a warning and skip.

Mode B — ``live-eval`` (opt-in via ``--live``, slow, ~24 h for 35 configs):
    Re-run :meth:`evaluate_accuracy` against the on-disk engine for each
    configuration.  This simultaneously populates the cache *and* extracts
    per-class AP.  Use Mode B once end-to-end; subsequent re-runs are Mode A.

Expected workflow
-----------------
1. Run ``scripts/build_per_class_ap.py --live --limit 5000`` once (slow).
2. Verify 35 stage JSONs have ``per_class_ap`` of length 80.
3. Re-run anytime with just ``scripts/build_per_class_ap.py`` (Mode A, fast).
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.cli import (
    MODEL_REGISTRY,
    STAGE_REGISTRY,
    _build_calibration_dataloader,
    _get_adapter,
)
from benchmark.data.coco_loader import COCODataLoader
from benchmark.engines.onnx_engine import OnnxRuntimeEngine
from benchmark.engines.pytorch_engine import PyTorchEngine
from benchmark.engines.tensorrt_engine import TensorRTEngine
from benchmark.eval.per_class import compute_per_class_ap_from_cache

# Determinism convention: set seeds at module load even though this script
# does not use RNG directly (guards against future additions).
np.random.seed(0)
random.seed(0)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

VARIANT_DIRS: dict[str, str] = {
    "rt-detr": "quant",
    "yolo11l": "quant",
    "yolo26l": "quant",
    "rfdetr-l": "rfdetr_v1",
}

# RF-DETR-L INT8/Mixed stages whose TRT auto-tuner rolled back to FP16 —
# excluded from all diploma artifacts.
RFDETR_L_INVALID_STAGES: frozenset[str] = frozenset(
    {
        "5_trt_int8_entropy",
        "5_trt_int8_minmax",
        "5_trt_int8_percentile",
        "6_trt_mixed_a",
        "6_trt_mixed_b",
    }
)

app = typer.Typer(
    name="build-per-class-ap",
    help="Backfill per-class COCO AP into the 35 valid stage JSON reports.",
    add_completion=False,
)


def _is_valid(model: str, stage: str) -> bool:
    """Return False for the 5 defective RF-DETR-L configurations."""
    return not (model == "rfdetr-l" and stage in RFDETR_L_INVALID_STAGES)


def _stage_json_path(results_root: Path, model: str, stage: str) -> Path:
    """Return the on-disk path for a stage JSON report."""
    return results_root / model / VARIANT_DIRS[model] / f"{stage}.json"


def _patch_json(json_path: Path, per_class_ap: list[dict[str, int | float | str]]) -> None:
    """Read stage JSON, set per_class_ap, write back with stable formatting."""
    data: dict[str, object] = json.loads(json_path.read_text(encoding="utf-8"))
    data["per_class_ap"] = per_class_ap
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_mode_a(
    model: str,
    stage: str,
    results_root: Path,
    cache_root: Path,
) -> str:
    """Mode A: recompute per-class AP from cached predictions.

    Returns one of "updated", "skipped-no-cache", "skipped-no-json".
    """
    cache_path = cache_root / f"coco_dt_{model}_{stage}.json"
    json_path = _stage_json_path(results_root, model, stage)

    if not cache_path.exists():
        logger.warning("Cache missing — skipping %s/%s  (run --live to populate)", model, stage)
        return "skipped-no-cache"

    if not json_path.exists():
        logger.warning("Stage JSON missing — skipping %s/%s  (%s)", model, stage, json_path)
        return "skipped-no-json"

    dataloader = COCODataLoader()
    per_class_ap = compute_per_class_ap_from_cache(cache_path, dataloader)
    _patch_json(json_path, per_class_ap)
    logger.info("Updated %s/%s — %d classes", model, stage, len(per_class_ap))
    return "updated"


def _run_mode_b(
    model: str,
    stage: str,
    results_root: Path,
    cache_root: Path,
    limit: int | None,
    engine_dir: Path,
) -> str:
    """Mode B: re-run evaluate_accuracy against on-disk engine, populate cache.

    Returns "updated" on success, "skipped-*" on any error.
    """
    json_path = _stage_json_path(results_root, model, stage)
    if not json_path.exists():
        logger.warning("Stage JSON missing — skipping %s/%s  (%s)", model, stage, json_path)
        return "skipped-no-json"

    cache_root.mkdir(parents=True, exist_ok=True)
    dataloader = COCODataLoader(limit=limit)
    adapter = _get_adapter(model)

    try:
        if stage == "1_pytorch_fp32":
            engine: PyTorchEngine | OnnxRuntimeEngine | TensorRTEngine = PyTorchEngine(
                model_name=model, adapter=adapter  # type: ignore[arg-type]
            )
            engine.load_model(Path(MODEL_REGISTRY[model]["weights"]))

        elif stage == "2_onnx_fp32":
            onnx_path = Path(MODEL_REGISTRY[model]["onnx"])
            if not onnx_path.exists():
                logger.warning("ONNX missing — skipping %s/%s", model, stage)
                return "skipped-missing-engine"
            engine = OnnxRuntimeEngine(
                model_name=model,
                onnx_path=onnx_path,
                adapter=adapter,  # type: ignore[arg-type]
                input_size=(640, 640),
            )
            engine.load_model(onnx_path)

        elif stage in ("3_trt_tf32", "4_trt_fp16", "4_trt_bf16"):
            precision_map = {"3_trt_tf32": "tf32", "4_trt_fp16": "fp16", "4_trt_bf16": "bf16"}
            onnx_path = Path(MODEL_REGISTRY[model]["onnx"])
            engine = TensorRTEngine(
                model_name=model,
                precision=precision_map[stage],
                engine_dir=engine_dir,
                adapter=adapter,  # type: ignore[arg-type]
                force_rebuild=False,
            )
            engine.load_model(onnx_path)

        elif stage in ("5_trt_int8_minmax", "5_trt_int8_entropy", "5_trt_int8_percentile"):
            cal_map = {
                "5_trt_int8_minmax": "minmax",
                "5_trt_int8_entropy": "entropy",
                "5_trt_int8_percentile": "percentile",
            }
            onnx_path = Path(MODEL_REGISTRY[model]["onnx"])
            cal_dl = _build_calibration_dataloader()
            engine = TensorRTEngine(
                model_name=model,
                precision="int8",
                adapter=adapter,  # type: ignore[arg-type]
                calibrator_method=cal_map[stage],  # type: ignore[arg-type]
                engine_dir=engine_dir,
                force_rebuild=False,
            )
            engine.load_model(onnx_path, calibration_dataloader=cal_dl)

        elif stage in ("6_trt_mixed_a", "6_trt_mixed_b"):
            strategy = "a" if stage == "6_trt_mixed_a" else "b"
            onnx_path = Path(MODEL_REGISTRY[model]["onnx"])
            best_cal_file = results_root / model / VARIANT_DIRS[model] / "int8_best_calibrator.json"
            calibrator = "entropy"
            if best_cal_file.exists():
                try:
                    cal_data = json.loads(best_cal_file.read_text(encoding="utf-8"))
                    calibrator = cal_data.get("best_calibrator", "entropy")
                except (json.JSONDecodeError, OSError):
                    pass
            cal_dl = _build_calibration_dataloader()
            engine = TensorRTEngine(
                model_name=model,
                precision="int8",
                adapter=adapter,  # type: ignore[arg-type]
                calibrator_method=calibrator,
                engine_dir=engine_dir,
                force_rebuild=False,
                mixed_strategy=strategy,
            )
            engine.load_model(onnx_path, calibration_dataloader=cal_dl)

        else:
            logger.warning("Unknown stage %s — skipping %s/%s", stage, model, stage)
            return "skipped-unknown-stage"

    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("Engine load failed for %s/%s: %s — skipping", model, stage, exc)
        return "skipped-engine-error"

    accuracy = engine.evaluate_accuracy(dataloader, cache_stage=stage, cache_predictions=True)
    per_class_ap: list[dict[str, int | float | str]] = accuracy.pop("per_class_ap", [])  # type: ignore[assignment]
    _patch_json(json_path, per_class_ap)
    logger.info("Updated %s/%s — %d classes (live eval)", model, stage, len(per_class_ap))
    return "updated"


@app.command()
def main(
    models: Annotated[
        list[str],
        typer.Option("--model", "-m", help="Model(s) to process"),
    ] = ["rt-detr", "yolo11l", "yolo26l", "rfdetr-l"],  # noqa: B006
    stages: Annotated[
        list[str] | None,
        typer.Option("--stage", "-s", help="Stage ID(s) to process (default: all)"),
    ] = None,
    results_root: Annotated[
        Path,
        typer.Option("--results-root", help="Root directory of results"),
    ] = Path("results"),
    cache_root: Annotated[
        Path,
        typer.Option("--cache-root", help="Directory containing cached COCO prediction JSONs"),
    ] = Path("cache/predictions"),
    live: Annotated[
        bool,
        typer.Option("--live", help="Re-run evaluate_accuracy against on-disk engines (Mode B)"),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="COCO image limit for live eval (None = all 5000)"),
    ] = None,
    engine_dir: Annotated[
        Path,
        typer.Option("--engine-dir", help="Directory containing TRT .engine files"),
    ] = Path("engines"),
) -> None:
    """Backfill per_class_ap into 35 valid stage JSON reports.

    Default: Mode A (reads cache, no GPU needed).
    With --live: Mode B (re-runs inference, ~24 h for all 35 configs).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    active_stages = stages if stages is not None else STAGE_REGISTRY

    counts: dict[str, int] = {"updated": 0, "skipped-no-cache": 0, "skipped-no-json": 0, "other": 0}

    for model in models:
        if model not in VARIANT_DIRS:
            logger.warning("Unknown model '%s' — skipping", model)
            continue
        for stage in active_stages:
            if not _is_valid(model, stage):
                logger.info("Excluded config %s/%s (RF-DETR-L invalid stage)", model, stage)
                continue
            if live:
                status = _run_mode_b(model, stage, results_root, cache_root, limit, engine_dir)
            else:
                status = _run_mode_a(model, stage, results_root, cache_root)

            if status in counts:
                counts[status] += 1
            else:
                counts["other"] += 1

    typer.echo(
        f"\nDone. updated={counts['updated']}  "
        f"skipped-no-cache={counts['skipped-no-cache']}  "
        f"skipped-no-json={counts['skipped-no-json']}  "
        f"other={counts['other']}"
    )


if __name__ == "__main__":
    app()
