"""Typer CLI entry point for the VKR benchmark pipeline.

Commands:
    benchmark run --model MODEL --stage STAGE [--limit N] [--output-dir PATH]
    benchmark run --model MODEL --all-stages [--limit N] [--output-dir PATH]
    benchmark merge --model MODEL [--output-dir PATH]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from benchmark.data.coco_loader import COCODataLoader
from benchmark.engines.onnx_engine import OnnxRuntimeEngine
from benchmark.engines.onnx_export import export_yolo_to_onnx
from benchmark.engines.pytorch_engine import PyTorchEngine
from benchmark.engines.tensorrt_engine import TensorRTEngine
from benchmark.utils.hardware import HardwareInfo
from benchmark.utils.logger import ResultLogger
from benchmark.utils.macs import compute_macs

app = typer.Typer(
    name="benchmark",
    help="VKR transformer object detection benchmark pipeline.",
    add_completion=False,
)

# Phase 7 D-07 / D-08: the INT8 calibration set is fixed at 500 COCO val2017 images
# and is the SAME 500 across all three calibrators (MinMax, Entropy, Percentile) and
# across both Stage 5 and Stage 6 — the calibrator algorithm (or mixed-precision
# strategy) must be the only variable. COCODataLoader is deterministic by
# construction: `__post_init__` does `sorted(self._coco.getImgIds())[:limit]` (no
# shuffle, no seed), so `COCODataLoader(limit=500)` returns the identical 500
# image_ids in the identical order on every construction. This module-level
# constant + helper centralizes that contract so Stages 5 and 6 share one
# call-site (Plan 07-03, Task 1).
_CALIBRATION_IMAGE_COUNT: int = 500


def _build_calibration_dataloader() -> COCODataLoader:
    """Build the single canonical calibration dataloader (Plan 07-03, D-07/D-08).

    Returns a :class:`COCODataLoader` limited to the first
    ``_CALIBRATION_IMAGE_COUNT`` image_ids in COCO's stable sorted order — the
    same 500 images, deterministically, used by all three INT8 calibrators and by
    the Stage 6 mixed-precision rebuilds.
    """
    return COCODataLoader(limit=_CALIBRATION_IMAGE_COUNT)

# CLI-01 / CLI-02 stage registry — ordered list for --all-stages
# Stage 1 and 2 only in Phase 2; later phases append stages 3-6
STAGE_REGISTRY: list[str] = [
    "1_pytorch_fp32",
    "2_onnx_fp32",
    "3_trt_tf32",
    "4_trt_fp16",
    "4_trt_bf16",
    "5_trt_int8_minmax",
    "5_trt_int8_entropy",
    "5_trt_int8_percentile",
    "6_trt_mixed_a",
    "6_trt_mixed_b",
]

# Model registry — maps CLI name to weights directory and ONNX path
# Extended in later phases when additional adapters are added
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "rt-detr": {
        "weights": "weights/rtdetr-r50vd/",
        "onnx": "weights/rtdetr-r50vd/rtdetr_r50_sim.onnx",
        "family": "detr",  # routes MACs computation
    },
    "yolo11l": {
        "weights": "weights/yolo11l/yolo11l.pt",
        "onnx": "weights/yolo11l/yolo11l_sim.onnx",
        "family": "yolo",
    },
    "yolo26l": {
        "weights": "weights/yolo26l/yolo26l.pt",
        "onnx": "weights/yolo26l/yolo26l_sim.onnx",
        "family": "yolo",
    },
    "rfdetr-l": {
        "weights": "weights/rfdetr-l/",
        "onnx": "weights/rfdetr-l/rfdetr_l_sim.onnx",
        "family": "rfdetr",
    },
}


def _configure_logging() -> None:
    """Configure root logger at INFO level (D-16)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _get_adapter(model_name: str) -> object:
    """Return the ModelAdapter instance for the given model name.

    Raises
    ------
    typer.BadParameter
        If model_name is not in MODEL_REGISTRY.
    """
    if model_name == "rt-detr":
        # RTDETRAdapter lives in benchmark.models, added in a later phase
        try:
            from benchmark.models.rtdetr_adapter import RTDETRAdapter  # type: ignore[import-not-found]  # noqa: PLC0415 I001
        except ImportError as exc:
            msg = f"RTDETRAdapter not available: {exc}"
            raise typer.BadParameter(msg) from exc
        return RTDETRAdapter()

    if model_name in ("yolo11l", "yolo26l"):
        from benchmark.models.yolo_adapter import YOLOAdapter  # noqa: PLC0415

        is_nms_free = model_name == "yolo26l"
        return YOLOAdapter(is_nms_free=is_nms_free)

    if model_name == "rfdetr-l":
        from benchmark.models.rfdetr_adapter import RFDETRAdapter  # noqa: PLC0415

        return RFDETRAdapter()

    msg = f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
    raise typer.BadParameter(msg)


def _run_stage(  # noqa: PLR0912, PLR0915
    model_name: str,
    stage: str,
    limit: int | None,
    result_logger: ResultLogger,
    baseline_map: float,
    macs: float | None,
    flops: float | None,
    engine_dir: Path = Path("engines"),
    force_rebuild: bool = False,
) -> tuple[float, float, float]:
    """Execute a single benchmark stage. Returns (map_50_95, macs, flops)."""
    dataloader = COCODataLoader(limit=limit)
    adapter = _get_adapter(model_name)

    if stage == "1_pytorch_fp32":
        engine = PyTorchEngine(model_name=model_name, adapter=adapter)  # type: ignore[arg-type]
        weights_path = Path(MODEL_REGISTRY[model_name]["weights"])
        engine.load_model(weights_path)

        # D-09: compute MACs once at stage 1 — read input shape from adapter so
        # non-640 models (e.g. RF-DETR @ 704x704) produce correct FLOPs.
        if macs is None:
            h, w = adapter.input_size  # type: ignore[union-attr]
            macs, flops = compute_macs(
                engine.model,
                model_name,
                input_shape=(1, 3, h, w),
            )

        result = engine.run_full_benchmark(
            dataloader,
            stage=stage,
            baseline_map_50_95=baseline_map,
            macs=macs,
            flops=flops,
        )

    elif stage == "2_onnx_fp32":
        onnx_path = Path(MODEL_REGISTRY[model_name]["onnx"])
        if not onnx_path.exists():
            if MODEL_REGISTRY[model_name].get("family") == "yolo":
                logging.info(
                    "ONNX model not found at %s — exporting from .pt weights on demand",
                    onnx_path,
                )
                export_yolo_to_onnx(
                    weights_path=Path(MODEL_REGISTRY[model_name]["weights"]),
                    output_path=onnx_path,
                )
            else:
                logging.warning(
                    "ONNX model not found at %s — run stage 1 first to export it",
                    onnx_path,
                )
                msg = f"ONNX model missing: {onnx_path}"
                raise FileNotFoundError(msg)

        engine_onnx = OnnxRuntimeEngine(
            model_name=model_name,
            onnx_path=onnx_path,
            adapter=adapter,  # type: ignore[arg-type]
            input_size=(640, 640),
        )
        engine_onnx.load_model(onnx_path)  # weights_path ignored by OnnxRuntimeEngine

        result = engine_onnx.run_full_benchmark(
            dataloader,
            stage=stage,
            baseline_map_50_95=baseline_map,
            macs=macs,  # reuse from stage 1 (D-09)
            flops=flops,
        )

    elif stage in ("3_trt_tf32", "4_trt_fp16", "4_trt_bf16"):
        precision_map = {
            "3_trt_tf32": "tf32",
            "4_trt_fp16": "fp16",
            "4_trt_bf16": "bf16",
        }
        precision = precision_map[stage]
        onnx_path = Path(MODEL_REGISTRY[model_name]["onnx"])
        if not onnx_path.exists():
            msg = f"ONNX model missing: {onnx_path} — run stage 2 first"
            raise FileNotFoundError(msg)

        engine = TensorRTEngine(
            model_name=model_name,
            precision=precision,
            engine_dir=engine_dir,
            adapter=adapter,  # type: ignore[arg-type]
            force_rebuild=force_rebuild,
        )
        engine.load_model(onnx_path)

        result = engine.run_full_benchmark(
            dataloader,
            stage=stage,
            baseline_map_50_95=baseline_map,
            macs=macs,
            flops=flops,
        )

    elif stage in ("5_trt_int8_minmax", "5_trt_int8_entropy", "5_trt_int8_percentile"):
        cal_method_map = {
            "5_trt_int8_minmax": "minmax",
            "5_trt_int8_entropy": "entropy",
            "5_trt_int8_percentile": "percentile",
        }
        cal_method = cal_method_map[stage]
        onnx_path = Path(MODEL_REGISTRY[model_name]["onnx"])
        if not onnx_path.exists():
            msg = f"ONNX model missing: {onnx_path} — run stage 2 first"
            raise FileNotFoundError(msg)

        # D-07/D-08: single shared 500-image calibration dataloader (see
        # _build_calibration_dataloader docstring). MUST be the identical call for
        # both Stage 5 and Stage 6 so the calibrator algorithm is the only variable.
        cal_dataloader = _build_calibration_dataloader()

        engine = TensorRTEngine(
            model_name=model_name,
            precision="int8",
            adapter=adapter,  # type: ignore[arg-type]
            calibrator_method=cal_method,  # type: ignore[arg-type]
            engine_dir=engine_dir,
            force_rebuild=force_rebuild,
        )
        engine.load_model(onnx_path, calibration_dataloader=cal_dataloader)

        result = engine.run_full_benchmark(
            dataloader,
            stage=stage,
            baseline_map_50_95=baseline_map,
            macs=macs,
            flops=flops,
        )

    elif stage in ("6_trt_mixed_a", "6_trt_mixed_b"):
        strategy = "a" if stage == "6_trt_mixed_a" else "b"
        onnx_path = Path(MODEL_REGISTRY[model_name]["onnx"])
        if not onnx_path.exists():
            msg = f"ONNX model missing: {onnx_path} — run stage 2 first"
            raise FileNotFoundError(msg)

        best_calibrator_file = (
            result_logger.output_dir
            / model_name
            / result_logger.run_id
            / "int8_best_calibrator.json"
        )
        if best_calibrator_file.exists():
            try:
                cal_data = json.loads(best_calibrator_file.read_text(encoding="utf-8"))
                calibrator = cal_data.get("best_calibrator", "entropy")
            except (json.JSONDecodeError, OSError):
                logging.warning("Failed to parse int8_best_calibrator.json, fallback to entropy")
                calibrator = "entropy"
        else:
            logging.warning("int8_best_calibrator.json missing, fallback to entropy")
            calibrator = "entropy"

        # D-07/D-08: same 500-image calibration set as Stage 5 (see helper docstring)
        cal_dataloader = _build_calibration_dataloader()

        engine = TensorRTEngine(
            model_name=model_name,
            precision="int8",
            adapter=adapter,  # type: ignore[arg-type]
            calibrator_method=calibrator,
            engine_dir=engine_dir,
            force_rebuild=force_rebuild,
            mixed_strategy=strategy,
        )
        engine.load_model(onnx_path, calibration_dataloader=cal_dataloader)

        result = engine.run_full_benchmark(
            dataloader,
            stage=stage,
            baseline_map_50_95=baseline_map,
            macs=macs,
            flops=flops,
        )

    else:
        msg = f"Stage '{stage}' not implemented. Available: {STAGE_REGISTRY}"
        raise typer.BadParameter(msg)

    result_logger.add(result)
    result_logger.save_stage_files(result)
    return result.map_50_95, macs or 0.0, flops or 0.0


@app.command("run")
def run_benchmark(
    model: Annotated[str, typer.Option("--model", help="Model name (e.g. rt-detr)")] = "rt-detr",
    stage: Annotated[
        str | None,
        typer.Option(
            "--stage",
            help=(
                "Stage ID or comma-separated list of stage IDs, "
                "e.g. 5_trt_int8_minmax,5_trt_int8_entropy,5_trt_int8_percentile"
            ),
        ),
    ] = None,
    all_stages: Annotated[
        bool,
        typer.Option("--all-stages", help="Run all registered stages sequentially"),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Limit COCO images for dev runs"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Results output directory"),
    ] = Path("results"),
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Run ID to resume (auto-generated timestamp if omitted)"),
    ] = "",
    force_rebuild: Annotated[
        bool,
        typer.Option("--force-rebuild", help="Force TRT engine rebuild even if cached"),
    ] = False,
    engine_dir: Annotated[
        Path,
        typer.Option("--engine-dir", help="Directory to cache TRT .engine files"),
    ] = Path("engines"),
) -> None:
    """Run benchmark for a model (CLI-01 + CLI-02, D-13)."""
    _configure_logging()

    if model not in MODEL_REGISTRY:
        msg = f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY)}"
        raise typer.BadParameter(msg)
    if not all_stages and stage is None:
        raise typer.BadParameter("Provide --stage STAGE_NAME or --all-stages")
    if all_stages and stage is not None:
        raise typer.BadParameter("Cannot use both --stage and --all-stages")

    # T-02-05: resolve output_dir to avoid path traversal via shell expansion
    resolved_dir = Path(output_dir).resolve()
    # T-03-06: resolve engine_dir to avoid path traversal
    engine_dir = engine_dir.resolve()

    # D-03: collect hardware info once at startup
    hw = HardwareInfo.collect()
    result_logger = ResultLogger(output_dir=resolved_dir, hardware=hw, run_id=run_id)
    typer.echo(f"Run ID: {result_logger.run_id}")

    if all_stages:
        stages_to_run: list[str] = STAGE_REGISTRY
    else:
        # Support comma-separated stage IDs: --stage a,b,c
        parsed = [s.strip() for s in stage.split(",") if s.strip()]  # type: ignore[union-attr]
        unknown = [s for s in parsed if s not in STAGE_REGISTRY]
        if unknown:
            msg = f"Unknown stage(s): {unknown}. Available: {STAGE_REGISTRY}"
            raise typer.BadParameter(msg)
        stages_to_run = parsed

    baseline_map: float = 0.0
    macs: float | None = None
    flops: float | None = None

    for s in stages_to_run:
        typer.echo(f"--- Running stage: {s} ---")
        try:
            map_result, macs, flops = _run_stage(
                model_name=model,
                stage=s,
                limit=limit,
                result_logger=result_logger,
                baseline_map=baseline_map,
                macs=macs,
                flops=flops,
                engine_dir=engine_dir,
                force_rebuild=force_rebuild,
            )
            # First stage sets the baseline for accuracy_drop_pct
            if s == "1_pytorch_fp32":
                baseline_map = map_result
            # After any INT8 stage, update best-calibrator summary (CAL-05)
            if s in ("5_trt_int8_minmax", "5_trt_int8_entropy", "5_trt_int8_percentile"):
                result_logger.save_int8_best_calibrator(model)
        except Exception as exc:
            logging.getLogger(__name__).exception("Stage %s failed", s)
            typer.echo(f"Stage {s} failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    typer.echo("All stages complete.")


@app.command("merge")
def merge_results(
    model: Annotated[str, typer.Option("--model", help="Model name (e.g. rt-detr)")] = "rt-detr",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Results directory containing per-stage files"),
    ] = Path("results"),
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Run ID to merge (required if multiple runs exist)"),
    ] = "",
) -> None:
    """Merge per-stage CSVs into unified results files (CLI-03, D-06)."""
    _configure_logging()

    # T-02-05: resolve to avoid path traversal
    resolved_dir = Path(output_dir).resolve()

    result_logger = ResultLogger(output_dir=resolved_dir, run_id=run_id)
    try:
        csv_path, json_path = result_logger.merge_to_unified(model)
        typer.echo(f"Merged: {csv_path}, {json_path}")
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
