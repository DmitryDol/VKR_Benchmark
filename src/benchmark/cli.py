"""Typer CLI entry point for the VKR benchmark pipeline.

Commands:
    benchmark run --model MODEL --stage STAGE [--limit N] [--output-dir PATH]
    benchmark run --model MODEL --all-stages [--limit N] [--output-dir PATH]
    benchmark merge --model MODEL [--output-dir PATH]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from benchmark.data.coco_loader import COCODataLoader
from benchmark.engines.onnx_engine import OnnxRuntimeEngine
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

# CLI-01 / CLI-02 stage registry — ordered list for --all-stages
# Stage 1 and 2 only in Phase 2; later phases append stages 3-6
STAGE_REGISTRY: list[str] = [
    "1_pytorch_fp32",
    "2_onnx_fp32",
    "3_trt_tf32",
    "4_trt_fp16",
    "4_trt_bf16",
]

# Model registry — maps CLI name to weights directory and ONNX path
# Extended in later phases when additional adapters are added
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "rt-detr": {
        "weights": "weights/rtdetr-r50/",
        "onnx": "weights/rtdetr-r50/rtdetr_r50_sim.onnx",
        "family": "detr",  # routes MACs computation
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
    msg = f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
    raise typer.BadParameter(msg)


def _run_stage(
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

    if stage == "1_pytorch_fp32":
        adapter = _get_adapter(model_name)
        engine = PyTorchEngine(model_name=model_name, adapter=adapter)  # type: ignore[arg-type]
        weights_path = Path(MODEL_REGISTRY[model_name]["weights"])
        engine.load_model(weights_path)

        # D-09: compute MACs once at stage 1
        if macs is None:
            macs, flops = compute_macs(
                engine.model,
                model_name,
                input_shape=(1, 3, 640, 640),
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
            logging.warning(
                "ONNX model not found at %s — run stage 1 first to export it",
                onnx_path,
            )
            msg = f"ONNX model missing: {onnx_path}"
            raise FileNotFoundError(msg)

        engine_onnx = OnnxRuntimeEngine(
            model_name=model_name,
            onnx_path=onnx_path,
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
        typer.Option("--stage", help="Stage ID, e.g. 1_pytorch_fp32"),
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

    stages_to_run = STAGE_REGISTRY if all_stages else [stage]  # type: ignore[list-item]

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
        except Exception as exc:
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
