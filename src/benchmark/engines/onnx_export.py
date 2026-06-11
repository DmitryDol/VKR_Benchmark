"""ONNX export and optimization pipeline.

Converts PyTorch models to ONNX format with onnx-simplifier
optimization for subsequent TensorRT conversion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import onnx
import onnxsim
import torch

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Default dynamic axes: batch dimension only, fixed spatial dims for TensorRT
DEFAULT_DYNAMIC_AXES: dict[str, dict[int, str]] = {
    "input": {0: "batch"},
    "output": {0: "batch"},
}


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 18,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Parameters
    ----------
    model : nn.Module
        PyTorch model in eval mode.
    output_path : Path
        Destination path for the .onnx file.
    input_size : tuple[int, int]
        Model input resolution (height, width).
    opset_version : int
        ONNX opset version (17+ recommended for transformers).
    dynamic_axes : dict | None
        Dynamic axis specification. Defaults to batch-only.

    Returns
    -------
    Path
        Path to the exported ONNX model.
    """
    if dynamic_axes is None:
        dynamic_axes = DEFAULT_DYNAMIC_AXES

    _input_names = input_names if input_names is not None else ["input"]
    _output_names = output_names if output_names is not None else ["output"]

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, *input_size, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting to ONNX: %s (opset=%d, input=%s)",
        output_path,
        opset_version,
        input_size,
    )

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=_input_names,
            output_names=_output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,  # Force legacy TorchScript backend; required for TensorRT
        )

    validate_onnx(output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("ONNX export complete: %.1f MB", size_mb)

    return output_path


def simplify_onnx(
    model_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Simplify an ONNX model using onnx-simplifier.

    Parameters
    ----------
    model_path : Path
        Path to the input ONNX model.
    output_path : Path | None
        Destination for simplified model. Defaults to ``{stem}_sim.onnx``.

    Returns
    -------
    Path
        Path to the simplified ONNX model.
    """
    if output_path is None:
        output_path = model_path.with_stem(f"{model_path.stem}_sim")

    logger.info("Simplifying ONNX model: %s", model_path)

    model = onnx.load(str(model_path))
    simplified, check_ok = onnxsim.simplify(model)

    if not check_ok:
        logger.warning("onnxsim validation failed for %s — saving anyway", model_path)

    onnx.save(simplified, str(output_path))

    original_mb = model_path.stat().st_size / (1024 * 1024)
    simplified_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Simplified: %.1f MB → %.1f MB (%.1f%% reduction)",
        original_mb,
        simplified_mb,
        (1.0 - simplified_mb / original_mb) * 100.0 if original_mb > 0 else 0.0,
    )

    return output_path


def validate_onnx(model_path: Path) -> bool:
    """Validate an ONNX model using onnx.checker.

    Parameters
    ----------
    model_path : Path
        Path to the ONNX model file.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    onnx.checker.ValidationError
        If the model is invalid.
    """
    onnx.checker.check_model(str(model_path))
    logger.info("ONNX validation passed: %s", model_path)
    return True


def export_and_simplify(
    model: nn.Module,
    output_path: Path,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 18,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> Path:
    """Export PyTorch model to ONNX and run onnx-simplifier.

    Convenience function that chains ``export_to_onnx`` and
    ``simplify_onnx``. The simplified model replaces the original.

    Returns
    -------
    Path
        Path to the simplified ONNX model.
    """
    raw_path = export_to_onnx(
        model,
        output_path,
        input_size=input_size,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
    )

    return simplify_onnx(raw_path, output_path=raw_path)


def export_yolo_to_onnx(
    weights_path: Path,
    output_path: Path,
    opset_version: int = 17,
) -> Path:
    """Export a YOLO model to simplified ONNX using the ultralytics exporter.

    Uses the ultralytics ``YOLO.export()`` method (which handles YOLO-specific
    graph quirks reliably) with ``simplify=False``, then runs the project's own
    ``simplify_onnx()`` step for consistent graph optimization across all models
    (D-01).  Batch size is fixed at 1 (``dynamic=False``); opset 17 is used per
    D-02.

    Parameters
    ----------
    weights_path : Path
        Path to the ``.pt`` YOLO weights file.
    output_path : Path
        Destination path for the simplified ``_sim.onnx`` file.
    opset_version : int
        ONNX opset version (default 17, per D-02).

    Returns
    -------
    Path
        Path to the simplified ONNX file (equals ``output_path``).

    Raises
    ------
    RuntimeError
        If the ultralytics export step fails or produces no output file.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    logger.info("Exporting YOLO model to ONNX: %s (opset=%d)", weights_path, opset_version)

    yolo = YOLO(str(weights_path))

    # ultralytics .export() returns the path to the raw ONNX file as a string.
    # simplify=False — the project runs its own onnxsim step below (D-01).
    # dynamic=False — batch=1 fixed; no dynamic axes needed for TRT (D-02).
    raw_onnx_str: str | None = yolo.export(
        format="onnx",
        simplify=False,
        opset=opset_version,
        dynamic=False,
    )

    if not raw_onnx_str:
        msg = f"ultralytics YOLO.export() returned no output path for {weights_path}"
        raise RuntimeError(msg)

    raw_onnx_path = Path(raw_onnx_str)
    logger.info("ultralytics ONNX export complete: %s", raw_onnx_path)

    # Project onnxsim step — consistent graph optimization for all models (D-01).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sim_path = simplify_onnx(raw_onnx_path, output_path=output_path)

    # validate the final simplified graph before handing to downstream stages.
    validate_onnx(sim_path)

    logger.info("YOLO simplified ONNX ready: %s", sim_path)
    return sim_path
