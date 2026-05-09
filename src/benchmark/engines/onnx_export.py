"""ONNX export and optimization pipeline.

Converts PyTorch models to ONNX format with onnx-simplifier
optimization for subsequent TensorRT conversion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import onnx
import onnxsim
import torch

if TYPE_CHECKING:
    from pathlib import Path

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
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
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
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
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
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
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
    )

    return simplify_onnx(raw_path, output_path=raw_path)
