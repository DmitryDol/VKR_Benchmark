"""MACs/FLOPs computation for benchmarked models.

Strategy (D-07):
  - DETR family (rt-detr, rf-detr, d-fine, deimv2): calflops.calculate_flops()
  - YOLO family (yolo11, yolo26): model.info() native Ultralytics method

MACs computed once at stage 1 (PyTorch), reused for stages 2-6 (D-09).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

try:
    from calflops import calculate_flops as _calculate_flops  # type: ignore[import-untyped]

    _CALFLOPS_AVAILABLE = True
except ImportError:
    _CALFLOPS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Model families that use calflops (HuggingFace / PyTorch-native DETR architectures)
_DETR_FAMILY: frozenset[str] = frozenset({"rt-detr", "rf-detr", "d-fine", "deimv2"})

# Model families that use native Ultralytics model.info()
_YOLO_FAMILY: frozenset[str] = frozenset({"yolo11", "yolo26"})


def compute_macs(
    model: nn.Module,
    model_name: str,
    input_shape: tuple[int, int, int, int] = (1, 3, 640, 640),
) -> tuple[float, float]:
    """Compute MACs and FLOPs for the given model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model in eval mode. Must support forward pass with the
        given input_shape.
    model_name : str
        Lowercase model identifier (e.g. "rt-detr"). Used to select the
        computation strategy (D-07).
    input_shape : tuple[int, int, int, int]
        (batch, channels, height, width) — always batch=1 per CLAUDE.md.

    Returns
    -------
    tuple[float, float]
        (macs, flops) — both as raw float counts (not GMACs).
        Returns (0.0, 0.0) on failure with a logged warning.

    Notes
    -----
    If calflops reports 0 MACs for any sub-operation (e.g.,
    MultiScaleDeformableAttention C++ extension), a warning is emitted
    per D-08. The returned values may be underestimated in that case.
    """
    normalized = model_name.lower()

    if normalized in _YOLO_FAMILY:
        return _compute_macs_yolo(model, model_name)
    if normalized in _DETR_FAMILY:
        return _compute_macs_calflops(model, model_name, input_shape)

    # Unknown family — attempt calflops with warning
    logger.warning(
        "compute_macs: unknown model family '%s' — attempting calflops, "
        "results may be inaccurate",
        model_name,
    )
    return _compute_macs_calflops(model, model_name, input_shape)


def _compute_macs_calflops(
    model: nn.Module,
    model_name: str,
    input_shape: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Compute MACs using calflops.calculate_flops()."""
    if not _CALFLOPS_AVAILABLE:
        logger.warning(
            "calflops not installed — MACs will be 0.0 for %s. "
            "Run: uv add calflops",
            model_name,
        )
        return 0.0, 0.0

    try:
        flops_obj, macs_obj, _ = _calculate_flops(
            model=model,
            input_shape=input_shape,
            output_as_string=False,
            output_precision=6,
            print_results=False,
        )
        macs = float(macs_obj)
        flops = float(flops_obj)

        # D-08: warn if any component reports 0 (unsupported ops, e.g.,
        # MultiScaleDeformableAttention C++ extension in DETR variants)
        if macs == 0.0:
            logger.warning(
                "calflops: unsupported ops detected — MACs may be "
                "underestimated for %s (MultiScaleDeformableAttention?)",
                model_name,
            )

        logger.info("MACs=%.3e FLOPs=%.3e for %s", macs, flops, model_name)
        return macs, flops

    except Exception:  # calflops raises varied exceptions; failure must not abort benchmark
        logger.warning("calflops failed for %s — MACs will be 0.0", model_name)
        return 0.0, 0.0


def _compute_macs_yolo(model: nn.Module, model_name: str) -> tuple[float, float]:
    """Extract MACs from Ultralytics native model.info()."""
    try:
        # Ultralytics models expose .info() which returns (layers, params, gradients, flops)
        # flops is in GFLOPs; multiply by 1e9 for raw count
        info = model.info(verbose=False)  # type: ignore[attr-defined]
        # Ultralytics returns (n_layers, n_params, n_gradients, gflops)
        gflops = float(info[3])
        flops = gflops * 1e9
        macs = flops / 2.0  # FLOPs ~= 2x MACs for conv layers
        logger.info("MACs=%.3e FLOPs=%.3e for %s (via model.info)", macs, flops, model_name)
        return macs, flops
    except (AttributeError, TypeError, IndexError):
        logger.warning(
            "model.info() unavailable or unexpected format for %s — MACs will be 0.0",
            model_name,
        )
        return 0.0, 0.0
