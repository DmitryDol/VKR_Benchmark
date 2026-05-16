from benchmark.engines.base import BaseEngine
from benchmark.engines.onnx_engine import OnnxRuntimeEngine
from benchmark.engines.pytorch_engine import PyTorchEngine

__all__ = [
    "BaseEngine",
    "EntropyCalibrator",
    "MinMaxCalibrator",
    "OnnxRuntimeEngine",
    "PercentileCalibrator",
    "PyTorchEngine",
    "TensorRTEngine",
    "apply_strategy_a",
    "apply_strategy_b",
]


def __getattr__(name: str) -> object:
    """Lazy-load TensorRT-dependent symbols on first access.

    ``tensorrt`` is an optional dependency (installed separately via
    ``pip install tensorrt``). Importing it at module level would break all
    tests in environments where TensorRT is not installed. By deferring to
    ``__getattr__``, the package remains importable without TensorRT and only
    raises ``ImportError`` when a TRT-specific symbol is actually used.
    """
    if name in ("TensorRTEngine",):
        from benchmark.engines.tensorrt_engine import TensorRTEngine  # noqa: PLC0415

        return TensorRTEngine
    if name in ("MinMaxCalibrator", "EntropyCalibrator", "PercentileCalibrator"):
        from benchmark.engines.int8_calibrators import (  # noqa: PLC0415
            EntropyCalibrator,
            MinMaxCalibrator,
            PercentileCalibrator,
        )

        if name == "MinMaxCalibrator":
            return MinMaxCalibrator
        if name == "EntropyCalibrator":
            return EntropyCalibrator
        return PercentileCalibrator
    if name in ("apply_strategy_a", "apply_strategy_b"):
        from benchmark.engines.mixed_precision import (  # noqa: PLC0415
            apply_strategy_a,
            apply_strategy_b,
        )

        if name == "apply_strategy_a":
            return apply_strategy_a
        return apply_strategy_b
    msg = f"module 'benchmark.engines' has no attribute '{name}'"
    raise AttributeError(msg)
