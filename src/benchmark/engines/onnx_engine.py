"""ONNX Runtime inference engine for stage 2 benchmarking."""

from __future__ import annotations

import logging
import os
import site
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
from PIL import Image

from benchmark.engines.base import BaseEngine, Detection

if TYPE_CHECKING:
    from benchmark.data.coco_loader import COCOSample
    from benchmark.engines.pytorch_engine import ModelAdapter

# Minimum number of output tensors expected from RT-DETR ONNX model
_MIN_ONNX_OUTPUTS: int = 2

logger = logging.getLogger(__name__)

_cuda_dll_dirs_registered = False


def _register_cuda_dll_dirs() -> None:
    """Add the pip-installed CUDA 12 runtime libs to the Windows DLL search path.

    onnxruntime-gpu 1.26 is built against CUDA 12.x, but the project's torch
    ships CUDA 13.x — so ORT's CUDA EP cannot use torch's bundled libs. The
    CUDA 12 runtime/cuDNN come from the ``nvidia-*-cu12`` pip packages, which
    land in ``site-packages/nvidia/*/bin``. ORT does not auto-discover them on
    Windows, so register them here. Both ``os.add_dll_directory`` AND a PATH
    prepend are required: ``cudnn64_9.dll`` is a thin loader that pulls its
    sub-DLLs (``cudnn_graph64_9.dll`` etc.) via the legacy PATH search.
    """
    global _cuda_dll_dirs_registered  # noqa: PLW0603
    if _cuda_dll_dirs_registered or sys.platform != "win32":
        return

    bin_dirs: list[str] = []
    for site_dir in site.getsitepackages():
        nvidia_root = Path(site_dir) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        bin_dirs.extend(str(p) for p in nvidia_root.glob("*/bin") if p.is_dir())

    if not bin_dirs:
        logger.warning(
            "ONNX CUDA EP: no nvidia-*-cu12 package bin dirs found — "
            "install them with `uv pip install nvidia-cuda-runtime-cu12 "
            "nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12`"
        )
        return

    for d in bin_dirs:
        os.add_dll_directory(d)
    os.environ["PATH"] = os.pathsep.join([*bin_dirs, os.environ.get("PATH", "")])
    _cuda_dll_dirs_registered = True
    logger.info("ONNX CUDA EP: registered %d CUDA 12 DLL dir(s)", len(bin_dirs))


class OnnxRuntimeEngine(BaseEngine):
    """ONNX Runtime GPU inference engine (stage 2: ONNX FP32).

    Uses the CUDA ExecutionProvider when available, falls back to CPU
    with a warning per T-02-06.

    Parameters
    ----------
    model_name : str
        Human-readable model identifier (e.g. "rt-detr-l").
    onnx_path : Path
        Path to the simplified .onnx model file.
    input_size : tuple[int, int]
        Model input resolution (height, width). Must match ONNX model.
    score_threshold : float
        Minimum detection confidence for postprocessing.
    """

    def __init__(
        self,
        model_name: str,
        onnx_path: Path,
        adapter: ModelAdapter,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.001,
    ) -> None:
        super().__init__(model_name, engine_type="onnx", precision="fp32")
        self._adapter = adapter
        self._onnx_path = onnx_path
        self._input_size = input_size
        self._score_threshold = score_threshold
        self._session: ort.InferenceSession | None = None

    def load_model(self, weights_path: Path) -> None:  # noqa: ARG002
        """Create OnnxRuntime InferenceSession with CUDA EP if available.

        The weights_path argument is ignored — this engine uses the
        onnx_path passed at construction. The parameter is kept to
        satisfy the BaseEngine abstract method contract.

        T-02-06 mitigation: check available providers; fall back to CPU
        with a logged warning if CUDA EP is unavailable.
        """
        _register_cuda_dll_dirs()
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("OnnxRuntimeEngine: using CUDA ExecutionProvider")
        else:
            providers = ["CPUExecutionProvider"]
            logger.warning(
                "OnnxRuntimeEngine: CUDA ExecutionProvider unavailable — "
                "falling back to CPU. Available: %s",
                available,
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self._onnx_path), sess_options=opts, providers=providers
        )
        active = self._session.get_providers()
        if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in active:
            logger.warning(
                "OnnxRuntimeEngine: CUDA EP requested but ORT fell back to %s — "
                "stage 2 latency will reflect CPU inference",
                active,
            )
        logger.info(
            "ONNX session loaded: %s (%.1f MB) — active provider: %s",
            self._onnx_path.name,
            self.model_size_mb,
            active[0],
        )

    def preprocess(self, sample: COCOSample) -> np.ndarray:
        """Resize image and convert to model input tensor using adapter.

        If the adapter exposes a ``preprocess`` method (e.g. YOLO letterbox),
        delegate to it and return the tensor as a CPU numpy array. Otherwise
        fall back to the generic stretch-resize used by RT-DETR.

        Returns (1, 3, H, W) float32 numpy array.
        """
        adapter_pre = getattr(self._adapter, "preprocess", None)
        if callable(adapter_pre):
            tensor = adapter_pre(sample, device=None)
            return tensor.cpu().numpy().astype(np.float32)

        h, w = self._adapter.input_size
        img = Image.fromarray(sample.image).resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0, 1]
        arr = arr.transpose(2, 0, 1)  # CHW
        return arr[np.newaxis, ...]  # (1, 3, H, W)

    def infer(self, inputs: object) -> object:
        """Run ONNX Runtime inference.

        Parameters
        ----------
        inputs : object
            Expected: np.ndarray of shape (1, 3, H, W) float32.
        """
        if self._session is None:
            msg = "Model not loaded. Call load_model() first."
            raise RuntimeError(msg)
        input_name = self._session.get_inputs()[0].name
        return self._session.run(None, {input_name: inputs})

    def postprocess(self, raw_outputs: object, sample: COCOSample) -> Detection:
        """Delegate to adapter for model-specific output parsing.

        Parameters
        ----------
        raw_outputs : object
            List of numpy arrays from session.run().
        sample : COCOSample
            Original image metadata.
        """
        return self._adapter.parse_outputs(
            raw_outputs,
            original_size=sample.original_size,
            input_size=self._adapter.input_size,
            score_threshold=self._score_threshold,
        )

    @property
    def model_size_mb(self) -> float:
        """ONNX file size in MB."""
        if not self._onnx_path.exists():
            return 0.0
        return self._onnx_path.stat().st_size / (1024 * 1024)
