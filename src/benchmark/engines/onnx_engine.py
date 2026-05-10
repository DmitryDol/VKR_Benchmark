"""ONNX Runtime inference engine for stage 2 benchmarking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
from PIL import Image

from benchmark.data.coco_loader import COCO_80_TO_91
from benchmark.engines.base import BaseEngine, Detection

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.data.coco_loader import COCOSample

# Minimum number of output tensors expected from RT-DETR ONNX model
_MIN_ONNX_OUTPUTS: int = 2

logger = logging.getLogger(__name__)


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
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.01,
    ) -> None:
        super().__init__(model_name, engine_type="onnx", precision="fp32")
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
        logger.info(
            "ONNX session loaded: %s (%.1f MB)",
            self._onnx_path.name,
            self.model_size_mb,
        )

    def preprocess(self, sample: COCOSample) -> np.ndarray:
        """Resize image and convert to (1, 3, H, W) float32 numpy array.

        Matches PyTorchEngine preprocessing: resize to input_size, scale
        to [0, 1], no ImageNet normalization (RT-DETR convention).
        """
        h, w = self._input_size
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
        """Parse ONNX RT-DETR outputs to Detection.

        RT-DETR ONNX model outputs two tensors:
          - logits: (1, num_queries, num_classes)
          - pred_boxes: (1, num_queries, 4) in [cx, cy, w, h] normalized

        This matches the HuggingFace RT-DETR ONNX export convention from Phase 1.
        """
        if not isinstance(raw_outputs, list) or len(raw_outputs) < _MIN_ONNX_OUTPUTS:
            msg = f"Unexpected ONNX output format: {type(raw_outputs)}"
            raise RuntimeError(msg)

        logits: np.ndarray = raw_outputs[0][0]  # (num_queries, num_classes)
        pred_boxes: np.ndarray = raw_outputs[1][0]  # (num_queries, 4) cx cy w h norm

        # Softmax scores + argmax labels
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        scores = probs.max(axis=-1)  # (num_queries,)
        labels = probs.argmax(axis=-1).astype(np.int64)  # (num_queries,)

        # Filter by score threshold
        keep = scores >= self._score_threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes_norm = pred_boxes[keep]  # (N, 4) cx cy w h normalized

        # Convert cx cy w h -> x1 y1 x2 y2 in original pixel space
        orig_h, orig_w = sample.original_size
        cx, cy, w, h = (
            boxes_norm[:, 0] * orig_w,
            boxes_norm[:, 1] * orig_h,
            boxes_norm[:, 2] * orig_w,
            boxes_norm[:, 3] * orig_h,
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # Labels from ONNX are 0-indexed (80 classes) -> map to COCO 91-class IDs
        coco_labels = np.array(
            [COCO_80_TO_91.get(int(lbl), int(lbl)) for lbl in labels], dtype=np.int64
        )

        return Detection(
            boxes=boxes.reshape(-1, 4),
            scores=scores.astype(np.float32),
            labels=coco_labels,
        )

    @property
    def model_size_mb(self) -> float:
        """ONNX file size in MB."""
        if not self._onnx_path.exists():
            return 0.0
        return self._onnx_path.stat().st_size / (1024 * 1024)
