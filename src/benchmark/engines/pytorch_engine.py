"""PyTorch FP32 baseline inference engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import torchvision.transforms.functional as tvf
from PIL import Image

from benchmark.engines.base import BaseEngine, Detection

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from benchmark.data.coco_loader import COCOSample

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for model-specific loading and output parsing.

    Each target architecture (RT-DETR, RF-DETR, D-FINE, DEIMv2,
    YOLO11, YOLO26) implements this protocol to handle its
    unique model loading and output format.
    """

    @property
    def input_size(self) -> tuple[int, int]:
        """Model input resolution (height, width)."""
        ...

    def load(self, weights_path: Path, device: torch.device) -> nn.Module:
        """Load model weights and return the model ready for inference."""
        ...

    def parse_outputs(
        self,
        raw_outputs: object,
        original_size: tuple[int, int],
        input_size: tuple[int, int],
        score_threshold: float,
    ) -> Detection:
        """Convert raw model outputs to a Detection object.

        Parameters
        ----------
        raw_outputs : object
            Raw model forward pass output.
        original_size : tuple[int, int]
            Original image size (height, width) for box rescaling.
        input_size : tuple[int, int]
            Model input size (height, width) used during preprocessing.
        score_threshold : float
            Minimum confidence score to keep a detection.

        Returns
        -------
        Detection
            Filtered detections with boxes in x1y1x2y2 format,
            scores, and COCO 91-class label IDs.
        """
        ...


class PyTorchEngine(BaseEngine):
    """FP32 baseline inference engine using pure PyTorch.

    TF32 is explicitly disabled to ensure reproducible FP32 baseline
    measurements as required by the benchmarking protocol.

    Parameters
    ----------
    model_name : str
        Human-readable model identifier (e.g. "rt-detr-l").
    adapter : ModelAdapter
        Model-specific adapter for loading and output parsing.
    device : str
        Target device ("cuda" or "cpu").
    score_threshold : float
        Minimum detection confidence for postprocessing.
    """

    def __init__(
        self,
        model_name: str,
        adapter: ModelAdapter,
        device: str = "cuda",
        score_threshold: float = 0.01,
    ) -> None:
        super().__init__(model_name, engine_type="pytorch", precision="fp32")
        self._adapter = adapter
        self._device = torch.device(device)
        self._score_threshold = score_threshold
        self._model: nn.Module | None = None
        self._weights_path: Path | None = None

    def load_model(self, weights_path: Path) -> None:
        """Load model and disable TF32 for FP32 baseline integrity."""
        # Disable TF32 — critical for accurate FP32 baseline
        torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
        logger.info("TF32 disabled for FP32 baseline integrity")

        self._weights_path = weights_path
        self._model = self._adapter.load(weights_path, self._device)
        self._model.eval()
        logger.info("Model loaded: %s (%.1f MB)", self.model_name, self.model_size_mb)

    def preprocess(self, sample: COCOSample) -> torch.Tensor:
        """Resize, normalize, and convert image to model input tensor.

        Returns (1, 3, H, W) float32 tensor on the target device.
        """
        h, w = self._adapter.input_size
        img = Image.fromarray(sample.image).resize((w, h), Image.BILINEAR)
        tensor = tvf.to_tensor(img)  # (3, H, W) float32 [0, 1]
        tensor = tvf.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return tensor.unsqueeze(0).to(self._device)

    def infer(self, inputs: object) -> object:
        """Run forward pass under torch.no_grad()."""
        if self._model is None:
            msg = "Model not loaded. Call load_model() first."
            raise RuntimeError(msg)
        with torch.no_grad():
            return self._model(inputs)

    def postprocess(self, raw_outputs: object, sample: COCOSample) -> Detection:
        """Delegate to adapter for model-specific output parsing."""
        return self._adapter.parse_outputs(
            raw_outputs,
            original_size=sample.original_size,
            input_size=self._adapter.input_size,
            score_threshold=self._score_threshold,
        )

    @property
    def model_size_mb(self) -> float:
        """Model size in MB calculated from parameter memory."""
        if self._model is None:
            return 0.0
        return sum(p.numel() * p.element_size() for p in self._model.parameters()) / (1024 * 1024)

    @property
    def model(self) -> nn.Module:
        """Access the underlying PyTorch model (for ONNX export)."""
        if self._model is None:
            msg = "Model not loaded. Call load_model() first."
            raise RuntimeError(msg)
        return self._model

    def dummy_input(self) -> torch.Tensor:
        """Create a dummy input tensor for tracing/export."""
        h, w = self._adapter.input_size
        return torch.randn(1, 3, h, w, device=self._device)
