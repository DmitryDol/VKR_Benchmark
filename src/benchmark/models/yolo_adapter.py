"""YOLO family adapter for Ultralytics models (YOLO11, YOLO26)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

from benchmark.data.coco_loader import COCO_80_TO_91
from benchmark.engines.base import Detection

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

logger = logging.getLogger(__name__)


class YOLOAdapter:
    """Adapter for YOLO11 and YOLO26 models using Ultralytics.

    Supports both NMS-based (YOLO11) and NMS-free (YOLO26) models.
    
    Parameters
    ----------
    input_size : tuple[int, int]
        Model input resolution (height, width).
    is_nms_free : bool
        Whether the model is NMS-free (e.g. YOLO26, YOLOv10).
    """

    def __init__(self, input_size: tuple[int, int] = (640, 640), is_nms_free: bool = False) -> None:
        self._input_size = input_size
        self._is_nms_free = is_nms_free

    @property
    def input_size(self) -> tuple[int, int]:
        """Model input resolution (height, width)."""
        return self._input_size

    def load(self, weights_path: Path, device: torch.device) -> nn.Module:
        """Load YOLO model and return the underlying nn.Module."""
        # We load the full YOLO wrapper but only return the inner model for PyTorchEngine
        yolo = YOLO(str(weights_path))
        yolo.to(device)
        return yolo.model

    def infer(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run forward pass."""
        return model(inputs)

    def parse_outputs(
        self,
        raw_outputs: torch.Tensor | list[torch.Tensor] | list[np.ndarray],
        original_size: tuple[int, int],
        input_size: tuple[int, int],
        score_threshold: float,
    ) -> Detection:
        """Convert raw YOLO outputs to Detection object.

        Handles both NMS-free (YOLO26) and NMS-based (YOLO11) outputs.
        Accepts torch.Tensor (PyTorch/TensorRT path) or list[np.ndarray]
        (ONNX Runtime path).
        """
        if self._is_nms_free:
            return self._parse_nms_free(raw_outputs, original_size, input_size, score_threshold)

        return self._parse_nms(raw_outputs, original_size, input_size, score_threshold)

    def _parse_nms(
        self,
        raw_outputs: torch.Tensor | list[torch.Tensor] | list[np.ndarray],
        original_size: tuple[int, int],
        input_size: tuple[int, int],
        score_threshold: float,
    ) -> Detection:
        """Post-process YOLO11 outputs with NMS.

        Accepts either torch.Tensor (PyTorch path) or list[np.ndarray] (ONNX
        path). NumPy arrays are converted to CPU tensors before calling
        non_max_suppression, which internally calls .amax() — a method removed
        from numpy.ndarray in NumPy 2.0 (numpy/numpy#24889).
        """
        # raw_outputs for YOLO11 is typically [tensor(1, 84, 8400)]
        preds = raw_outputs[0] if isinstance(raw_outputs, (list, tuple)) else raw_outputs

        # Convert numpy arrays (ONNX path) to torch.Tensor so that
        # ultralytics.utils.nms.non_max_suppression can call .amax() safely.
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)

        # Apply NMS
        # conf_thres=score_threshold, iou_thres=0.7 (default)
        results = non_max_suppression(
            preds,
            conf_thres=score_threshold,
            iou_thres=0.7,
            nc=80,
        )[0]  # (N, 6) -> [x1, y1, x2, y2, conf, cls]

        if results is None or len(results) == 0:
            return Detection(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                labels=np.zeros(0, dtype=np.int64),
            )

        # Scale boxes to original size
        img_h, img_w = original_size
        in_h, in_w = input_size

        boxes = results[:, :4]
        # YOLO boxes are in input_size coordinates
        # Simple scaling
        boxes[:, [0, 2]] *= (img_w / in_w)
        boxes[:, [1, 3]] *= (img_h / in_h)

        scores = results[:, 4].cpu().numpy()
        labels_80 = results[:, 5].cpu().numpy().astype(np.int64)
        labels_91 = np.array([COCO_80_TO_91[idx] for idx in labels_80], dtype=np.int64)

        return Detection(
            boxes=boxes.cpu().numpy().astype(np.float32),
            scores=scores,
            labels=labels_91,
        )

    def _parse_nms_free(
        self,
        raw_outputs: torch.Tensor | list[torch.Tensor] | list[np.ndarray],
        original_size: tuple[int, int],
        input_size: tuple[int, int],
        score_threshold: float,
    ) -> Detection:
        """Post-process YOLO26 (NMS-free) outputs.

        Accepts either torch.Tensor (PyTorch path) or list[np.ndarray] (ONNX
        path). Works with both — numpy arrays use module-level functions and
        avoid instance-method APIs removed in NumPy 2.0.
        """
        # For NMS-free models (YOLOv10, YOLO26), output is often (1, 300, 6)
        # where each detection is [x1, y1, x2, y2, conf, cls]
        if isinstance(raw_outputs, (list, tuple)):
            preds = raw_outputs[0]
        else:
            preds = raw_outputs

        is_numpy = isinstance(preds, np.ndarray)

        # Squeeze leading dimensions if necessary
        while preds.ndim > 2:
            preds = preds[0]

        mask = preds[:, 4] > score_threshold
        results = preds[mask]

        if len(results) == 0:
            return Detection(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                labels=np.zeros(0, dtype=np.int64),
            )

        img_h, img_w = original_size
        in_h, in_w = input_size

        boxes = results[:, :4].copy() if is_numpy else results[:, :4]
        boxes[:, [0, 2]] *= img_w / in_w
        boxes[:, [1, 3]] *= img_h / in_h

        if is_numpy:
            scores = results[:, 4].astype(np.float32)
            labels_80 = results[:, 5].astype(np.int64)
            boxes_np = boxes.astype(np.float32).reshape(-1, 4)
        else:
            scores = results[:, 4].cpu().numpy().astype(np.float32)
            labels_80 = results[:, 5].cpu().numpy().astype(np.int64)
            boxes_np = boxes.cpu().numpy().astype(np.float32).reshape(-1, 4)

        labels_91 = np.array([COCO_80_TO_91[idx] for idx in labels_80], dtype=np.int64)

        return Detection(
            boxes=boxes_np,
            scores=scores,
            labels=labels_91,
        )
