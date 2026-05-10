"""RT-DETR ModelAdapter for HuggingFace transformers RTDetrForObjectDetection.

Implements the ModelAdapter protocol defined in pytorch_engine.py.
Model: PekingU/rtdetr-r50 (ResNet-50, 640x640 input, COCO-91 classes).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from transformers import RTDetrForObjectDetection

from benchmark.engines.base import Detection

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Background is at index 0 in HF RT-DETR logits — strip it before argmax.
# Logits shape: (batch, 300, 92) = 91 COCO classes + 1 background.
_BACKGROUND_IDX: int = 0
# Verified shape of last logit dimension (91 COCO classes + 1 background).
_EXPECTED_NUM_CLASSES_WITH_BG: int = 92


class RTDetrONNXWrapper(nn.Module):
    """Thin nn.Module wrapper making RTDetrForObjectDetection ONNX-traceable.

    torch.onnx.export requires positional tensor arguments. HF models use
    keyword arguments (pixel_values=...) and return a dataclass, not a tuple.
    This wrapper converts positional input and returns (logits, pred_boxes).

    Parameters
    ----------
    model : RTDetrForObjectDetection
        Loaded HF model in eval mode.
    """

    def __init__(self, model: RTDetrForObjectDetection) -> None:
        super().__init__()
        self._model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning plain tensor tuple for ONNX tracing.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Shape (1, 3, 640, 640) float32.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (logits, pred_boxes) — (1,300,92) and (1,300,4).
        """
        outputs = self._model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes


class RTDETRAdapter:
    """ModelAdapter for PekingU/rtdetr-r50 via HuggingFace transformers.

    Implements the ModelAdapter protocol (pytorch_engine.py:29-72).
    Handles: weight loading, output parsing (sigmoid + threshold + box convert).
    """

    @property
    def input_size(self) -> tuple[int, int]:
        """Model input resolution (height, width)."""
        return (640, 640)

    def load(self, weights_path: Path, device: torch.device) -> nn.Module:
        """Load RT-DETR weights and return model ready for inference.

        Parameters
        ----------
        weights_path : Path
            Local directory containing HF model files (config.json + safetensors).
        device : torch.device
            Target device (cuda or cpu).

        Returns
        -------
        nn.Module
            RTDetrForObjectDetection in eval mode on target device.
        """
        logger.info("Loading RTDetrForObjectDetection from %s", weights_path)
        model = RTDetrForObjectDetection.from_pretrained(str(weights_path))
        model.eval()
        model = model.to(device)

        # Verify logits shape assumption on first load via a probe forward pass.
        # This is lightweight (no grad) and catches shape mismatches early.
        with torch.no_grad():
            probe = torch.zeros(1, 3, 640, 640, device=device)
            probe_out = model(pixel_values=probe)
            actual_shape = tuple(probe_out.logits.shape)
            logger.info(
                "RT-DETR logits shape: %s  pred_boxes shape: %s",
                actual_shape,
                tuple(probe_out.pred_boxes.shape),
            )
            if actual_shape[-1] != _EXPECTED_NUM_CLASSES_WITH_BG:
                logger.warning(
                    "Expected logits last dim=%d, got %d. "
                    "parse_outputs() class index logic may need adjustment.",
                    _EXPECTED_NUM_CLASSES_WITH_BG,
                    actual_shape[-1],
                )

        return model

    def parse_outputs(
        self,
        raw_outputs: object,
        original_size: tuple[int, int],
        input_size: tuple[int, int],  # noqa: ARG002
        score_threshold: float,
    ) -> Detection:
        """Convert RTDetrObjectDetectionOutput to Detection.

        Parameters
        ----------
        raw_outputs : object
            RTDetrObjectDetectionOutput with .logits and .pred_boxes.
        original_size : tuple[int, int]
            Original image (height, width) for pixel coordinate scaling.
        input_size : tuple[int, int]
            Model input resolution (unused; boxes are normalized to [0,1]).
        score_threshold : float
            Minimum sigmoid score to keep a detection.

        Returns
        -------
        Detection
            boxes: (N, 4) x1y1x2y2 in pixel coords of original image
            scores: (N,) float32 in [0, 1]
            labels: (N,) int64 COCO-91 category IDs (1-indexed, no background)
        """
        # Access HF output attributes — tensors on the model's device.
        logits: torch.Tensor = raw_outputs.logits[0]  # type: ignore[union-attr]  # (300, 92)
        pred_boxes: torch.Tensor = raw_outputs.pred_boxes[0]  # type: ignore[union-attr]  # (300, 4)

        # Strip background class at index 0 → (300, 91) probabilities.
        scores_all = torch.sigmoid(logits[:, (_BACKGROUND_IDX + 1) :])  # (300, 91)
        scores, class_indices = scores_all.max(dim=-1)  # each (300,)

        # class_indices are 0-indexed over 91 COCO classes.
        # COCO-91 category_id = class_index + 1 (1-indexed, no background).
        label_ids = class_indices + 1  # (300,) — COCO-91 IDs

        # Filter detections below threshold.
        keep = scores >= score_threshold
        scores = scores[keep]
        label_ids = label_ids[keep]
        boxes_norm = pred_boxes[keep]  # (N, 4) cx,cy,w,h in [0,1]

        # Convert normalized cx,cy,w,h → x1,y1,x2,y2 in original pixel coords.
        orig_h, orig_w = original_size
        cx, cy, w, h = boxes_norm.unbind(-1)
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # (N, 4)

        return Detection(
            boxes=boxes_xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4),
            scores=scores.cpu().numpy().astype(np.float32),
            labels=label_ids.cpu().numpy().astype(np.int64),
        )
