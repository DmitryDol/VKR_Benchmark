"""RT-DETR ModelAdapter for HuggingFace transformers RTDetrForObjectDetection.

Implements the ModelAdapter protocol defined in pytorch_engine.py.
Model: PekingU/rtdetr_r50vd (ResNet-50 Visual Dependency, 640x640 input, COCO-80 classes).
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

# HF RT-DETR outputs 80 COCO classes directly — no background index.
# Logits shape: (batch, 300, 80).
_EXPECTED_NUM_CLASSES: int = 80

# Canonical COCO-80 index (0-79) → COCO-91 category_id.
# Hardcoded to guarantee correctness independent of coco_loader.py.
# Accounts for 11 missing COCO-91 IDs: 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91.
_COCO80_LUT: np.ndarray = np.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,  # idx  0- 9
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,  # idx 10-19
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,  # idx 20-29
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,  # idx 30-39
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,  # idx 40-49
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,  # idx 50-59
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,  # idx 60-69
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,  # idx 70-79
    ],
    dtype=np.int64,
)


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
            (logits, pred_boxes) — (1,300,80) and (1,300,4).
        """
        outputs = self._model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes


class RTDETRAdapter:
    """ModelAdapter for PekingU/rtdetr_r50vd via HuggingFace transformers.

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
            if actual_shape[-1] != _EXPECTED_NUM_CLASSES:
                logger.warning(
                    "Expected logits last dim=%d, got %d. "
                    "parse_outputs() class index logic may need adjustment.",
                    _EXPECTED_NUM_CLASSES,
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
            labels: (N,) int64 COCO-91 category IDs (mapped from COCO-80 via LUT)
        """
        # Access HF output attributes — tensors on the model's device.
        logits: torch.Tensor = raw_outputs.logits[0]  # type: ignore[union-attr]  # (300, 80)
        pred_boxes: torch.Tensor = raw_outputs.pred_boxes[0]  # type: ignore[union-attr]  # (300, 4)

        # No background class — model outputs 80 COCO classes directly.
        scores_all = torch.sigmoid(logits)  # (300, 80)
        scores, class_indices = scores_all.max(dim=-1)  # each (300,)

        # Filter detections below threshold.
        keep = scores >= score_threshold
        scores = scores[keep]
        kept_indices = class_indices[keep].cpu().numpy()  # (N,) 0-79
        boxes_norm = pred_boxes[keep]  # (N, 4) cx,cy,w,h in [0,1]

        # Map COCO-80 index (0-79) → COCO-91 category ID via pre-built LUT.
        label_ids = _COCO80_LUT[kept_indices]  # (N,) int64

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
            labels=label_ids,
        )
