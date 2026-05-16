"""RF-DETR ModelAdapter for Roboflow's `rfdetr` package (RFDETRLarge).

Implements the ModelAdapter protocol defined in pytorch_engine.py.
Model: rfdetr.RFDETRLarge (DINOv2 backbone + DETR decoder, 704x704 input, COCO-91 classes).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from PIL import Image
from rfdetr import RFDETRLarge
from torch import nn

from benchmark.engines.base import Detection

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.data.coco_loader import COCOSample

logger = logging.getLogger(__name__)

# Module-level constants (D-RF-04, source: rfdetr/detr.py:373-375)
_INPUT_SIZE: tuple[int, int] = (704, 704)  # native RFDETRLarge resolution
_IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# RF-DETR ONNX output names (per rfdetr/export/main.py:120): ["dets", "labels"]
# Logits shape: (B, 300, 91). Slot 0 = N/A (no COCO id=0), slots 1..89 = COCO-91 ids,
# slot 90 = background (DETR convention).
_EXPECTED_NUM_CLASSES: int = 91  # NOT 80 (unlike RT-DETR)
_NUM_QUERIES: int = 300  # ModelConfig.num_queries == num_select
_BG_INDEX: int = 90  # DETR-convention background slot
_BOX_DIM: int = 4  # boxes are (N, 4) = cx, cy, w, h or x1, y1, x2, y2


class RFDETRAdapter:
    """ModelAdapter for rfdetr.RFDETRLarge.

    Implements the ModelAdapter protocol (pytorch_engine.py:29-72).
    Handles: weight loading, preprocessing (vendor ImageNet + direct resize),
    inference, output parsing (sigmoid + top-k over flattened (queries x classes)).

    Notes
    -----
    RF-DETR outputs COCO-91 category IDs directly — no COCO-80 to COCO-91 LUT
    needed (unlike RT-DETR). Slot 0 is N/A (no COCO id=0), slots 1..89 are
    COCO-91 IDs sparse, slot 90 is the DETR-convention background class.
    Both slot 0 and slot 90 are filtered in parse_outputs before COCOeval sees results.
    """

    @property
    def input_size(self) -> tuple[int, int]:
        """Model input resolution (height, width)."""
        return _INPUT_SIZE

    def load(self, weights_path: Path, device: torch.device) -> nn.Module:
        """Load RF-DETR-L weights and return model ready for inference.

        Parameters
        ----------
        weights_path : Path
            Conventionally ``weights/rfdetr-l/``. The rfdetr package manages its own
            checkpoint cache; this directory exists only to keep the project's per-model
            layout uniform with ``rtdetr-r50vd/``, ``yolo11l/``, ``yolo26l/``.
            First call triggers a ~150 MB download (~30-60 s) into vendor cache.
            Subsequent calls hit the cache and start instantly (Landmine #6).
        device : torch.device
            Target device. ``RFDETRLarge()`` initialises on CPU by design — this
            adapter MUST call ``.to(device)`` explicitly. The vendor's
            ``_ensure_model_on_device`` at ``detr.py:345`` only fires on the first
            ``.predict()`` call (Landmine #7).

        Returns
        -------
        nn.Module
            LWDETR inner ``nn.Module`` in eval mode on target device.
        """
        logger.info("Loading RFDETRLarge (weights dir: %s)", weights_path)
        m = RFDETRLarge()  # downloads rf-detr-large-2026.pth on first call
        nn_model = m.model.model  # unwrap: RFDETR -> ModelContext -> LWDETR (nn.Module)
        nn_model.eval()
        nn_model = nn_model.to(device)

        # Verify forward shape on a probe — keeps Landmine #8 (output ordering)
        # explicit in logs and catches checkpoint corruption early.
        with torch.no_grad():
            probe = torch.zeros(1, 3, _INPUT_SIZE[0], _INPUT_SIZE[1], device=device)
            probe_out = nn_model(probe)
            logger.info(
                "RF-DETR pred_logits shape: %s  pred_boxes shape: %s",
                tuple(probe_out["pred_logits"].shape),
                tuple(probe_out["pred_boxes"].shape),
            )

        return nn_model

    def preprocess(
        self, sample: COCOSample, device: torch.device | None = None
    ) -> torch.Tensor:
        """Direct-resize + ImageNet-normalize to (1, 3, 704, 704) per D-RF-04.

        Vendor pipeline from rfdetr/detr.py:1180-1183:
        - ``F.to_tensor``: HWC RGB uint8 -> CHW float32 / 255 -> [0, 1]
        - ``F.resize``: direct stretch to 704x704 (NOT letterbox — DINOv2 patch alignment)
        - ``F.normalize``: ImageNet mean/std (DINOv2 inherits)

        Parameters
        ----------
        sample : COCOSample
            HWC RGB uint8 image with ``original_size = (H, W)``.
        device : torch.device | None
            If provided, the output tensor is moved to this device.

        Returns
        -------
        torch.Tensor
            Shape (1, 3, 704, 704) float32 on the requested device.
        """
        img = Image.fromarray(sample.image)  # RGB (project convention)
        t = tvf.to_tensor(img)  # (3, H, W) float32 [0, 1]
        t = tvf.resize(t, [_INPUT_SIZE[0], _INPUT_SIZE[1]])  # direct stretch (no letterbox)
        t = tvf.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)
        t = t.unsqueeze(0)  # (1, 3, 704, 704)
        if device is not None:
            t = t.to(device)
        return t

    def infer(self, model: nn.Module, inputs: torch.Tensor) -> object:
        """Run forward pass. RF-DETR's LWDETR takes a single positional tensor.

        Parameters
        ----------
        model : nn.Module
            LWDETR model returned by ``load()``.
        inputs : torch.Tensor
            Shape (1, 3, 704, 704) preprocessed input.

        Returns
        -------
        object
            dict with keys {pred_logits, pred_boxes, aux_outputs, enc_outputs}.
            Only ``pred_logits`` (1, 300, 91) and ``pred_boxes`` (1, 300, 4)
            are consumed by ``parse_outputs``.

        Notes
        -----
        Uses POSITIONAL argument, NOT ``model(pixel_values=inputs)`` as in HF
        RT-DETR. RF-DETR's LWDETR uses positional single-tensor input (RESEARCH
        line 268; detr.py:1217).
        """
        return model(inputs)

    def parse_outputs(
        self,
        raw_outputs: object,
        original_size: tuple[int, int],
        input_size: tuple[int, int],  # noqa: ARG002
        score_threshold: float,
    ) -> Detection:
        """Convert raw model outputs to Detection.

        Handles both dict (PyTorch path) and list/tuple (ONNX/TRT path).
        Output-order detection is SHAPE-based, not INDEX-based (Landmine #8):
        RF-DETR ONNX outputs [dets, labels] = [(N,4), (N,91)] — OPPOSITE of
        RT-DETR's [logits, pred_boxes]. Detection by shape prevents regression.

        Parameters
        ----------
        raw_outputs : object
            - dict with ``pred_logits`` and ``pred_boxes`` keys (PyTorch path)
            - list/tuple of numpy arrays [dets (1,300,4), labels (1,300,91)] or
              [labels (1,300,91), dets (1,300,4)] (ONNX/TRT path — shape-detected)
        original_size : tuple[int, int]
            Original image (height, width) for pixel coordinate scaling.
        input_size : tuple[int, int]
            Model input resolution (unused; boxes are normalized to [0, 1]).
        score_threshold : float
            Minimum sigmoid score to keep a detection.

        Returns
        -------
        Detection
            boxes: (N, 4) x1y1x2y2 in pixel coords of original image
            scores: (N,) float32 in [0, 1]
            labels: (N,) int64 COCO-91 category IDs (direct, no LUT needed)
        """
        # --- Unify input format ---
        if isinstance(raw_outputs, dict):
            # PyTorch path: dict with named tensors
            logits = raw_outputs["pred_logits"][0]  # (300, 91) torch
            boxes_norm = raw_outputs["pred_boxes"][0]  # (300, 4) torch
        elif isinstance(raw_outputs, (list, tuple)):
            # ONNX/TRT path: detect output order by SHAPE not INDEX (Landmine #8).
            # RF-DETR ONNX exports [dets (1,300,4), labels (1,300,91)] per
            # rfdetr/export/main.py:120. Shape-based detection is robust to
            # source reordering (RT-DETR outputs [logits, pred_boxes]).
            a, b = raw_outputs[0][0], raw_outputs[1][0]
            if a.shape[-1] == _BOX_DIM:
                boxes_norm = torch.from_numpy(a)  # (300, 4)
                logits = torch.from_numpy(b)  # (300, 91)
            else:
                logits = torch.from_numpy(a)  # (300, 91)
                boxes_norm = torch.from_numpy(b)  # (300, 4)
        else:
            msg = f"Unsupported raw_outputs type: {type(raw_outputs)}"
            raise TypeError(msg)

        # --- Sigmoid + top-k over flattened (queries x classes) ---
        # Matches PostProcess in rfdetr/models/postprocess.py:27-80:
        # topk over the JOINT (queries x classes) space, so a single query can
        # contribute multiple detections (different classes) to the top-300 set.
        # This is different from RT-DETR's per-query argmax.
        probs = logits.sigmoid()  # (300, 91)
        flat = probs.view(-1)  # (300 * 91,)
        topk_vals, topk_idx = torch.topk(flat, _NUM_QUERIES)  # (300,) each
        # Decompose flat index -> (query_idx, class_idx)
        query_idx = topk_idx // probs.shape[1]  # (300,) values in 0..299
        class_idx = topk_idx % probs.shape[1]  # (300,) values in 0..90

        # --- Threshold filter + remove no-object / N/A slots ---
        # class_idx == 0: COCO-91 has no id=0 (never trained as positive — drop)
        # class_idx == _BG_INDEX (90): background "no-object" per DETR convention (drop)
        keep = (topk_vals >= score_threshold) & (class_idx != 0) & (class_idx != _BG_INDEX)
        scores = topk_vals[keep].cpu().numpy().astype(np.float32)
        labels = class_idx[keep].cpu().numpy().astype(np.int64)  # direct COCO-91 IDs
        sel_queries = query_idx[keep]  # which of the 300 queries

        # --- Gather + denormalize boxes ---
        sel_boxes = boxes_norm[sel_queries]  # (N, 4) cx, cy, w, h in [0, 1]
        orig_h, orig_w = original_size
        cx, cy, w, h = sel_boxes.unbind(-1)
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy().astype(np.float32)

        return Detection(
            boxes=boxes_xyxy.reshape(-1, 4),
            scores=scores,
            labels=labels,
        )
