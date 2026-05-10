# Phase 1: RT-DETR Adapter & ONNX Pipeline — Execution Plan

**Phase:** 1 of 5
**Goal:** RT-DETR FP32 baseline runs correctly end-to-end and produces a verified ONNX model
**Waves:** 5 (sequential — each wave must pass its acceptance gate before the next begins)

---

## Source Audit

| Source | Item | Covered By |
|--------|------|------------|
| GOAL | RT-DETR FP32 baseline end-to-end | Wave 1 (FIX-01), Wave 2 (ADPT), Wave 4 (runner) |
| REQ FIX-01 | Fix double infer() in warm-up | Wave 1, Task P1-T01 |
| REQ ADPT-01 | RT-DETR weights load + inference | Wave 2, Task P1-T03 |
| REQ ADPT-02 | Parse outputs → Detection format | Wave 2, Task P1-T03 |
| REQ ADPT-03 | Download/manage weights | Wave 2, Task P1-T02 |
| REQ ONNX-01 | ONNX export opset 17 | Wave 3, Task P1-T05 |
| REQ ONNX-02 | onnxsim applied | Wave 3, Task P1-T05 |
| REQ ONNX-03 | onnx.checker validation | Wave 3, Task P1-T05 |
| REQ BENCH-01 | 50 warm-up iterations | Wave 1, Task P1-T01 (FIX-01 enables correct count) |
| REQ BENCH-02 | 1000 measured iterations | Already correct in base.py; confirmed in Wave 5 |
| REQ BENCH-03 | Batch size 1 | Already enforced by COCODataLoader; confirmed in Wave 5 |
| REQ BENCH-04 | TF32 disabled | Already in PyTorchEngine.load_model(); confirmed in Wave 5 |
| REQ BENCH-05 | VRAM reset between engines | Already in BaseEngine.reset_vram_tracking(); used in Wave 4 |
| REQ BENCH-06 | CUDA sync at timing boundaries | Already in benchmark_latency(); confirmed in Wave 5 |
| D-01 (CONTEXT.md) | Use HF transformers RTDetrForObjectDetection | Wave 0, Wave 2 |
| D-02 (CONTEXT.md) | Model ID: PekingU/rtdetr-r50 | Wave 2, Task P1-T02 |
| D-03 (CONTEXT.md) | weights/rtdetr-r50/ directory via snapshot_download | Wave 2, Task P1-T02 |
| D-04 (CONTEXT.md) | Adapter in src/benchmark/models/rtdetr_adapter.py | Wave 2, Task P1-T03 |
| D-05 (CONTEXT.md) | FIX-01 is first task | Wave 1, Task P1-T01 |

---

## Threat Model

### Trust Boundaries

| Boundary | Description |
|----------|-------------|
| HuggingFace Hub → local weights/ | Model weights downloaded from external server |
| ONNX file → onnx.checker | Exported graph consumed by validator |
| logits tensor → parse_outputs() | Raw model output trusted but shape must be verified |

### STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-01-01 | Tampering | HF weight download (safetensors) | accept | `from_pretrained()` defaults to safetensors format; avoids pickle deserialization. No pickle `.pth` files. |
| T-01-02 | Tampering | ONNX model file | mitigate | `validate_onnx()` calls `onnx.checker.check_model()` immediately after export — raises on malformed graph before any runtime use |
| T-01-03 | Information Disclosure | VRAM state bleed between engines | mitigate | `BaseEngine.reset_vram_tracking()` calls `torch.cuda.reset_peak_memory_stats()` + `torch.cuda.empty_cache()` before each engine run in runner script |
| T-01-04 | Denial of Service | logits shape assumption wrong → mAP=0 | mitigate | Add `print(outputs.logits.shape)` debug log in `parse_outputs()` on first run; assert shape[-1] == 92 with informative error message |

---

## Wave 0 — Environment Setup

**Goal:** `transformers` available; pytest scaffolded; `weights/` gitignored.
**Gate:** `uv run python -c "from transformers import RTDetrForObjectDetection; print('OK')"` exits 0.

### Task P1-T00a: Add `transformers` dependency

**Files:** `pyproject.toml`, `uv.lock` (auto-updated)

**Action:**
Run `uv add "transformers[torch]"` in the project root. This installs transformers and its transitive dependency `huggingface_hub`. Verify `pyproject.toml` lists `transformers>=4.43.0` under `[project.dependencies]`.

**Verify:**
```bash
uv run python -c "import transformers; print(transformers.__version__)"
uv run python -c "from transformers import RTDetrForObjectDetection; print('RTDetrForObjectDetection importable')"
```

**Done:** Both commands exit 0 with no ImportError. `uv.lock` updated.

---

### Task P1-T00b: Add `pytest` dev dependency and create test scaffolding

**Files:**
- `pyproject.toml` (add pytest to dev group)
- `tests/conftest.py` (shared fixtures)
- `tests/test_base_engine.py` (FIX-01 tests — initially stubs)
- `tests/test_rtdetr_adapter.py` (adapter tests — initially stubs)
- `tests/test_onnx_export.py` (ONNX tests — initially stubs)
- `.gitignore` (add `weights/` entry)

**Action:**

1. Run `uv add --dev pytest`. Verify added to `[tool.uv.dev-dependencies]` in `pyproject.toml`.

2. Create `tests/conftest.py` with shared fixtures:
```python
"""Shared pytest fixtures for benchmark tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmark.data.coco_loader import COCOAnnotation, COCOSample
from benchmark.engines.base import Detection


@pytest.fixture
def dummy_sample() -> COCOSample:
    """Minimal COCOSample for unit tests (no real image file needed)."""
    return COCOSample(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        image_id=1,
        original_size=(480, 640),
        annotations=COCOAnnotation(
            boxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros(0, dtype=np.int64),
            areas=np.zeros(0, dtype=np.float32),
            iscrowd=np.zeros(0, dtype=np.int64),
        ),
    )


@pytest.fixture
def dummy_detection() -> Detection:
    """Detection with two boxes for postprocess unit tests."""
    return Detection(
        boxes=np.array([[10.0, 20.0, 110.0, 120.0], [50.0, 60.0, 150.0, 160.0]], dtype=np.float32),
        scores=np.array([0.9, 0.7], dtype=np.float32),
        labels=np.array([1, 2], dtype=np.int64),
    )
```

3. Create `tests/test_base_engine.py` with FIX-01 unit test (uses mock):
```python
"""Tests for BaseEngine benchmarking protocol."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

from benchmark.engines.base import WARMUP_RUNS, BaseEngine, Detection


class ConcreteEngine(BaseEngine):
    """Minimal concrete subclass for testing abstract BaseEngine."""

    def load_model(self, weights_path):  # noqa: ANN001
        pass

    def preprocess(self, sample):  # noqa: ANN001
        return torch.zeros(1, 3, 640, 640)

    def infer(self, inputs):  # noqa: ANN001
        return MagicMock()

    def postprocess(self, raw_outputs, sample):  # noqa: ANN001
        return Detection(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            labels=np.zeros(0, dtype=np.int64),
        )

    @property
    def model_size_mb(self) -> float:
        return 0.0


def test_warmup_calls_infer_exactly_once_per_iteration(dummy_sample):
    """FIX-01: warm-up loop must call infer() exactly once per iteration."""
    engine = ConcreteEngine("test", "pytorch", "fp32")

    infer_call_count = 0
    original_infer = engine.infer

    def counting_infer(inputs):  # noqa: ANN001
        nonlocal infer_call_count
        infer_call_count += 1
        return original_infer(inputs)

    engine.infer = counting_infer  # type: ignore[method-assign]

    # Build a minimal dataloader mock with one sample
    dataloader = MagicMock()
    dataloader.__len__ = MagicMock(return_value=1)
    dataloader.__getitem__ = MagicMock(return_value=dummy_sample)
    dataloader.__iter__ = MagicMock(return_value=iter([dummy_sample]))

    # Only run warm-up portion (patch MEASURE_RUNS to 0 to avoid full benchmark)
    with patch("benchmark.engines.base.MEASURE_RUNS", 0):
        try:
            engine.benchmark_latency(dataloader)
        except Exception:  # noqa: BLE001
            pass  # measurement phase may error with 0 runs; that's OK

    assert infer_call_count == WARMUP_RUNS, (
        f"Expected {WARMUP_RUNS} infer() calls during warm-up, got {infer_call_count}. "
        "FIX-01: warm-up loop calls infer() twice per iteration."
    )
```

4. Create `tests/test_rtdetr_adapter.py` — stubs only (marked xfail until Wave 2 complete):
```python
"""Tests for RTDETRAdapter. Requires transformers + GPU."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.mark.skip(reason="Implement after Wave 2 completes")
def test_parse_outputs_box_format():
    """parse_outputs() returns boxes in x1y1x2y2 pixel coords."""
    pass


@pytest.mark.skip(reason="Implement after Wave 2 completes")
def test_parse_outputs_scores_in_range():
    """parse_outputs() returns scores in [0, 1]."""
    pass


@pytest.mark.skip(reason="Implement after Wave 2 completes")
def test_parse_outputs_coco91_labels():
    """parse_outputs() returns COCO-91 label IDs (no category_id=0)."""
    pass
```

5. Create `tests/test_onnx_export.py` — stubs only:
```python
"""Tests for ONNX export pipeline. Requires GPU + weights downloaded."""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Implement after Wave 3 completes")
def test_export_creates_file():
    """Exported .onnx file exists and is >10 MB."""
    pass


@pytest.mark.skip(reason="Implement after Wave 3 completes")
def test_validate_passes():
    """validate_onnx() passes without exception."""
    pass
```

6. Add `weights/` to `.gitignore`. Read current `.gitignore` first, then append the line `weights/` if not already present.

**Verify:**
```bash
uv run pytest tests/ -x -q
```
Expected: All tests pass (stubs are skipped, FIX-01 test FAILS — this is the RED phase of TDD).

**Done:** `uv run pytest tests/ -x -q` exits with the FIX-01 test as the only failure (expected RED). `weights/` in `.gitignore`.

---

## Wave 1 — Fix FIX-01 (Warm-up Double Infer Bug)

**Depends on:** Wave 0 complete
**Goal:** `base.py` warm-up calls `infer()` exactly once per iteration.
**Gate:** `uv run pytest tests/test_base_engine.py -x -q` exits 0 (GREEN).

### Task P1-T01: Fix double infer() in benchmark_latency warm-up loop

**Files:** `src/benchmark/engines/base.py`

**Action:**

Read `src/benchmark/engines/base.py`. Locate lines 93–97 (the warm-up loop):

```python
# CURRENT (buggy):
for i in range(WARMUP_RUNS):
    sample = samples[i % n_samples]
    inputs = self.preprocess(sample)
    self.infer(inputs)
    self.postprocess(self.infer(inputs), sample)
```

Replace with:

```python
# FIXED:
for i in range(WARMUP_RUNS):
    sample = samples[i % n_samples]
    inputs = self.preprocess(sample)
    raw = self.infer(inputs)
    self.postprocess(raw, sample)
```

Change only these lines. Do not touch the measurement phase (lines 109–133), `evaluate_accuracy()`, or any other method.

**Verify:**
```bash
uv run pytest tests/test_base_engine.py::test_warmup_calls_infer_exactly_once_per_iteration -x -v
```
Expected: PASSED.

Also run ruff:
```bash
uv run ruff check src/benchmark/engines/base.py
uv run ruff format --check src/benchmark/engines/base.py
```
Expected: No errors.

**Done:** `test_warmup_calls_infer_exactly_once_per_iteration` passes. `base.py` has no ruff violations.

---

## Wave 2 — RT-DETR Adapter

**Depends on:** Wave 1 complete
**Goal:** `RTDETRAdapter` loads weights, runs inference, returns correct `Detection`.
**Gate:** `uv run pytest tests/test_rtdetr_adapter.py -x -v` all pass (un-skip and implement the tests).

### Task P1-T02: Weight download script

**Files:**
- `scripts/download_weights.py`

**Action:**

Create `scripts/download_weights.py`:

```python
"""Download RT-DETR pretrained weights from HuggingFace Hub.

Usage:
    uv run python scripts/download_weights.py

Downloads PekingU/rtdetr-r50 to weights/rtdetr-r50/ (gitignored).
Skips download if weights already present.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "PekingU/rtdetr-r50"
LOCAL_DIR = Path("weights/rtdetr-r50")
# Skip non-PyTorch serialization formats to save disk space
IGNORE_PATTERNS = ["*.msgpack", "*.h5", "flax_*", "tf_*", "rust_*"]


def download_rtdetr_r50(local_dir: Path = LOCAL_DIR) -> Path:
    """Download PekingU/rtdetr-r50 weights to local directory.

    Parameters
    ----------
    local_dir : Path
        Destination directory. Created if it does not exist.

    Returns
    -------
    Path
        Path to the downloaded weights directory.
    """
    config_path = local_dir / "config.json"
    if config_path.exists():
        logger.info("Weights already present at %s — skipping download.", local_dir)
        return local_dir

    logger.info("Downloading %s → %s", REPO_ID, local_dir)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(local_dir),
        ignore_patterns=IGNORE_PATTERNS,
    )
    logger.info("Download complete: %s", local_dir)
    return local_dir


if __name__ == "__main__":
    result = download_rtdetr_r50()
    logger.info("Weights ready at: %s", result)
    sys.exit(0)
```

**Verify:**
```bash
uv run python scripts/download_weights.py
```
Expected: `weights/rtdetr-r50/config.json` exists after run. Second run prints "skipping download" and exits 0.

```bash
uv run ruff check scripts/download_weights.py
```
Expected: No errors.

**Done:** `weights/rtdetr-r50/config.json` exists. Download is idempotent.

---

### Task P1-T03: RTDETRAdapter implementation

**Files:**
- `src/benchmark/models/__init__.py` (new)
- `src/benchmark/models/rtdetr_adapter.py` (new)
- `tests/test_rtdetr_adapter.py` (update stubs → real tests)
- `src/benchmark/engines/__init__.py` (add RTDETRAdapter export)

**Action:**

**Step 1 — Create `src/benchmark/models/__init__.py`:**
```python
"""Model adapter implementations for supported architectures."""

from benchmark.models.rtdetr_adapter import RTDETRAdapter, RTDetrONNXWrapper

__all__ = ["RTDETRAdapter", "RTDetrONNXWrapper"]
```

**Step 2 — Create `src/benchmark/models/rtdetr_adapter.py`:**

```python
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

    from benchmark.data.coco_loader import COCOSample

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
            logger.info("RT-DETR logits shape: %s  pred_boxes shape: %s",
                        actual_shape, tuple(probe_out.pred_boxes.shape))
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
        input_size: tuple[int, int],
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
            Model input resolution (unused; boxes are normalized).
        score_threshold : float
            Minimum sigmoid score to keep a detection.

        Returns
        -------
        Detection
            boxes: (N, 4) x1y1x2y2 in pixel coords of original image
            scores: (N,) float32 in [0, 1]
            labels: (N,) int64 COCO-91 category IDs (1-indexed, no background)
        """
        # Access HF output attributes — these are tensors on the model's device.
        logits: torch.Tensor = raw_outputs.logits[0]    # (300, 92)  # type: ignore[union-attr]
        pred_boxes: torch.Tensor = raw_outputs.pred_boxes[0]  # (300, 4)  # type: ignore[union-attr]

        # Strip background class at index 0 → (300, 91) probabilities.
        scores_all = torch.sigmoid(logits[:, (_BACKGROUND_IDX + 1):])  # (300, 91)
        scores, class_indices = scores_all.max(dim=-1)  # each (300,)

        # class_indices are 0-indexed over 91 COCO classes.
        # COCO-91 category_id = class_index + 1 (1-indexed).
        label_ids = class_indices + 1  # (300,) — COCO-91 IDs, no background

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
```

**Step 3 — Update `tests/test_rtdetr_adapter.py`** (replace stubs with real unit tests for `parse_outputs`; GPU load test stays manual):

```python
"""Tests for RTDETRAdapter — parse_outputs unit tests use mocked tensors (no GPU needed)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmark.engines.base import Detection
from benchmark.models.rtdetr_adapter import RTDETRAdapter


@pytest.fixture
def adapter() -> RTDETRAdapter:
    return RTDETRAdapter()


def _make_fake_output(
    num_queries: int = 300,
    num_classes_with_bg: int = 92,
    score_for_class_idx: int = 5,
    score_value: float = 5.0,
) -> object:
    """Build a fake RTDetrObjectDetectionOutput-like namespace."""
    logits = torch.full((1, num_queries, num_classes_with_bg), -10.0)
    # Give query 0 a high score for class_idx=5 (background stripped → COCO ID=6).
    # logits[:, :, 0] = background (stripped); class index 5 → slice index 5 after strip.
    logits[0, 0, score_for_class_idx + 1] = score_value  # +1 for background offset
    pred_boxes = torch.zeros(1, num_queries, 4)
    # Query 0: cx=0.5, cy=0.5, w=0.5, h=0.5 normalized
    pred_boxes[0, 0] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    return SimpleNamespace(logits=logits, pred_boxes=pred_boxes)


def test_parse_outputs_returns_detection(adapter, dummy_sample):
    """parse_outputs() returns a Detection instance."""
    raw = _make_fake_output()
    result = adapter.parse_outputs(raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.01)
    assert isinstance(result, Detection)


def test_parse_outputs_box_format_xyxy(adapter):
    """parse_outputs() converts cx,cy,w,h normalized → x1y1x2y2 pixel coords."""
    raw = _make_fake_output()
    # cx=0.5, cy=0.5, w=0.5, h=0.5 in normalized coords
    # orig_w=640, orig_h=480
    # x1=(0.5-0.25)*640=160, y1=(0.5-0.25)*480=120, x2=(0.5+0.25)*640=480, y2=(0.5+0.25)*480=360
    result = adapter.parse_outputs(raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.01)
    assert result.boxes.shape[1] == 4, "Boxes must be (N, 4)"
    if len(result.boxes) > 0:
        x1, y1, x2, y2 = result.boxes[0]
        assert x1 < x2, "x1 must be < x2"
        assert y1 < y2, "y1 must be < y2"
        assert abs(x1 - 160.0) < 1e-3, f"Expected x1=160.0, got {x1}"
        assert abs(y1 - 120.0) < 1e-3, f"Expected y1=120.0, got {y1}"


def test_parse_outputs_scores_in_range(adapter):
    """parse_outputs() scores are in [0, 1]."""
    raw = _make_fake_output()
    result = adapter.parse_outputs(raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.0)
    assert np.all(result.scores >= 0.0), "All scores must be >= 0"
    assert np.all(result.scores <= 1.0), "All scores must be <= 1"


def test_parse_outputs_no_background_label(adapter):
    """parse_outputs() never produces category_id=0 (background)."""
    raw = _make_fake_output()
    result = adapter.parse_outputs(raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.0)
    assert 0 not in result.labels, "Background label (0) must never appear in output"


def test_parse_outputs_label_is_coco91_class_index_plus_one(adapter):
    """Class at logit position 5 (after background strip) → COCO-91 ID 6."""
    raw = _make_fake_output(score_for_class_idx=5, score_value=10.0)
    result = adapter.parse_outputs(raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.01)
    assert len(result.labels) > 0, "Should have at least one detection"
    assert result.labels[0] == 6, f"Expected COCO-91 ID=6, got {result.labels[0]}"


def test_parse_outputs_threshold_filters(adapter):
    """Detections below score_threshold are dropped."""
    raw = _make_fake_output(score_value=-5.0)  # sigmoid(-5)≈0.007 — below 0.01
    result = adapter.parse_outputs(raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.01)
    assert len(result.scores) == 0, "Low-score detections must be filtered"


def test_parse_outputs_empty_when_all_filtered(adapter):
    """parse_outputs() with high threshold returns empty arrays."""
    raw = _make_fake_output(score_value=1.0)  # sigmoid(1)≈0.73 — below threshold=0.99
    result = adapter.parse_outputs(raw, original_size=(480, 640), input_size=(640, 640), score_threshold=0.99)
    assert result.boxes.shape == (0, 4), f"Expected (0,4), got {result.boxes.shape}"
    assert result.scores.shape == (0,)
    assert result.labels.shape == (0,)
```

**Step 4 — Update `src/benchmark/engines/__init__.py`** to also export `RTDETRAdapter`:

```python
from benchmark.engines.base import BaseEngine
from benchmark.engines.pytorch_engine import PyTorchEngine
from benchmark.models.rtdetr_adapter import RTDETRAdapter, RTDetrONNXWrapper

__all__ = ["BaseEngine", "PyTorchEngine", "RTDETRAdapter", "RTDetrONNXWrapper"]
```

**Verify:**
```bash
uv run ruff check src/benchmark/models/ src/benchmark/engines/__init__.py
uv run ruff format --check src/benchmark/models/ src/benchmark/engines/__init__.py
uv run pytest tests/test_rtdetr_adapter.py -x -v
```
Expected: All 6 parse_outputs tests pass. No ruff errors.

**Done:** `tests/test_rtdetr_adapter.py` all pass. `RTDETRAdapter` importable as `from benchmark.models import RTDETRAdapter`.

---

## Wave 3 — ONNX Export Pipeline

**Depends on:** Wave 2 complete (weights downloaded, RTDETRAdapter available)
**Goal:** `rtdetr_r50_sim.onnx` exists, passes `onnx.checker`, is smaller than raw export.
**Gate:** `uv run pytest tests/test_onnx_export.py -x -v` all pass.

### Task P1-T04: Update `export_to_onnx()` to accept `input_names`/`output_names`

**Files:** `src/benchmark/engines/onnx_export.py`

**Action:**

Read `src/benchmark/engines/onnx_export.py`. Modify `export_to_onnx()` signature to add two backward-compatible parameters. Also update `export_and_simplify()` to forward them.

Current signature (lines 30–36):
```python
def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
```

New signature — add `input_names` and `output_names` with defaults that preserve backward compatibility:
```python
def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> Path:
```

Inside the function body, replace the hardcoded `input_names=["input"]` and `output_names=["output"]` in the `torch.onnx.export()` call with:
```python
    _input_names = input_names if input_names is not None else ["input"]
    _output_names = output_names if output_names is not None else ["output"]
    # ... then in torch.onnx.export():
    input_names=_input_names,
    output_names=_output_names,
```

Also update `export_and_simplify()` to accept and forward these params:
```python
def export_and_simplify(
    model: nn.Module,
    output_path: Path,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> Path:
```

Forward them to `export_to_onnx()` in the call.

**Verify:**
```bash
uv run ruff check src/benchmark/engines/onnx_export.py
uv run ruff format --check src/benchmark/engines/onnx_export.py
uv run python -c "
from benchmark.engines.onnx_export import export_to_onnx
import inspect
sig = inspect.signature(export_to_onnx)
assert 'input_names' in sig.parameters, 'input_names param missing'
assert 'output_names' in sig.parameters, 'output_names param missing'
print('Signature OK:', list(sig.parameters.keys()))
"
```
Expected: Prints param list including `input_names` and `output_names`. No ruff errors.

**Done:** `export_to_onnx()` accepts `input_names`/`output_names`; existing callers unaffected (defaults `["input"]`/`["output"]`).

---

### Task P1-T05: ONNX export, simplify, validate for RT-DETR

**Files:**
- `tests/test_onnx_export.py` (un-stub and implement tests)
- Output artifacts: `weights/rtdetr-r50/rtdetr_r50.onnx`, `weights/rtdetr-r50/rtdetr_r50_sim.onnx`

**Action:**

**Step 1 — Update `tests/test_onnx_export.py`** with real tests:

```python
"""Tests for RT-DETR ONNX export pipeline. Requires GPU + downloaded weights."""
from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import torch

WEIGHTS_DIR = Path("weights/rtdetr-r50")
RAW_ONNX = WEIGHTS_DIR / "rtdetr_r50.onnx"
SIM_ONNX = WEIGHTS_DIR / "rtdetr_r50_sim.onnx"

requires_weights = pytest.mark.skipif(
    not (WEIGHTS_DIR / "config.json").exists(),
    reason="Weights not downloaded — run: uv run python scripts/download_weights.py",
)
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required for ONNX export test",
)


@requires_weights
@requires_gpu
def test_export_creates_raw_onnx():
    """export_to_onnx() creates a .onnx file >10 MB."""
    # Import here to avoid ImportError if transformers not installed
    from benchmark.engines.onnx_export import export_to_onnx
    from benchmark.models.rtdetr_adapter import RTDETRAdapter, RTDetrONNXWrapper
    from transformers import RTDetrForObjectDetection

    device = torch.device("cuda")
    model = RTDetrForObjectDetection.from_pretrained(str(WEIGHTS_DIR)).to(device).eval()
    wrapper = RTDetrONNXWrapper(model)

    dynamic_axes = {
        "pixel_values": {0: "batch"},
        "logits": {0: "batch"},
        "pred_boxes": {0: "batch"},
    }

    export_to_onnx(
        wrapper,
        output_path=RAW_ONNX,
        input_size=(640, 640),
        opset_version=17,
        dynamic_axes=dynamic_axes,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
    )

    assert RAW_ONNX.exists(), "Raw ONNX file not created"
    assert RAW_ONNX.stat().st_size > 10 * 1024 * 1024, "ONNX file too small (<10MB)"


@requires_weights
@requires_gpu
def test_simplify_produces_sim_onnx():
    """simplify_onnx() creates a simplified model and check_ok is True."""
    if not RAW_ONNX.exists():
        pytest.skip("Raw ONNX not present — run test_export_creates_raw_onnx first")

    from benchmark.engines.onnx_export import simplify_onnx
    sim_path = simplify_onnx(RAW_ONNX, output_path=SIM_ONNX)
    assert sim_path.exists(), "Simplified ONNX file not created"


@requires_weights
def test_validate_onnx_passes():
    """validate_onnx() passes onnx.checker without exception."""
    target = SIM_ONNX if SIM_ONNX.exists() else RAW_ONNX
    if not target.exists():
        pytest.skip("No ONNX file present — run export tests first")

    from benchmark.engines.onnx_export import validate_onnx
    result = validate_onnx(target)
    assert result is True


@requires_weights
def test_onnx_output_names():
    """Exported ONNX has output nodes named logits and pred_boxes."""
    target = SIM_ONNX if SIM_ONNX.exists() else RAW_ONNX
    if not target.exists():
        pytest.skip("No ONNX file present")

    model = onnx.load(str(target))
    output_names = [o.name for o in model.graph.output]
    assert "logits" in output_names, f"Expected 'logits' in outputs, got: {output_names}"
    assert "pred_boxes" in output_names, f"Expected 'pred_boxes' in outputs, got: {output_names}"
```

**Step 2 — Run the export pipeline manually** (test runner calls these, but also verify interactively):

```bash
# Export (takes ~2-3 minutes on RTX 3070)
uv run pytest tests/test_onnx_export.py::test_export_creates_raw_onnx -x -v -s
# Simplify
uv run pytest tests/test_onnx_export.py::test_simplify_produces_sim_onnx -x -v -s
# Validate
uv run pytest tests/test_onnx_export.py::test_validate_onnx_passes -x -v -s
# Output names check
uv run pytest tests/test_onnx_export.py::test_onnx_output_names -x -v -s
```

**Verify:**
```bash
uv run pytest tests/test_onnx_export.py -v
```
Expected: All 4 tests pass. `weights/rtdetr-r50/rtdetr_r50_sim.onnx` exists on disk.

**Done:** `weights/rtdetr-r50/rtdetr_r50_sim.onnx` exists. `onnx.checker.check_model()` passes. Output nodes named `logits` and `pred_boxes`.

---

## Wave 4 — End-to-End Runner Script

**Depends on:** Wave 3 complete
**Goal:** Single command runs FP32 benchmark and saves results to `results/`.
**Gate:** `uv run python scripts/run_phase1.py --limit 5` exits 0 and writes `results/phase1_*.csv`.

### Task P1-T06: Create `scripts/run_phase1.py`

**Files:** `scripts/run_phase1.py`

**Action:**

Create `scripts/run_phase1.py`. This script ties together all components from Waves 0–3 and produces the Phase 1 deliverable: a complete FP32 benchmark run with CSV/JSON output.

```python
"""Phase 1 end-to-end runner: RT-DETR FP32 baseline + ONNX export.

Usage:
    uv run python scripts/run_phase1.py
    uv run python scripts/run_phase1.py --limit 100
    uv run python scripts/run_phase1.py --skip-onnx
    uv run python scripts/run_phase1.py --images-dir data/val2017 --annotations data/annotations/instances_val2017.json

Outputs:
    results/phase1_<timestamp>.csv   — per-stage metrics
    results/phase1_<timestamp>.json  — full results (all stages)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase1")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Phase 1: RT-DETR FP32 benchmark")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/val2017"),
        help="Path to COCO val2017 images directory",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/instances_val2017.json"),
        help="Path to COCO instances_val2017.json",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights/rtdetr-r50"),
        help="Path to downloaded RT-DETR weights directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV/JSON output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit dataset to first N images (for quick smoke tests)",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export step",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Target device",
    )
    return parser.parse_args()


def main() -> int:
    """Run Phase 1 pipeline."""
    args = parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        logger.error("CUDA not available. Use --device cpu or ensure GPU is accessible.")
        return 1

    # --- Step 0: Validate prerequisites ---
    if not args.weights_dir.exists() or not (args.weights_dir / "config.json").exists():
        logger.error(
            "Weights not found at %s. Run: uv run python scripts/download_weights.py",
            args.weights_dir,
        )
        return 1

    if not args.images_dir.exists():
        logger.warning("Images dir not found: %s. mAP evaluation will fail.", args.images_dir)

    # --- Step 1: FP32 Baseline ---
    logger.info("=== Stage 1: PyTorch FP32 Baseline ===")

    from benchmark.data.coco_loader import COCODataLoader
    from benchmark.engines.pytorch_engine import PyTorchEngine
    from benchmark.models.rtdetr_adapter import RTDETRAdapter
    from benchmark.utils.logger import ResultLogger

    dataloader = COCODataLoader(
        images_dir=args.images_dir,
        annotations_path=args.annotations,
        limit=args.limit,
    )
    logger.info("Dataset: %d images", len(dataloader))

    adapter = RTDETRAdapter()
    engine = PyTorchEngine(
        model_name="rtdetr-r50",
        adapter=adapter,
        device=args.device,
        score_threshold=0.01,
    )

    # Reset VRAM before loading (BENCH-05)
    engine.reset_vram_tracking()

    engine.load_model(args.weights_dir)

    # run_full_benchmark: warm-up (50 runs fixed) + 1000 measured + mAP eval
    result = engine.run_full_benchmark(dataloader, baseline_map_50_95=0.0)

    logger.info(
        "FP32 Results — latency: %.1f ms | FPS: %.1f | mAP@50:95: %.3f | VRAM: %.0f MB | Size: %.0f MB",
        result.latency_total_ms,
        result.throughput_fps,
        result.map_50_95,
        result.vram_peak_mb,
        result.model_size_mb,
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_logger = ResultLogger(
        csv_path=args.output_dir / "phase1_results.csv",
        json_path=args.output_dir / "phase1_results.json",
    )
    result_logger.log(result)

    # --- Step 2: ONNX Export ---
    if not args.skip_onnx:
        logger.info("=== Stage 2: ONNX Export ===")

        from benchmark.engines.onnx_export import export_to_onnx, simplify_onnx
        from benchmark.models.rtdetr_adapter import RTDetrONNXWrapper

        onnx_raw = args.weights_dir / "rtdetr_r50.onnx"
        onnx_sim = args.weights_dir / "rtdetr_r50_sim.onnx"

        wrapper = RTDetrONNXWrapper(engine.model)
        wrapper.eval()

        dynamic_axes = {
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        }

        export_to_onnx(
            wrapper,
            output_path=onnx_raw,
            input_size=(640, 640),
            opset_version=17,
            dynamic_axes=dynamic_axes,
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
        )
        simplify_onnx(onnx_raw, output_path=onnx_sim)

        logger.info("ONNX artifacts: %s (raw) %s (simplified)", onnx_raw, onnx_sim)

    result_logger.save_json()
    logger.info("Results saved to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Verify:**
```bash
# Smoke run with 5 images (fast, ~2-3 min including ONNX)
uv run python scripts/run_phase1.py --limit 5

# Verify outputs exist
ls results/phase1_results.csv results/phase1_results.json
ls weights/rtdetr-r50/rtdetr_r50_sim.onnx
```
Expected:
- Script exits 0
- `results/phase1_results.csv` exists with 1 data row
- `weights/rtdetr-r50/rtdetr_r50_sim.onnx` exists
- Log shows TF32 disabled, warm-up 50 runs, measure 1000 runs

**Done:** Runner exits 0. CSV + JSON written. ONNX artifact on disk.

---

## Wave 5 — Final Verification Against Success Criteria

**Depends on:** Wave 4 complete
**Goal:** All 5 phase success criteria verified with concrete evidence.

### Task P1-T07: Verification checklist execution

**Files:** None (read-only verification)

**Action:**

Execute each check below in order. All must pass before the phase is considered complete.

**SC-1: Warm-up runs exactly 50 iterations, 1000 measured iterations complete.**
```bash
uv run pytest tests/test_base_engine.py::test_warmup_calls_infer_exactly_once_per_iteration -v
# Must show PASSED
```
Additionally inspect the runner log from Wave 4 — it must show:
- `Warm-up: 50 runs...`
- `Measuring: 1000 runs...`

**SC-2: RT-DETR weights load and produce Detection-format outputs.**
```bash
uv run python -c "
import torch
from pathlib import Path
from benchmark.models.rtdetr_adapter import RTDETRAdapter

adapter = RTDETRAdapter()
device = torch.device('cuda')
model = adapter.load(Path('weights/rtdetr-r50'), device)

# Run one real forward pass
dummy = torch.zeros(1, 3, 640, 640, device=device)
with torch.no_grad():
    out = model(pixel_values=dummy)

from benchmark.engines.base import Detection
det = adapter.parse_outputs(out, original_size=(480, 640), input_size=(640, 640), score_threshold=0.01)
assert isinstance(det, Detection)
assert det.boxes.shape[1] == 4
assert det.scores.dtype.name == 'float32'
assert det.labels.dtype.name == 'int64'
assert 0 not in det.labels, 'No background labels allowed'
print('SC-2 PASSED: Detection boxes=%s scores=%s labels=%s' % (det.boxes.shape, det.scores.shape, det.labels.shape))
"
```

**SC-3: TF32 disabled during FP32 baseline.**
```bash
uv run python -c "
import torch
from pathlib import Path
from benchmark.engines.pytorch_engine import PyTorchEngine
from benchmark.models.rtdetr_adapter import RTDETRAdapter

# Before load: TF32 may be True (default)
engine = PyTorchEngine('rtdetr-r50', RTDETRAdapter(), device='cuda')
engine.load_model(Path('weights/rtdetr-r50'))

matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
cudnn_tf32 = torch.backends.cudnn.allow_tf32
assert not matmul_tf32, f'matmul TF32 should be False, got {matmul_tf32}'
assert not cudnn_tf32, f'cudnn TF32 should be False, got {cudnn_tf32}'
print('SC-3 PASSED: TF32 disabled (matmul=%s, cudnn=%s)' % (matmul_tf32, cudnn_tf32))
"
```

**SC-4: ONNX export passes onnxsim and onnx.checker.**
```bash
uv run python -c "
import onnx
import onnxsim
from pathlib import Path

sim_path = Path('weights/rtdetr-r50/rtdetr_r50_sim.onnx')
assert sim_path.exists(), 'Simplified ONNX not found'
model = onnx.load(str(sim_path))
onnx.checker.check_model(model)
_, check_ok = onnxsim.simplify(model)
print('SC-4 PASSED: onnx.checker OK. onnxsim check_ok=%s' % check_ok)
output_names = [o.name for o in model.graph.output]
assert 'logits' in output_names
assert 'pred_boxes' in output_names
print('Output nodes: %s' % output_names)
"
```

**SC-5: VRAM cache cleared and peak counter reset between engines.**
```bash
uv run python -c "
import torch
from benchmark.engines.base import BaseEngine

# Simulate two engine transitions
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Allocate something
t = torch.zeros(1000, 1000, device='cuda')
peak_before = torch.cuda.max_memory_allocated()
assert peak_before > 0, 'Expected non-zero peak after allocation'

# Reset (as done between engines)
BaseEngine.reset_vram_tracking()
del t
peak_after = torch.cuda.max_memory_allocated()

# After reset, peak counter is 0 (tracks from the reset point)
assert peak_after == 0, f'Expected 0 after reset, got {peak_after}'
print('SC-5 PASSED: VRAM reset works. Peak before=%d bytes, after=%d bytes' % (peak_before, peak_after))
"
```

**Run full test suite:**
```bash
uv run pytest tests/ -v --ignore=tests/test_onnx_export.py  # fast unit tests
# Then full including ONNX (slow):
uv run pytest tests/ -v
```
Expected: All non-skipped tests pass.

**Done:** All 5 success criteria verified with concrete passing output. Phase 1 complete.

---

## Verification Checklist (Success Criteria Map)

| # | Success Criterion | Verified By | Task |
|---|-------------------|-------------|------|
| SC-1 | Warm-up: 50 iterations, no double infer(); 1000 measured complete | `test_warmup_calls_infer_exactly_once_per_iteration` PASSED + runner log | P1-T01, P1-T07 |
| SC-2 | RT-DETR loads from disk, produces Detection (boxes x1y1x2y2, scores, labels COCO-91) | `test_parse_outputs_*` suite + SC-2 smoke script | P1-T03, P1-T07 |
| SC-3 | TF32 disabled: `torch.backends.cuda.matmul.allow_tf32 == False` after `load_model()` | SC-3 python assertion script | existing `PyTorchEngine`, P1-T07 |
| SC-4 | ONNX file passes onnxsim and `onnx.checker.check_model()` | `test_validate_onnx_passes` + SC-4 script | P1-T04, P1-T05, P1-T07 |
| SC-5 | VRAM reset + cache clear between engines via `reset_vram_tracking()` | SC-5 assertion script | existing `BaseEngine.reset_vram_tracking()`, runner `P1-T06` |

---

## File Manifest

| File | Status | Wave | Task |
|------|--------|------|------|
| `pyproject.toml` | Modify (add transformers, pytest) | 0 | P1-T00a, P1-T00b |
| `.gitignore` | Modify (add `weights/`) | 0 | P1-T00b |
| `tests/conftest.py` | Create | 0 | P1-T00b |
| `tests/test_base_engine.py` | Create | 0 | P1-T00b |
| `tests/test_rtdetr_adapter.py` | Create → Update | 0→2 | P1-T00b, P1-T03 |
| `tests/test_onnx_export.py` | Create → Update | 0→3 | P1-T00b, P1-T05 |
| `src/benchmark/engines/base.py` | Modify (FIX-01, lines 96-97) | 1 | P1-T01 |
| `scripts/download_weights.py` | Create | 2 | P1-T02 |
| `src/benchmark/models/__init__.py` | Create | 2 | P1-T03 |
| `src/benchmark/models/rtdetr_adapter.py` | Create | 2 | P1-T03 |
| `src/benchmark/engines/__init__.py` | Modify (add RTDETRAdapter export) | 2 | P1-T03 |
| `src/benchmark/engines/onnx_export.py` | Modify (add input/output_names params) | 3 | P1-T04 |
| `scripts/run_phase1.py` | Create | 4 | P1-T06 |

---

## Critical Path

```
Wave 0 (env + test stubs)
    ↓
Wave 1 (FIX-01 — RED→GREEN)
    ↓
Wave 2 (RTDETRAdapter + weight download)
    ↓
Wave 3 (onnx_export.py update + ONNX pipeline)
    ↓
Wave 4 (runner: end-to-end)
    ↓
Wave 5 (verify all 5 SC)
```

Each wave must be verified before proceeding. The gate command for each wave is stated at the wave header.
