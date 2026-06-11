"""Tests for BaseEngine benchmarking protocol."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from benchmark.engines.base import WARMUP_RUNS, BaseEngine, Detection


class ConcreteEngine(BaseEngine):
    """Minimal concrete subclass for testing abstract BaseEngine."""

    def load_model(self, weights_path):
        pass

    def preprocess(self, sample):
        return torch.zeros(1, 3, 640, 640)

    def infer(self, inputs):
        return MagicMock()

    def postprocess(self, raw_outputs, sample):
        return Detection(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            labels=np.zeros(0, dtype=np.int64),
        )

    @property
    def model_size_mb(self) -> float:
        return 0.0


def test_warmup_calls_infer_exactly_once_per_iteration(dummy_sample):
    """Warm-up loop must call infer() exactly once per iteration."""
    engine = ConcreteEngine("test", "pytorch", "fp32")

    infer_call_count = 0
    original_infer = engine.infer

    def counting_infer(inputs):
        nonlocal infer_call_count
        infer_call_count += 1
        return original_infer(inputs)

    engine.infer = counting_infer  # type: ignore[method-assign]

    # Build a minimal dataloader mock with one sample.
    # __len__ must return >= WARMUP_RUNS so samples list is non-empty.
    dataloader = MagicMock()
    dataloader.__len__ = MagicMock(return_value=WARMUP_RUNS)
    dataloader.__getitem__ = MagicMock(return_value=dummy_sample)
    dataloader.__iter__ = MagicMock(return_value=iter([dummy_sample] * WARMUP_RUNS))

    # Run only warm-up + measurement with 0 measurement runs via raising after warm-up.
    # We do this by intercepting _after_ the warm-up loop with a sentinel exception.
    from unittest.mock import patch

    call_order: list[str] = []

    original_time_perf = __import__("time").perf_counter

    measurement_started = False

    def side_effect_sync():
        nonlocal measurement_started
        measurement_started = True
        # Raise after sync that starts measurement, to short-circuit measurement loop.
        raise KeyboardInterrupt("stop after warmup sync")

    # Patch cuda.synchronize to raise after warm-up sync (first post-warmup sync).
    sync_call_count = 0

    original_sync = torch.cuda.synchronize

    def counting_sync():
        nonlocal sync_call_count
        sync_call_count += 1
        # The post-warmup sync (sync_call_count == 1) should stop execution.
        if sync_call_count == 1:
            raise KeyboardInterrupt("stop after warmup")

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.synchronize", side_effect=counting_sync
    ):
        try:
            engine.benchmark_latency(dataloader)
        except KeyboardInterrupt:
            pass  # Expected — we interrupted after warm-up

    assert infer_call_count == WARMUP_RUNS, (
        f"Expected {WARMUP_RUNS} infer() calls during warm-up, got {infer_call_count}. "
        "warm-up loop calls infer() twice per iteration."
    )
