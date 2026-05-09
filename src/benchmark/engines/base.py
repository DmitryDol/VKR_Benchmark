"""Abstract base engine for inference benchmarking."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

from benchmark.utils.logger import BenchmarkResult

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from benchmark.data.coco_loader import COCODataLoader, COCOSample

logger = logging.getLogger(__name__)

WARMUP_RUNS: int = 50
MEASURE_RUNS: int = 1000


@dataclass
class Detection:
    """Single detection result."""

    boxes: NDArray[np.float32]    # (N, 4) x1y1x2y2
    scores: NDArray[np.float32]   # (N,)
    labels: NDArray[np.int64]     # (N,) COCO 91-class IDs


class BaseEngine(ABC):
    """Abstract inference engine with built-in benchmarking.

    Subclasses must implement:
        - load_model(): load and prepare the model/engine
        - preprocess(sample): image → model input tensor(s)
        - infer(inputs): run forward pass, return raw outputs
        - postprocess(raw_outputs, sample): raw outputs → Detection
        - model_size_mb: property returning model file size in MB

    Benchmarking protocol (per CLAUDE.md):
        - 50 warm-up runs, then 1000 measured iterations
        - Latency split: preprocess + inference + postprocess
        - VRAM via torch.cuda.max_memory_allocated()
    """

    def __init__(self, model_name: str, engine_type: str, precision: str) -> None:
        self.model_name = model_name
        self.engine_type = engine_type
        self.precision = precision

    @abstractmethod
    def load_model(self, weights_path: Path) -> None:
        """Load model weights and prepare for inference."""

    @abstractmethod
    def preprocess(self, sample: COCOSample) -> object:
        """Preprocess image for the model. Returns model-specific input."""

    @abstractmethod
    def infer(self, inputs: object) -> object:
        """Run model inference. Returns raw model output."""

    @abstractmethod
    def postprocess(self, raw_outputs: object, sample: COCOSample) -> Detection:
        """Convert raw model outputs to Detection."""

    @property
    @abstractmethod
    def model_size_mb(self) -> float:
        """Model file size in megabytes."""

    def benchmark_latency(self, dataloader: COCODataLoader) -> dict[str, float]:
        """Measure latency with warm-up and averaging.

        Returns dict with keys:
            preprocess_ms, inference_ms, postprocess_ms, total_ms, jitter_ms, fps
        """
        # Collect samples (reuse for consistent measurement)
        samples = [dataloader[i] for i in range(min(MEASURE_RUNS, len(dataloader)))]
        n_samples = len(samples)

        # --- Warm-up phase: 50 runs ---
        logger.info("Warm-up: %d runs...", WARMUP_RUNS)
        for i in range(WARMUP_RUNS):
            sample = samples[i % n_samples]
            inputs = self.preprocess(sample)
            self.infer(inputs)
            self.postprocess(self.infer(inputs), sample)

        # Sync GPU before measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # --- Measurement phase: 1000 runs ---
        logger.info("Measuring: %d runs...", MEASURE_RUNS)
        times_preprocess: list[float] = []
        times_inference: list[float] = []
        times_postprocess: list[float] = []

        for i in range(MEASURE_RUNS):
            sample = samples[i % n_samples]

            # Preprocess
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            inputs = self.preprocess(sample)

            # Inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            raw_outputs = self.infer(inputs)

            # Postprocess
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            self.postprocess(raw_outputs, sample)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            times_preprocess.append((t1 - t0) * 1000)
            times_inference.append((t2 - t1) * 1000)
            times_postprocess.append((t3 - t2) * 1000)

        pre = np.array(times_preprocess, dtype=np.float64)
        inf = np.array(times_inference, dtype=np.float64)
        post = np.array(times_postprocess, dtype=np.float64)
        total = pre + inf + post

        return {
            "preprocess_ms": float(np.mean(pre)),
            "inference_ms": float(np.mean(inf)),
            "postprocess_ms": float(np.mean(post)),
            "total_ms": float(np.mean(total)),
            "jitter_ms": float(np.std(total)),
            "fps": float(1000.0 / np.mean(total)),
        }

    def evaluate_accuracy(self, dataloader: COCODataLoader) -> dict[str, float]:
        """Evaluate mAP on the full dataset using COCO API.

        Returns dict with keys: map_50, map_50_95
        """
        coco_results: list[dict[str, float | int]] = []

        logger.info("Evaluating accuracy on %d images...", len(dataloader))
        for sample in dataloader:
            inputs = self.preprocess(sample)
            raw_outputs = self.infer(inputs)
            detection = self.postprocess(raw_outputs, sample)

            for j in range(len(detection.scores)):
                x1, y1, x2, y2 = detection.boxes[j]
                coco_results.append({
                    "image_id": sample.image_id,
                    "category_id": int(detection.labels[j]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(detection.scores[j]),
                })

        if not coco_results:
            logger.warning("No detections produced!")
            return {"map_50": 0.0, "map_50_95": 0.0}

        coco_dt = dataloader.coco.loadRes(coco_results)
        coco_eval = COCOeval(dataloader.coco, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "map_50_95": float(coco_eval.stats[0]),  # AP @ IoU=0.50:0.95
            "map_50": float(coco_eval.stats[1]),      # AP @ IoU=0.50
        }

    def measure_vram(self) -> float:
        """Return peak VRAM usage in MB since last reset."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    @staticmethod
    def reset_vram_tracking() -> None:
        """Reset VRAM peak tracking and clear cache."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def run_full_benchmark(
        self,
        dataloader: COCODataLoader,
        baseline_map_50_95: float = 0.0,
    ) -> BenchmarkResult:
        """Run complete benchmark: latency + accuracy + VRAM.

        Parameters
        ----------
        dataloader : COCODataLoader
            Data loader with COCO val2017 images.
        baseline_map_50_95 : float
            FP32 baseline mAP for accuracy drop calculation.
        """
        self.reset_vram_tracking()

        # Latency benchmark
        latency = self.benchmark_latency(dataloader)

        # VRAM after latency (model loaded + inference peak)
        vram_mb = self.measure_vram()

        # Accuracy evaluation
        accuracy = self.evaluate_accuracy(dataloader)

        # Accuracy drop
        drop = 0.0
        if baseline_map_50_95 > 0:
            drop = (1.0 - accuracy["map_50_95"] / baseline_map_50_95) * 100.0

        return BenchmarkResult(
            model_name=self.model_name,
            engine_type=self.engine_type,
            precision=self.precision,
            latency_preprocess_ms=latency["preprocess_ms"],
            latency_inference_ms=latency["inference_ms"],
            latency_postprocess_ms=latency["postprocess_ms"],
            latency_total_ms=latency["total_ms"],
            throughput_fps=latency["fps"],
            jitter_ms=latency["jitter_ms"],
            map_50=accuracy["map_50"],
            map_50_95=accuracy["map_50_95"],
            accuracy_drop_pct=drop,
            model_size_mb=self.model_size_mb,
            vram_peak_mb=vram_mb,
            warmup_runs=WARMUP_RUNS,
            measure_runs=MEASURE_RUNS,
        )
