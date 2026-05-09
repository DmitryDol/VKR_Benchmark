"""Benchmark result logging to CSV and JSON."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""

    model_name: str
    engine_type: str           # "pytorch" | "onnx" | "tensorrt"
    precision: str             # "fp32" | "fp16" | "bf16" | "int8"

    # Latency
    latency_preprocess_ms: float
    latency_inference_ms: float
    latency_postprocess_ms: float
    latency_total_ms: float

    # Throughput & Jitter
    throughput_fps: float
    jitter_ms: float

    # Accuracy
    map_50: float
    map_50_95: float
    accuracy_drop_pct: float   # relative to FP32 baseline

    # Resources
    model_size_mb: float
    vram_peak_mb: float
    macs: float | None = None
    flops: float | None = None

    # Meta
    timestamp: str = ""
    warmup_runs: int = 50
    measure_runs: int = 1000

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(tz=UTC).isoformat()


class ResultLogger:
    """Writes benchmark results to CSV and JSON files."""

    def __init__(self, output_dir: Path = Path("results")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: list[BenchmarkResult] = []

    def add(self, result: BenchmarkResult) -> None:
        """Add a result and immediately append to CSV."""
        self._results.append(result)
        self._append_csv(result)
        logger.info(
            "Result logged: %s/%s/%s — mAP50=%.4f, latency=%.2fms",
            result.model_name,
            result.engine_type,
            result.precision,
            result.map_50,
            result.latency_total_ms,
        )

    def save_json(self, filename: str = "results.json") -> Path:
        """Save all accumulated results to a JSON file."""
        path = self.output_dir / filename
        data = [asdict(r) for r in self._results]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("JSON results saved to %s", path)
        return path

    def _append_csv(self, result: BenchmarkResult) -> None:
        """Append a single result row to the CSV file."""
        csv_path = self.output_dir / "results.csv"
        row = asdict(result)
        file_exists = csv_path.exists()

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
