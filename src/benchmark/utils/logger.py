"""Benchmark result logging to CSV and JSON."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.utils.hardware import HardwareInfo

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark run result with full metric set."""

    # Identity
    model_name: str
    stage: str  # e.g. "1_pytorch_fp32" (D-04)
    engine_type: str  # "pytorch" | "onnx" | "tensorrt"
    precision: str  # "fp32" | "fp16" | "bf16" | "int8"

    # Latency (ms)
    latency_preprocess_ms: float
    latency_inference_ms: float
    latency_postprocess_ms: float
    latency_total_ms: float

    # Throughput & Jitter
    throughput_fps: float
    jitter_ms: float

    # Accuracy — all 12 COCOeval stats (D-10/D-11)
    map_50_95: float  # AP @ IoU=0.50:0.95  (stats[0])
    map_50: float  # AP @ IoU=0.50        (stats[1])
    map_75: float  # AP @ IoU=0.75        (stats[2])
    map_small: float  # AP, area=small       (stats[3])
    map_medium: float  # AP, area=medium      (stats[4])
    map_large: float  # AP, area=large       (stats[5])
    ar_1: float  # AR @ maxDets=1       (stats[6])
    ar_10: float  # AR @ maxDets=10      (stats[7])
    ar_100: float  # AR @ maxDets=100     (stats[8])
    ar_small: float  # AR, area=small       (stats[9])
    ar_medium: float  # AR, area=medium      (stats[10])
    ar_large: float  # AR, area=large       (stats[11])

    # Accuracy derived
    accuracy_drop_pct: float

    # Resources
    model_size_mb: float
    vram_peak_mb: float
    macs: float | None = None
    flops: float | None = None

    # Hardware info (D-01) — flat columns, pandas-friendly
    hw_gpu: str = ""
    hw_cuda_version: str = ""
    hw_driver_version: str = ""
    hw_trt_version: str = ""  # "" for stages 1-2 (D-02)

    # Meta
    timestamp: str = ""
    warmup_runs: int = 50
    measure_runs: int = 1000
    skipped_reason: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(tz=UTC).isoformat()


class ResultLogger:
    """Writes benchmark results to CSV and JSON files."""

    def __init__(
        self,
        output_dir: Path = Path("results"),
        hardware: HardwareInfo | None = None,
        run_id: str = "",
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Auto-generate run_id from timestamp if not provided; sanitize against path traversal
        raw_id = run_id if run_id else datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        self.run_id: str = Path(raw_id).name
        self._results: list[BenchmarkResult] = []
        self._hardware = hardware

    def add(self, result: BenchmarkResult) -> None:
        """Add a result, inject hardware info, and immediately append to CSV."""
        if self._hardware is not None and not result.hw_gpu:
            # inject hardware info if not already set (D-03)
            result.hw_gpu = self._hardware.gpu_name
            result.hw_cuda_version = self._hardware.cuda_version
            result.hw_driver_version = self._hardware.driver_version
            result.hw_trt_version = self._hardware.trt_version
        self._results.append(result)
        self._append_csv(result)
        logger.info(
            "Result logged: %s/%s — mAP50=%.4f, latency=%.2fms",
            result.model_name,
            result.stage,
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

    def save_stage_files(self, result: BenchmarkResult) -> tuple[Path, Path]:
        """Write per-stage CSV and JSON for a single result (D-05).

        Files are written to:
            results/{model_name}/{run_id}/{stage}.csv
            results/{model_name}/{run_id}/{stage}.json

        Parameters
        ----------
        result : BenchmarkResult
            The benchmark result to persist.

        Returns
        -------
        tuple[Path, Path]
            (csv_path, json_path)
        """
        stage_dir = self.output_dir / result.model_name / self.run_id
        stage_dir.mkdir(parents=True, exist_ok=True)

        row = asdict(result)

        # CSV
        csv_path = stage_dir / f"{result.stage}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)

        # JSON
        json_path = stage_dir / f"{result.stage}.json"
        json_path.write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info("Stage files written: %s, %s", csv_path, json_path)
        return csv_path, json_path

    def merge_to_unified(self, model_name: str) -> tuple[Path, Path]:
        """Merge all per-stage CSVs for model_name into unified files (D-06).

        Reads: results/{model_name}/{run_id}/*.csv (sorted by filename = stage order)
        Writes:
            results/results.csv   (overwrites with merged content)
            results/results.json  (overwrites with merged content)

        Parameters
        ----------
        model_name : str
            The model whose per-stage files to merge.

        Returns
        -------
        tuple[Path, Path]
            (unified_csv_path, unified_json_path)

        Raises
        ------
        FileNotFoundError
            If no stage results directory or CSV files are found.
        """
        model_dir = self.output_dir / model_name / self.run_id
        if not model_dir.exists():
            msg = f"No stage results found for model '{model_name}' at {model_dir}"
            raise FileNotFoundError(msg)

        stage_csvs = sorted(model_dir.glob("*.csv"))
        if not stage_csvs:
            msg = f"No .csv files found in {model_dir}"
            raise FileNotFoundError(msg)

        all_rows: list[dict[str, object]] = []
        for csv_path in stage_csvs:
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_rows.extend(list(reader))

        unified_csv = self.output_dir / "results.csv"
        unified_json = self.output_dir / "results.json"

        # Write unified CSV (overwrite)
        if all_rows:
            with unified_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)

        # Write unified JSON (overwrite)
        unified_json.write_text(
            json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info(
            "Merged %d stage(s) for '%s' -> %s, %s",
            len(all_rows),
            model_name,
            unified_csv,
            unified_json,
        )
        return unified_csv, unified_json

    def _append_csv(self, result: BenchmarkResult) -> None:
        """Append a single result row to the unified CSV file."""
        csv_path = self.output_dir / "results.csv"
        row = asdict(result)
        file_exists = csv_path.exists()

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
