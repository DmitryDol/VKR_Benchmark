"""Benchmark result logging to CSV and JSON."""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.utils.hardware import HardwareInfo

logger = logging.getLogger(__name__)


def _fmt_metric(val: object, dec: int) -> str:
    """Format a numeric metric for the summary table, handling NaN and non-numerics.

    Used by :meth:`ResultLogger.merge_to_unified`. Defined at module scope so
    the helper is created once rather than per row. Uses ``math.isnan`` for the
    NaN check instead of the ``f != f`` self-comparison idiom.

    Parameters
    ----------
    val : object
        Raw cell value from the CSV row dict.
    dec : int
        Number of decimal places.

    Returns
    -------
    str
        Formatted numeric string, ``"NaN"`` for NaN, or ``str(val)`` for
        anything that cannot be coerced to ``float``.
    """
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(val)
    if math.isnan(x):
        return "NaN"
    return f"{x:.{dec}f}"


@dataclass
class BenchmarkResult:
    """Single benchmark run result with full metric set."""

    # Identity
    model_name: str
    stage: str  # e.g. "1_pytorch_fp32"
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

    # Accuracy — all 12 COCOeval stats
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

    # Per-class AP — JSON only; excluded from CSV writes
    per_class_ap: list[dict[str, int | float | str]] = field(default_factory=list)

    # Hardware info — flat columns, pandas-friendly
    hw_gpu: str = ""
    hw_cuda_version: str = ""
    hw_driver_version: str = ""
    hw_trt_version: str = ""  # "" for non-TRT stages

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
            # inject hardware info if not already set
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
        """Write per-stage CSV and JSON for a single result.

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

        # CSV — per_class_ap is a nested list; dropping it keeps the flat schema intact
        row_csv = dict(row)
        row_csv.pop("per_class_ap", None)
        csv_path = stage_dir / f"{result.stage}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row_csv.keys())
            writer.writeheader()
            writer.writerow(row_csv)

        # JSON — keep the full row including per_class_ap
        json_path = stage_dir / f"{result.stage}.json"
        json_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Stage files written: %s, %s", csv_path, json_path)
        return csv_path, json_path

    def save_int8_best_calibrator(
        self,
        model_name: str,
    ) -> Path | None:
        """Compare the three INT8 stage results and write int8_best_calibrator.json.

        Reads ``{run_id}/{stage}.json`` for minmax, entropy, and percentile.
        Picks the calibrator with the highest ``map_50_95``, tie-broken by the
        lower ``latency_total_ms``. NaN / missing stages are
        skipped; a missing/non-numeric latency falls back to ``+inf`` so it
        can never win a tie. Writes the result to::

            results/{model_name}/{run_id}/int8_best_calibrator.json

        and logs the winner at INFO level.

        Parameters
        ----------
        model_name : str
            The model whose INT8 results to compare.

        Returns
        -------
        Path | None
            Path to the written JSON file, or ``None`` if no valid INT8 results
            were found (e.g. all stages skipped or not yet run).
        """
        int8_stages = {
            "minmax": "5_trt_int8_minmax",
            "entropy": "5_trt_int8_entropy",
            "percentile": "5_trt_int8_percentile",
        }
        stage_dir = self.output_dir / model_name / self.run_id

        candidates: list[dict[str, object]] = []
        for method, stage_key in int8_stages.items():
            json_path = stage_dir / f"{stage_key}.json"
            if not json_path.exists():
                continue
            try:
                data: dict[str, object] = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not read INT8 result: %s", json_path)
                continue
            map_val = data.get("map_50_95")
            try:
                map_float = float(map_val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                map_float = float("nan")
            if math.isnan(map_float):
                continue
            # capture latency for the tie-break. Missing/non-numeric/NaN
            # latency falls back to +inf so it can never win a tie.
            lat_val = data.get("latency_total_ms")
            try:
                lat_float = float(lat_val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                lat_float = float("inf")
            if math.isnan(lat_float):
                lat_float = float("inf")
            candidates.append(
                {
                    "calibrator": method,
                    "stage": stage_key,
                    "map_50_95": map_float,
                    "latency_total_ms": lat_float,
                }
            )

        if not candidates:
            logger.warning("No valid INT8 results found — int8_best_calibrator.json not written")
            return None

        # rank by mAP descending, tie-break by latency ascending.
        # `min` with the negated mAP picks the highest-mAP candidate first;
        # on an exact mAP tie the lower latency wins.
        best = min(
            candidates,
            key=lambda x: (-float(x["map_50_95"]), float(x["latency_total_ms"])),
        )
        out: dict[str, object] = {
            "best_calibrator": best["calibrator"],
            "best_stage": best["stage"],
            "map_50_95": best["map_50_95"],
            "latency_total_ms": best["latency_total_ms"],
            "all_candidates": candidates,
        }

        out_path = stage_dir / "int8_best_calibrator.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "Best INT8 calibrator: %s (mAP_50:95=%.4f) — written to %s",
            best["calibrator"],
            float(best["map_50_95"]),
            out_path,
        )
        return out_path

    def merge_to_unified(self, model_name: str) -> tuple[Path, Path]:  # noqa: PLR0912, PLR0915
        """Merge all per-stage CSVs for model_name into unified files.

        Reads: results/{model_name}/{run_id}/*.csv (sorted by filename = stage order)
        Writes:
            results/results.csv                              (aggregated across models)
            results/results.json                             (aggregated across models)
            results/{model_name}/{run_id}/summary.txt        (per-model human-readable)
            results/{model_name}/{run_id}/summary.md         (per-model human-readable)

        The global ``results.csv`` / ``results.json`` files keep rows for every
        previously merged ``(model_name, stage)`` pair; only the rows whose
        ``model_name`` column matches the current call are replaced with the
        freshly merged content. This lets the user call ``benchmark merge`` once
        per model with the same ``--run-id`` without losing earlier models'
        rows.

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

        # Read existing aggregated rows for OTHER models so we do not clobber
        # them when merging this model. Rows whose ``model_name`` matches the
        # current call are dropped — they will be re-added from the freshly
        # read per-stage CSVs below.
        existing_other: list[dict[str, object]] = []
        if unified_csv.exists():
            with unified_csv.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_other = [r for r in reader if r.get("model_name") != model_name]

        merged_rows: list[dict[str, object]] = [*existing_other, *all_rows]

        # Union of fieldnames preserves current-call schema first, then any
        # extra keys carried by older rows (schema drift defensive).
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in merged_rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        if merged_rows:
            with unified_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(merged_rows)

        unified_json.write_text(
            json.dumps(merged_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Extract best stage
        best_stage = ""
        cal_file = model_dir / "int8_best_calibrator.json"
        if cal_file.exists():
            try:
                best_stage = json.loads(cal_file.read_text(encoding="utf-8")).get("best_stage", "")
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not parse %s: %s — summary will omit winner mark",
                    cal_file,
                    exc,
                )

        # Per-model human-readable summary lands under the model's run dir so
        # repeated merge calls for sibling models do not overwrite each other.
        txt_path = model_dir / "summary.txt"
        md_path = model_dir / "summary.md"

        headers = ["Stage", "mAP@50:95", "Latency (ms)", "FPS", "Drop %", "Size (MB)"]
        rows = []
        for r in all_rows:
            st = str(r.get("stage", ""))
            if st == best_stage and best_stage:
                st += " ★"

            rows.append(
                [
                    st,
                    _fmt_metric(r.get("map_50_95"), 3),
                    _fmt_metric(r.get("latency_total_ms"), 1),
                    _fmt_metric(r.get("throughput_fps"), 1),
                    _fmt_metric(r.get("accuracy_drop_pct"), 1),
                    _fmt_metric(r.get("model_size_mb"), 1),
                ]
            )

        # Write summary.txt
        col_widths = [
            max(len(str(item)) for item in col) for col in zip(headers, *rows, strict=False)
        ]
        txt_lines = []
        txt_lines.append(" | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=False)))
        txt_lines.append("-+-".join("-" * w for w in col_widths))
        for row in rows:
            txt_lines.append(
                " | ".join(str(item).ljust(w) for item, w in zip(row, col_widths, strict=False))
            )
        txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

        # Write summary.md
        md_lines = []
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("|-" + "-|-".join("-" * len(h) for h in headers) + "-|")
        for row in rows:
            md_lines.append("| " + " | ".join(str(item) for item in row) + " |")
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

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
        # per_class_ap is a nested list; dropping it keeps the flat CSV schema intact
        row.pop("per_class_ap", None)
        file_exists = csv_path.exists()

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
