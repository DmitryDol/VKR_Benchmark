"""Tests for BenchmarkResult schema and ResultLogger stage file output."""
import csv
import json
from pathlib import Path

import pytest

from benchmark.utils.logger import BenchmarkResult, ResultLogger


def _make_result(stage: str = "1_pytorch_fp32") -> BenchmarkResult:
    """Construct a minimal BenchmarkResult with all required fields."""
    return BenchmarkResult(
        model_name="test-model",
        stage=stage,
        engine_type="pytorch",
        precision="fp32",
        latency_preprocess_ms=1.0,
        latency_inference_ms=10.0,
        latency_postprocess_ms=2.0,
        latency_total_ms=13.0,
        throughput_fps=76.9,
        jitter_ms=0.5,
        map_50_95=0.42,
        map_50=0.60,
        map_75=0.45,
        map_small=0.22,
        map_medium=0.46,
        map_large=0.58,
        ar_1=0.33,
        ar_10=0.50,
        ar_100=0.52,
        ar_small=0.30,
        ar_medium=0.55,
        ar_large=0.67,
        accuracy_drop_pct=0.0,
        model_size_mb=85.0,
        vram_peak_mb=1200.0,
    )


def test_benchmark_result_has_all_new_fields() -> None:
    """All 12 COCO stats, 4 hw fields, and stage field must exist."""
    import dataclasses

    result = _make_result()
    fields = {f.name for f in dataclasses.fields(result)}

    required_new = {
        "stage", "map_75", "map_small", "map_medium", "map_large",
        "ar_1", "ar_10", "ar_100", "ar_small", "ar_medium", "ar_large",
        "hw_gpu", "hw_cuda_version", "hw_driver_version", "hw_trt_version",
    }
    assert not (required_new - fields), f"Missing fields: {required_new - fields}"


def test_benchmark_result_hw_trt_version_default_empty_string() -> None:
    """hw_trt_version must default to '' (not None) per D-02."""
    result = _make_result()
    assert result.hw_trt_version == ""
    assert result.hw_trt_version is not None


def test_save_stage_files_creates_csv_and_json(tmp_path: Path) -> None:
    """save_stage_files must create results/{model}/{stage}.csv and .json."""
    log = ResultLogger(output_dir=tmp_path)
    result = _make_result("1_pytorch_fp32")
    csv_path, json_path = log.save_stage_files(result)

    assert csv_path.exists(), "Stage CSV not created"
    assert json_path.exists(), "Stage JSON not created"
    assert csv_path == tmp_path / "test-model" / log.run_id / "1_pytorch_fp32.csv"
    assert json_path == tmp_path / "test-model" / log.run_id / "1_pytorch_fp32.json"


def test_save_stage_files_csv_has_all_fields(tmp_path: Path) -> None:
    """Stage CSV header must include all BenchmarkResult fields."""
    log = ResultLogger(output_dir=tmp_path)
    result = _make_result()
    csv_path, _ = log.save_stage_files(result)

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])

    assert "stage" in headers
    assert "map_75" in headers
    assert "hw_gpu" in headers


def test_merge_to_unified_combines_stage_files(tmp_path: Path) -> None:
    """merge_to_unified must combine per-stage CSVs into results.csv."""
    log = ResultLogger(output_dir=tmp_path)
    r1 = _make_result("1_pytorch_fp32")
    r2 = _make_result("2_onnx_fp32")
    log.save_stage_files(r1)
    log.save_stage_files(r2)

    unified_csv, unified_json = log.merge_to_unified("test-model")
    assert unified_csv.exists()
    assert unified_json.exists()

    with unified_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    stages = {r["stage"] for r in rows}
    assert stages == {"1_pytorch_fp32", "2_onnx_fp32"}


def test_result_logger_injects_hardware_info(tmp_path: Path) -> None:
    """ResultLogger.add() must inject hw_* fields from HardwareInfo (D-03)."""
    from benchmark.utils.hardware import HardwareInfo

    hw = HardwareInfo(
        gpu_name="Test GPU",
        cuda_version="12.1",
        driver_version="537.13",
        trt_version="",
    )
    log = ResultLogger(output_dir=tmp_path, hardware=hw)
    result = _make_result()
    assert result.hw_gpu == ""  # not set yet

    log.add(result)
    assert result.hw_gpu == "Test GPU"
    assert result.hw_cuda_version == "12.1"
    assert result.hw_trt_version == ""  # stays "" for stage 1 per D-02


def test_stage_file_contains_hw_fields_after_add(tmp_path: Path) -> None:
    """SC-5: stage JSON written after add() must contain non-empty hw_gpu (D-03 + D-05)."""
    from benchmark.utils.hardware import HardwareInfo

    hw = HardwareInfo(
        gpu_name="NVIDIA GeForce RTX 3070",
        cuda_version="13.0",
        driver_version="555.42",
        trt_version="",
    )
    log = ResultLogger(output_dir=tmp_path, hardware=hw)
    result = _make_result("1_pytorch_fp32")

    # Full production flow: add() injects hw, then save_stage_files writes file
    log.add(result)
    _, json_path = log.save_stage_files(result)

    data = json.loads(json_path.read_text())
    assert data["hw_gpu"] == "NVIDIA GeForce RTX 3070", "hw_gpu must be in stage JSON"
    assert data["hw_cuda_version"] == "13.0"
    assert data["hw_trt_version"] == ""  # D-02: "" for stage 1, not None


def test_json_stage_file_has_correct_stage_value(tmp_path: Path) -> None:
    """Stage JSON must have stage == stage name (D-04)."""
    log = ResultLogger(output_dir=tmp_path)
    result = _make_result("2_onnx_fp32")
    _, json_path = log.save_stage_files(result)
    data = json.loads(json_path.read_text())
    assert data["stage"] == "2_onnx_fp32"


def _write_int8_stage_json(
    stage_dir: Path, stage_key: str, map_50_95: float, latency_total_ms: float
) -> None:
    """Write a minimal per-stage INT8 JSON consumable by save_int8_best_calibrator."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage_key,
        "map_50_95": map_50_95,
        "latency_total_ms": latency_total_ms,
    }
    (stage_dir / f"{stage_key}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_save_int8_best_calibrator_tie_breaks_by_latency(tmp_path: Path) -> None:
    """D-12: when two calibrators share the highest mAP, the lower
    latency_total_ms wins.
    """
    log = ResultLogger(output_dir=tmp_path, run_id="t")
    stage_dir = tmp_path / "test-model" / "t"

    # minmax + entropy tie on mAP; entropy has lower latency, so entropy wins.
    # percentile has strictly lower mAP — must lose regardless of its latency.
    _write_int8_stage_json(stage_dir, "5_trt_int8_minmax", map_50_95=0.42, latency_total_ms=20.0)
    _write_int8_stage_json(stage_dir, "5_trt_int8_entropy", map_50_95=0.42, latency_total_ms=15.0)
    _write_int8_stage_json(
        stage_dir, "5_trt_int8_percentile", map_50_95=0.40, latency_total_ms=5.0
    )

    out_path = log.save_int8_best_calibrator("test-model")
    assert out_path is not None
    data = json.loads(out_path.read_text())

    assert data["best_calibrator"] == "entropy", (
        "D-12: lower latency must win on an exact mAP tie"
    )
    assert data["best_stage"] == "5_trt_int8_entropy"
    # Verify all_candidates carries latency_total_ms for downstream auditability.
    assert all("latency_total_ms" in c for c in data["all_candidates"])


def test_save_int8_best_calibrator_picks_highest_map(tmp_path: Path) -> None:
    """A strictly higher mAP wins regardless of latency — the tie-break MUST
    NOT override a real mAP difference.
    """
    log = ResultLogger(output_dir=tmp_path, run_id="t2")
    stage_dir = tmp_path / "test-model" / "t2"

    # entropy has highest mAP but slowest latency — must still win.
    _write_int8_stage_json(stage_dir, "5_trt_int8_minmax", map_50_95=0.40, latency_total_ms=5.0)
    _write_int8_stage_json(stage_dir, "5_trt_int8_entropy", map_50_95=0.45, latency_total_ms=30.0)
    _write_int8_stage_json(
        stage_dir, "5_trt_int8_percentile", map_50_95=0.42, latency_total_ms=10.0
    )

    out_path = log.save_int8_best_calibrator("test-model")
    assert out_path is not None
    data = json.loads(out_path.read_text())

    assert data["best_calibrator"] == "entropy"
    assert data["best_stage"] == "5_trt_int8_entropy"
    assert float(data["map_50_95"]) == pytest.approx(0.45)


def test_save_int8_best_calibrator_missing_latency_falls_back_to_inf(tmp_path: Path) -> None:
    """A candidate with missing/non-numeric latency_total_ms cannot win a tie
    (latency falls back to +inf). With an exact mAP tie, the candidate that
    HAS a finite latency must win.
    """
    log = ResultLogger(output_dir=tmp_path, run_id="t3")
    stage_dir = tmp_path / "test-model" / "t3"

    # minmax — tied on mAP, missing latency_total_ms (will fall back to +inf).
    bad_payload = {"stage": "5_trt_int8_minmax", "map_50_95": 0.42}
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "5_trt_int8_minmax.json").write_text(json.dumps(bad_payload), encoding="utf-8")

    _write_int8_stage_json(stage_dir, "5_trt_int8_entropy", map_50_95=0.42, latency_total_ms=20.0)
    _write_int8_stage_json(
        stage_dir, "5_trt_int8_percentile", map_50_95=0.40, latency_total_ms=5.0
    )

    out_path = log.save_int8_best_calibrator("test-model")
    assert out_path is not None
    data = json.loads(out_path.read_text())

    assert data["best_calibrator"] == "entropy", (
        "Missing latency must fall back to +inf and cannot win a tie"
    )


def test_save_int8_best_calibrator_returns_none_when_no_results(tmp_path: Path) -> None:
    """Regression guard: when no INT8 stage JSONs exist, the function logs a
    warning and returns ``None`` — does not crash and does not write a file.
    """
    log = ResultLogger(output_dir=tmp_path, run_id="empty")
    # Don't write any per-stage JSON files.
    out_path = log.save_int8_best_calibrator("test-model")
    assert out_path is None
    assert not (tmp_path / "test-model" / "empty" / "int8_best_calibrator.json").exists()
