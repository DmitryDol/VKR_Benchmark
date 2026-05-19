"""Per-class AP summary tables: top-10 by AP drop and top-10 by frequency.

Generates 8 CSV files and 8 Markdown files:
    results/per_class/<model>_drop_top10.{csv,md}  — sorted by worst AP drop
    results/per_class/<model>_freq_top10.{csv,md}  — sorted by n_gt descending

Requires the 35 valid stage JSONs to have been backfilled with ``per_class_ap``
by ``scripts/build_per_class_ap.py`` before running.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

np.random.seed(0)
random.seed(0)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants (locked in plan 13.03)
# ---------------------------------------------------------------------------

MODELS_ORDER: tuple[str, ...] = ("rt-detr", "yolo11l", "yolo26l", "rfdetr-l")

VARIANT_DIRS: dict[str, str] = {
    "rt-detr": "quant",
    "yolo11l": "quant",
    "yolo26l": "quant",
    "rfdetr-l": "rfdetr_v1",
}

# RF-DETR-L INT8/Mixed stages whose TRT auto-tuner rolled back to FP16 —
# excluded from all Phase 13 artifacts; these columns are written as "n/a".
RFDETR_L_INVALID_STAGES: frozenset[str] = frozenset(
    {
        "5_trt_int8_entropy",
        "5_trt_int8_minmax",
        "5_trt_int8_percentile",
        "6_trt_mixed_a",
        "6_trt_mixed_b",
    }
)

# Column → stage mapping (ordered).  Column order defines output column order.
# ONNX FP32 (2_onnx_fp32) is intentionally omitted — per CONTEXT.md decision.
COLUMN_TO_STAGE: tuple[tuple[str, str], ...] = (
    ("AP_FP32", "1_pytorch_fp32"),
    ("AP_TF32", "3_trt_tf32"),
    ("AP_FP16", "4_trt_fp16"),
    ("AP_BF16", "4_trt_bf16"),
    ("AP_INT8_Entropy", "5_trt_int8_entropy"),
    ("AP_INT8_MinMax", "5_trt_int8_minmax"),
    ("AP_INT8_Percentile", "5_trt_int8_percentile"),
    ("AP_Mixed1", "6_trt_mixed_a"),
    ("AP_Mixed2", "6_trt_mixed_b"),
)

app = typer.Typer(
    name="per-class-summary",
    help="Generate per-class AP summary tables (CSV + Markdown) for 4 models.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_per_class(
    model: str,
    stage: str,
    results_root: Path,
) -> dict[int, dict[str, object]] | None:
    """Load and index the per_class_ap list from a stage JSON.

    Parameters
    ----------
    model : str
        Model key (e.g. ``"rt-detr"``).
    stage : str
        Stage file stem (e.g. ``"1_pytorch_fp32"``).
    results_root : Path
        Root directory containing ``<model>/<variant>/`` sub-trees.

    Returns
    -------
    dict[int, dict[str, object]] | None
        Mapping ``class_id`` → entry dict, or ``None`` if the JSON is missing
        or its ``per_class_ap`` list is empty (not yet backfilled).
    """
    json_path = results_root / model / VARIANT_DIRS[model] / f"{stage}.json"
    if not json_path.exists():
        return None
    data: dict[str, object] = json.loads(json_path.read_text(encoding="utf-8"))
    entries: list[dict[str, object]] = data.get("per_class_ap", [])  # type: ignore[assignment]
    if not entries:
        return None
    return {int(e["class_id"]): e for e in entries}  # type: ignore[arg-type]


def _build_table(
    model: str,
    results_root: Path,
) -> tuple[list[str], list[list[object]]]:
    """Build the full 80-row per-class AP table for one model.

    Parameters
    ----------
    model : str
        Model key.
    results_root : Path
        Root directory containing stage JSON files.

    Returns
    -------
    tuple[list[str], list[list[object]]]
        ``(headers, rows)`` where ``headers`` has 12 entries and ``rows``
        contains 80 rows sorted ascending by ``class_id``.

    Raises
    ------
    typer.Exit
        Exit code 2 when the FP32 baseline JSON is missing or its
        ``per_class_ap`` field is empty (not yet backfilled by
        ``scripts/build_per_class_ap.py``).
    """
    fp32_data = _load_per_class(model, "1_pytorch_fp32", results_root)
    if fp32_data is None:
        typer.echo(
            f"[ERROR] {model}: FP32 baseline missing or not backfilled — "
            "run scripts/build_per_class_ap.py first",
            err=True,
        )
        raise typer.Exit(code=2)

    col_names: list[str] = [col for col, _ in COLUMN_TO_STAGE]
    headers: list[str] = ["class_name", "n_gt", *col_names, "min_delta_AP"]

    # Pre-load per-stage data (None if unavailable).
    stage_data: dict[str, dict[int, dict[str, object]] | None] = {}
    for col_name, stage in COLUMN_TO_STAGE:
        if model == "rfdetr-l" and stage in RFDETR_L_INVALID_STAGES:
            stage_data[col_name] = None  # will be emitted as "n/a"
        else:
            stage_data[col_name] = _load_per_class(model, stage, results_root)

    # Build one row per class_id, ascending.
    rows: list[list[object]] = []
    for class_id in sorted(fp32_data):
        fp32_entry = fp32_data[class_id]
        class_name: str = str(fp32_entry["class_name"])
        n_gt: int = int(fp32_entry["n_gt"])  # type: ignore[arg-type]
        ap_fp32: float = float(fp32_entry["ap_50_95"])  # type: ignore[arg-type]

        row: list[object] = [class_name, n_gt]
        numeric_deltas: list[float] = []

        for col_name, stage in COLUMN_TO_STAGE:
            if model == "rfdetr-l" and stage in RFDETR_L_INVALID_STAGES:
                row.append("n/a")
                continue

            sdata = stage_data[col_name]
            if sdata is None or class_id not in sdata:
                # Stage JSON missing — treat as n/a rather than crashing
                row.append("n/a")
                continue

            ap_val: float = float(sdata[class_id]["ap_50_95"])  # type: ignore[arg-type]
            row.append(round(ap_val, 3))

            # Collect delta vs FP32 baseline for all non-FP32 stages
            if col_name != "AP_FP32":
                numeric_deltas.append(ap_val - ap_fp32)

        # min_delta_AP = worst (most negative) drop across all valid stages
        min_delta = round(min(numeric_deltas), 3) if numeric_deltas else 0.0
        row.append(min_delta)
        rows.append(row)

    return headers, rows


def _sort_top10_drop(
    rows: list[list[object]],
    headers: list[str],
) -> list[list[object]]:
    """Return top 10 rows sorted by min_delta_AP ascending (worst drop first).

    Parameters
    ----------
    rows : list[list[object]]
        Full 80-row table from ``_build_table``.
    headers : list[str]
        Column header list used to locate the ``min_delta_AP`` index.

    Returns
    -------
    list[list[object]]
        10 rows with the worst (most negative) ``min_delta_AP`` values first.
    """
    idx = headers.index("min_delta_AP")
    sorted_rows = sorted(rows, key=lambda r: float(r[idx]))  # type: ignore[arg-type]
    return sorted_rows[:10]


def _sort_top10_freq(
    rows: list[list[object]],
    headers: list[str],
) -> list[list[object]]:
    """Return top 10 rows sorted by n_gt descending (most frequent first).

    Parameters
    ----------
    rows : list[list[object]]
        Full 80-row table from ``_build_table``.
    headers : list[str]
        Column header list used to locate the ``n_gt`` index.

    Returns
    -------
    list[list[object]]
        10 rows with the highest ground-truth annotation counts first.
    """
    idx = headers.index("n_gt")
    sorted_rows = sorted(rows, key=lambda r: int(r[idx]), reverse=True)  # type: ignore[arg-type]
    return sorted_rows[:10]


def _fmt_cell(value: object) -> str:
    """Format a table cell for CSV or Markdown output.

    Parameters
    ----------
    value : object
        A float (formatted to 3 decimals), int (as-is), or the string ``"n/a"``.

    Returns
    -------
    str
        String representation suitable for CSV or Markdown.
    """
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    """Write a UTF-8 BOM CSV file.

    Parameters
    ----------
    path : Path
        Destination file path (parent must exist).
    headers : list[str]
        Column names.
    rows : list[list[object]]
        Data rows.
    """
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow([_fmt_cell(cell) for cell in row])
    logger.info("Wrote CSV  %s (%d rows)", path, len(rows))


def _write_md(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    """Write a pipe-format Markdown table with explicit column alignment.

    Parameters
    ----------
    path : Path
        Destination file path (parent must exist).
    headers : list[str]
        Column names.
    rows : list[list[object]]
        Data rows.

    Notes
    -----
    Alignment rules (per plan spec):
    - ``class_name``: left-aligned (``:---``).
    - ``n_gt`` and ``min_delta_AP``: right-aligned (``---:``).
    - All AP columns: centered (``:---:``).
    """
    lines: list[str] = []

    # Header row
    lines.append("| " + " | ".join(headers) + " |")

    # Alignment row
    alignments: list[str] = []
    for col in headers:
        if col == "class_name":
            alignments.append(":---")
        elif col in ("n_gt", "min_delta_AP"):
            alignments.append("---:")
        else:
            alignments.append(":---:")
    lines.append("| " + " | ".join(alignments) + " |")

    # Data rows
    for row in rows:
        cells = [_fmt_cell(cell) for cell in row]
        lines.append("| " + " | ".join(cells) + " |")

    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote MD   %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    models: Annotated[
        list[str],
        typer.Option("--model", "-m", help="Model(s) to process"),
    ] = list(MODELS_ORDER),  # noqa: B006
    results_root: Annotated[
        Path,
        typer.Option("--results-root", help="Root directory of stage JSON files"),
    ] = Path("results"),
    out_csv: Annotated[
        Path,
        typer.Option("--out-csv", help="Output directory for CSV files"),
    ] = Path("results/per_class"),
    out_md: Annotated[
        Path,
        typer.Option("--out-md", help="Output directory for Markdown files"),
    ] = Path("media/per_class_md"),
) -> None:
    """Generate per-class AP summary tables for each model.

    Emits 4 output files per model (2 sort orders x {CSV, Markdown}):
        <out_csv>/<model>_drop_top10.csv  — top 10 classes by worst AP drop
        <out_csv>/<model>_freq_top10.csv  — top 10 classes by n_gt descending
        <out_md>/<model>_drop_top10.md
        <out_md>/<model>_freq_top10.md

    Requires stage JSONs to have been backfilled with per_class_ap by
    scripts/build_per_class_ap.py.  Exits with code 2 if the FP32 baseline
    for any requested model is missing or not yet backfilled.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    out_csv.mkdir(parents=True, exist_ok=True)
    out_md.mkdir(parents=True, exist_ok=True)

    for model in models:
        if model not in VARIANT_DIRS:
            typer.echo(f"[WARNING] Unknown model '{model}' — skipping", err=True)
            continue

        logger.info("Processing model: %s", model)
        headers, rows = _build_table(model, results_root)

        drop_rows = _sort_top10_drop(rows, headers)
        freq_rows = _sort_top10_freq(rows, headers)

        _write_csv(out_csv / f"{model}_drop_top10.csv", headers, drop_rows)
        _write_csv(out_csv / f"{model}_freq_top10.csv", headers, freq_rows)
        _write_md(out_md / f"{model}_drop_top10.md", headers, drop_rows)
        _write_md(out_md / f"{model}_freq_top10.md", headers, freq_rows)

    typer.echo("Done.")


if __name__ == "__main__":
    app()
