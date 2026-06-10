"""Render a latency vs mAP@[0.5:0.95] Pareto plot from ``results/results.csv``.

For each (model, stage) row in ``results/results.csv`` plots a scatter point
(X = ``latency_total_ms``, Y = ``map_50_95``), coloured by model and shaped
by stage family. Adds:

* A short stage label next to every point so the reader can identify the
  configuration without consulting the legend.
* A per-model quantisation trajectory line connecting the canonical
  precision sequence (FP32 -> TF32 -> FP16 -> BF16 -> INT8(best) ->
  Mixed(best)). For INT8 and Mixed the representative is the calibration
  or strategy with the highest mAP for that model.
* A step-style Pareto frontier across all 35 valid points.

Excludes the 5 RF-DETR-L INT8 and Mixed stages whose TRT auto-tuner rolled
back to FP16 (consistent with ``scripts/build_confusion.py``).

The "knee" annotation is enclosed in a clearly marked block so it can be
removed without touching the rest of the rendering code.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import typer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

RFDETR_L_INVALID_STAGES: frozenset[str] = frozenset(
    {
        "5_trt_int8_entropy",
        "5_trt_int8_minmax",
        "5_trt_int8_percentile",
        "6_trt_mixed_a",
        "6_trt_mixed_b",
    }
)

MODEL_DISPLAY: dict[str, str] = {
    "rt-detr": "RT-DETR",
    "yolo11l": "YOLO11L",
    "yolo26l": "YOLO26L",
    "rfdetr-l": "RF-DETR-L",
}

MODEL_COLOR: dict[str, str] = {
    "rt-detr": "#1f77b4",
    "rfdetr-l": "#d62728",
    "yolo11l": "#2ca02c",
    "yolo26l": "#ff7f0e",
}

STAGE_DISPLAY: dict[str, str] = {
    "1_pytorch_fp32": "PyTorch FP32",
    "2_onnx_fp32": "ONNX FP32",
    "3_trt_tf32": "TRT TF32",
    "4_trt_fp16": "TRT FP16",
    "4_trt_bf16": "TRT BF16",
    "5_trt_int8_entropy": "TRT INT8 Entropy",
    "5_trt_int8_minmax": "TRT INT8 MinMax",
    "5_trt_int8_percentile": "TRT INT8 Percentile",
    "6_trt_mixed_a": "TRT Смешанная точность A",
    "6_trt_mixed_b": "TRT Смешанная точность B",
}

# Marker per stage family. Half-precision (FP16/BF16) and INT8 calibrations
# share a family marker; colour conveys the model.
STAGE_MARKER: dict[str, str] = {
    "1_pytorch_fp32": "o",
    "2_onnx_fp32": "s",
    "3_trt_tf32": "^",
    "4_trt_fp16": "D",
    "4_trt_bf16": "D",
    "5_trt_int8_entropy": "P",
    "5_trt_int8_minmax": "P",
    "5_trt_int8_percentile": "P",
    "6_trt_mixed_a": "*",
    "6_trt_mixed_b": "*",
}

MIN_FRONT_FOR_STEP_LINE: int = 2

# Short labels placed next to each scatter point.
STAGE_SHORT: dict[str, str] = {
    "1_pytorch_fp32": "FP32",
    "2_onnx_fp32": "ONNX",
    "3_trt_tf32": "TF32",
    "4_trt_fp16": "FP16",
    "4_trt_bf16": "BF16",
    "5_trt_int8_entropy": "INT8-E",
    "5_trt_int8_minmax": "INT8-M",
    "5_trt_int8_percentile": "INT8-P",
    "6_trt_mixed_a": "Mix-A",
    "6_trt_mixed_b": "Mix-B",
}

# Canonical quantisation order followed by the per-model trajectory line.
# Each element is either a single stage id or a tuple of stages whose best
# (max-mAP) representative is used as the trajectory node.
TRAJECTORY_STEPS: list[str | tuple[str, ...]] = [
    "1_pytorch_fp32",
    "3_trt_tf32",
    "4_trt_fp16",
    "4_trt_bf16",
    ("5_trt_int8_entropy", "5_trt_int8_minmax", "5_trt_int8_percentile"),
    ("6_trt_mixed_a", "6_trt_mixed_b"),
]

STAGE_FAMILY_LEGEND: list[tuple[str, str]] = [
    ("PyTorch FP32 (baseline)", "o"),
    ("ONNX FP32", "s"),
    ("TRT TF32", "^"),
    ("TRT FP16 / BF16", "D"),
    ("TRT INT8 (3 калибровки)", "P"),
    ("TRT Смешанная точность", "*"),
]


app = typer.Typer(
    name="pareto-curve",
    help="Build latency-vs-mAP Pareto plot from results/results.csv.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read the benchmark CSV and return non-skipped rows for valid configs."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    valid: list[dict[str, str]] = []
    for row in rows:
        if row.get("skipped_reason", "").strip():
            continue
        if not row.get("latency_total_ms", "").strip():
            continue
        if not row.get("map_50_95", "").strip():
            continue
        if row["model_name"] == "rfdetr-l" and row["stage"] in RFDETR_L_INVALID_STAGES:
            continue
        valid.append(row)
    return valid


def _build_lookup(
    points: list[tuple[float, float, str, str]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Build a ``{model: {stage: (latency, mAP)}}`` lookup from scatter points."""
    lookup: dict[str, dict[str, tuple[float, float]]] = {}
    for lat, mp, model, stage in points:
        lookup.setdefault(model, {})[stage] = (lat, mp)
    return lookup


def _trajectory_for_model(
    lookup: dict[str, dict[str, tuple[float, float]]],
    model: str,
) -> list[tuple[float, float, str]]:
    """Return ordered ``(latency, mAP, stage)`` nodes for a model's quantisation path.

    For multi-variant steps (INT8 calibrations, Mixed strategies) the node
    with the highest mAP for this model is chosen. Missing stages are
    skipped, so RF-DETR-L's trajectory terminates at BF16.
    """
    model_data = lookup.get(model, {})
    trajectory: list[tuple[float, float, str]] = []
    for step in TRAJECTORY_STEPS:
        candidates = (step,) if isinstance(step, str) else step
        available = [(s, model_data[s]) for s in candidates if s in model_data]
        if not available:
            continue
        best_stage, (lat, mp) = max(available, key=lambda item: item[1][1])
        trajectory.append((lat, mp, best_stage))
    return trajectory


def _pareto_front(
    points: list[tuple[float, float, str, str]],
    map_tolerance: float = 1e-3,
) -> list[tuple[float, float, str, str]]:
    """Return non-dominated points (minimise X = latency, maximise Y = mAP).

    A point is added to the front only if its mAP exceeds the previously seen
    best by at least ``map_tolerance``. Differences below this threshold lie
    within COCOeval numerical noise and would otherwise produce
    visually-distracting "ties" on the frontier.

    Parameters
    ----------
    points : list of (latency, map, model, stage)
    map_tolerance : minimum mAP improvement required to extend the front.

    Returns
    -------
    list of Pareto-optimal points sorted by latency ascending. On the
    Pareto frontier, mAP is strictly increasing with latency.
    """
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front: list[tuple[float, float, str, str]] = []
    best_map = -float("inf")
    for pt in sorted_pts:
        _lat, mp, _model, _stage = pt
        if mp > best_map + map_tolerance:
            front.append(pt)
            best_map = mp
    return front


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@app.command()
def main(  # noqa: PLR0912, PLR0915
    results_csv: Annotated[
        Path,
        typer.Option("--results", help="Path to results.csv"),
    ] = Path("results/results.csv"),
    out_path: Annotated[
        Path,
        typer.Option("--out", help="Output PNG path"),
    ] = Path("media/pareto/pareto_latency_map.png"),
    annotate_knee: Annotated[
        bool,
        typer.Option(
            "--annotate-knee/--no-annotate-knee",
            help="Annotate the recommended trade-off point.",
        ),
    ] = True,
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="Output PNG DPI"),
    ] = 150,
    map_tolerance: Annotated[
        float,
        typer.Option(
            "--map-tolerance",
            help=(
                "Minimum mAP improvement (absolute) for a point to join the "
                "Pareto front. Filters out COCOeval numerical noise."
            ),
        ),
    ] = 1e-3,
) -> None:
    """Render the latency vs mAP@[0.5:0.95] Pareto plot."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if not results_csv.exists():
        typer.echo(f"ERROR: results CSV not found at {results_csv}")
        raise typer.Exit(code=1)

    rows = _load_rows(results_csv)
    logger.info("Loaded %d valid rows from %s", len(rows), results_csv)

    if not rows:
        typer.echo("ERROR: no valid rows to plot")
        raise typer.Exit(code=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    points: list[tuple[float, float, str, str]] = [
        (
            float(r["latency_total_ms"]),
            float(r["map_50_95"]),
            r["model_name"],
            r["stage"],
        )
        for r in rows
    ]
    front = _pareto_front(points, map_tolerance=map_tolerance)

    fig, ax = plt.subplots(figsize=(13.0, 8.0))

    # Per-model quantisation trajectory: FP32 -> TF32 -> FP16 -> BF16 ->
    # INT8(best) -> Mixed(best). Drawn before scatter so dots sit on top.
    lookup = _build_lookup(points)
    for model in MODEL_DISPLAY:
        trajectory = _trajectory_for_model(lookup, model)
        if len(trajectory) < MIN_FRONT_FOR_STEP_LINE:
            continue
        traj_x = [t[0] for t in trajectory]
        traj_y = [t[1] for t in trajectory]
        ax.plot(
            traj_x,
            traj_y,
            color=MODEL_COLOR[model],
            linewidth=1.6,
            alpha=0.55,
            linestyle="-",
            zorder=1.8,
        )

    # Scatter — colour by model, marker by stage family.
    for lat, mp, model, stage in points:
        ax.scatter(
            lat,
            mp,
            color=MODEL_COLOR.get(model, "#888888"),
            marker=STAGE_MARKER.get(stage, "x"),
            s=110,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
            zorder=3,
        )

    # Short stage label next to each point — readable without consulting the
    # legend. Offset is small and consistent; mild rotation reduces overlap
    # within tight latency clusters.
    for lat, mp, _model, stage in points:
        ax.annotate(
            STAGE_SHORT.get(stage, stage),
            xy=(lat, mp),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#222222",
            alpha=0.85,
            zorder=3.5,
        )

    # Pareto frontier — step line up-and-right (each step: horizontal hold
    # to next Pareto latency, then vertical jump to higher mAP). The step
    # plot makes the dominance relation explicit.
    if len(front) >= MIN_FRONT_FOR_STEP_LINE:
        step_x: list[float] = []
        step_y: list[float] = []
        for i, (lat, mp, _m, _s) in enumerate(front):
            if i == 0:
                step_x.append(lat)
                step_y.append(mp)
            else:
                step_x.append(lat)
                step_y.append(front[i - 1][1])
                step_x.append(lat)
                step_y.append(mp)
        ax.plot(
            step_x,
            step_y,
            color="#444444",
            linewidth=2.0,
            linestyle="--",
            zorder=2,
            label="Парето-фронт (мин. латентность, макс. mAP)",
        )
        # Emphasise Pareto points with an outer ring.
        for lat, mp, _m, _s in front:
            ax.scatter(
                lat,
                mp,
                s=260,
                facecolors="none",
                edgecolors="#444444",
                linewidths=1.8,
                zorder=4,
            )

    # -----------------------------------------------------------------
    # KNEE ANNOTATION BLOCK -- delete this block to disable annotations.
    # -----------------------------------------------------------------
    if annotate_knee and front:
        # Knee = Pareto point with the highest mAP (rightmost on the front).
        knee = max(front, key=lambda p: p[1])
        knee_lat, knee_map, knee_model, knee_stage = knee
        label = (
            f"Оптимум: {MODEL_DISPLAY.get(knee_model, knee_model)}\n"
            f"{STAGE_DISPLAY.get(knee_stage, knee_stage)}\n"
            f"{knee_lat:.2f} мс  |  mAP={knee_map:.3f}"
        )
        ax.annotate(
            label,
            xy=(knee_lat, knee_map),
            xytext=(knee_lat + 4.0, knee_map - 0.06),
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#fff8dc",
                "edgecolor": "#444444",
                "linewidth": 0.8,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": "#444444",
                "lw": 1.0,
            },
            zorder=5,
        )

        # Also annotate the minimum-latency Pareto point as the speed-first
        # alternative -- helps the reader read the trade-off explicitly.
        fast = min(front, key=lambda p: p[0])
        if fast != knee:
            fast_lat, fast_map, fast_model, fast_stage = fast
            fast_label = (
                f"Мин. латентность: {MODEL_DISPLAY.get(fast_model, fast_model)}\n"
                f"{STAGE_DISPLAY.get(fast_stage, fast_stage)}\n"
                f"{fast_lat:.2f} мс  |  mAP={fast_map:.3f}"
            )
            ax.annotate(
                fast_label,
                xy=(fast_lat, fast_map),
                xytext=(fast_lat + 2.0, fast_map - 0.10),
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "#e8f4ff",
                    "edgecolor": "#444444",
                    "linewidth": 0.8,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#444444",
                    "lw": 1.0,
                },
                zorder=5,
            )
    # ---------------------------- END KNEE ANNOTATION BLOCK ----------------

    # Two-part legend: models (colour) + stage families (marker shape).
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MODEL_COLOR[m],
            markeredgecolor="black",
            markersize=10,
            label=MODEL_DISPLAY[m],
        )
        for m in ("rt-detr", "rfdetr-l", "yolo11l", "yolo26l")
    ]
    stage_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            markerfacecolor="#bbbbbb",
            markeredgecolor="black",
            markersize=10,
            label=name,
        )
        for name, marker in STAGE_FAMILY_LEGEND
    ]

    trajectory_handle = plt.Line2D(
        [0],
        [0],
        color="#666666",
        linewidth=1.6,
        alpha=0.7,
        label="Траектория квантования (FP32→TF32→FP16→BF16→INT8→Mixed)",
    )
    pareto_handle = plt.Line2D(
        [0],
        [0],
        color="#444444",
        linewidth=2.0,
        linestyle="--",
        label="Парето-фронт",
    )

    legend_models = ax.legend(
        handles=model_handles,
        title="Модель (цвет точки и траектории)",
        loc="lower right",
        framealpha=0.92,
    )
    ax.add_artist(legend_models)
    legend_stages = ax.legend(
        handles=stage_handles,
        title="Этап оптимизации (форма маркера)",
        loc="lower left",
        framealpha=0.92,
    )
    ax.add_artist(legend_stages)
    ax.legend(
        handles=[trajectory_handle, pareto_handle],
        title="Линии",
        loc="upper right",
        framealpha=0.92,
    )

    ax.set_xlabel("Латентность, мс (Pre + Inference + Post, RTX 3070, batch=1)")
    ax.set_ylabel("mAP@[0.5:0.95] на COCO val2017")
    ax.set_title(
        "Парето-кривая: компромисс между латентностью и точностью\n"
        "при аппаратной оптимизации детекторов на трансформерах"
    )
    ax.grid(visible=True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Pareto front (%d points):", len(front))
    for lat, mp, m, s in front:
        logger.info(
            "  %-10s %-25s  lat=%.2fms  mAP=%.4f",
            MODEL_DISPLAY.get(m, m),
            STAGE_DISPLAY.get(s, s),
            lat,
            mp,
        )

    typer.echo(f"Saved: {out_path}  ({len(points)} points, {len(front)} on Pareto front)")


if __name__ == "__main__":
    app()
