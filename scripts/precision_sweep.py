"""Render the precision-sweep chart: mAP and latency vs decreasing precision.

For each of the four models plots two lines against a categorical X axis
representing the canonical precision sequence
``FP32 -> TF32 -> FP16 -> BF16 -> INT8 -> Mixed``:

* A solid line on the **left** Y axis tracks ``map_50_95`` as precision is
  reduced.
* A dashed line on the **right** Y axis (twin) tracks ``latency_total_ms``.

Both lines share the model colour. For multi-variant precision categories
(INT8 with three calibrations, Mixed with two strategies) the variant with
the highest mAP for that model is selected as the representative.
RF-DETR-L's INT8 and Mixed stages are excluded (TRT auto-tuner rolled back
to FP16) so its lines naturally terminate at BF16.

The "optimum" precision per model -- the last step where the relative
latency improvement still exceeds the relative mAP loss -- is annotated on
the chart so the trade-off question ("where to stop quantising") can be
read off directly. The annotation block is clearly demarcated so it can be
removed independently.
"""

from __future__ import annotations

import csv
import logging
import math
import sys
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import typer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants (shared with scripts/pareto_curve.py)
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

# Categorical X axis. Each label maps to one stage id or a tuple of stages
# whose best-mAP representative is chosen as the data point.
PRECISION_CATEGORIES: list[str] = ["FP32", "TF32", "FP16", "BF16", "INT8", "Mixed"]

PRECISION_STAGE_MAP: dict[str, str | tuple[str, ...]] = {
    "FP32": "1_pytorch_fp32",
    "TF32": "3_trt_tf32",
    "FP16": "4_trt_fp16",
    "BF16": "4_trt_bf16",
    "INT8": ("5_trt_int8_entropy", "5_trt_int8_minmax", "5_trt_int8_percentile"),
    "Mixed": ("6_trt_mixed_a", "6_trt_mixed_b"),
}


app = typer.Typer(
    name="precision-sweep",
    help="Build precision-sweep twin-axis chart from results/results.csv.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_lookup(
    csv_path: Path,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Return ``{model: {stage: (latency_ms, mAP)}}`` from the benchmark CSV.

    Skips rows with a non-empty ``skipped_reason`` and the 5 RF-DETR-L INT8 /
    Mixed stages that rolled back to FP16.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    lookup: dict[str, dict[str, tuple[float, float]]] = {}
    for row in rows:
        if row.get("skipped_reason", "").strip():
            continue
        if not row.get("latency_total_ms", "").strip():
            continue
        if not row.get("map_50_95", "").strip():
            continue
        model = row["model_name"]
        stage = row["stage"]
        if model == "rfdetr-l" and stage in RFDETR_L_INVALID_STAGES:
            continue
        lookup.setdefault(model, {})[stage] = (
            float(row["latency_total_ms"]),
            float(row["map_50_95"]),
        )
    return lookup


def _model_sequence(
    lookup: dict[str, dict[str, tuple[float, float]]],
    model: str,
) -> list[tuple[float, float, str | None]]:
    """Return one ``(latency, mAP, source_stage)`` per precision category.

    Length always equals ``len(PRECISION_CATEGORIES)``. Missing data is
    returned as ``(nan, nan, None)`` so the line breaks naturally in the
    plot.
    """
    model_data = lookup.get(model, {})
    sequence: list[tuple[float, float, str | None]] = []
    for cat in PRECISION_CATEGORIES:
        spec = PRECISION_STAGE_MAP[cat]
        candidates = (spec,) if isinstance(spec, str) else spec
        available = [(s, model_data[s]) for s in candidates if s in model_data]
        if not available:
            sequence.append((math.nan, math.nan, None))
            continue
        best_stage, (lat, mp) = max(available, key=lambda item: item[1][1])
        sequence.append((lat, mp, best_stage))
    return sequence


def _optimum_index(sequence: list[tuple[float, float, str | None]]) -> int | None:
    """Return the X index of the optimum precision for a model.

    Heuristic: walk left-to-right (FP32 -> Mixed). At step ``i -> i+1``
    compute the relative latency gain (``lat[i] - lat[i+1]) / lat[i]``) and
    the relative mAP loss (``(map[i] - map[i+1]) / map[i]``). The optimum
    is the last step where the gain strictly exceeds the loss -- continue
    quantising while it pays off, stop once mAP degrades faster than
    latency improves. Returns ``None`` if no valid baseline exists.
    """
    baseline_idx: int | None = None
    for i, (lat, mp, _stage) in enumerate(sequence):
        if not (math.isnan(lat) or math.isnan(mp)):
            baseline_idx = i
            break
    if baseline_idx is None:
        return None

    optimum = baseline_idx
    for i in range(baseline_idx, len(sequence) - 1):
        lat_a, map_a, _ = sequence[i]
        lat_b, map_b, _ = sequence[i + 1]
        if math.isnan(lat_b) or math.isnan(map_b):
            break
        lat_gain = (lat_a - lat_b) / lat_a if lat_a > 0 else 0.0
        map_loss = (map_a - map_b) / map_a if map_a > 0 else 0.0
        if lat_gain > map_loss:
            optimum = i + 1
        else:
            break
    return optimum


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@app.command()
def main(  # noqa: PLR0915
    results_csv: Annotated[
        Path,
        typer.Option("--results", help="Path to results.csv"),
    ] = Path("results/results.csv"),
    out_path: Annotated[
        Path,
        typer.Option("--out", help="Output PNG path"),
    ] = Path("media/pareto/precision_sweep.png"),
    annotate_optimum: Annotated[
        bool,
        typer.Option(
            "--annotate-optimum/--no-annotate-optimum",
            help="Annotate per-model optimum precision (where Δlatency > ΔmAP).",
        ),
    ] = True,
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="Output PNG DPI"),
    ] = 150,
) -> None:
    """Render the precision-sweep twin-axis chart."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if not results_csv.exists():
        typer.echo(f"ERROR: results CSV not found at {results_csv}")
        raise typer.Exit(code=1)

    lookup = _load_lookup(results_csv)
    if not lookup:
        typer.echo("ERROR: no valid rows in results CSV")
        raise typer.Exit(code=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax_map = plt.subplots(figsize=(13.0, 8.0))
    ax_lat = ax_map.twinx()

    x_positions = list(range(len(PRECISION_CATEGORIES)))

    model_order = ("rt-detr", "rfdetr-l", "yolo11l", "yolo26l")
    sequences: dict[str, list[tuple[float, float, str | None]]] = {}

    for model in model_order:
        seq = _model_sequence(lookup, model)
        sequences[model] = seq

        map_vals = [pt[1] for pt in seq]
        lat_vals = [pt[0] for pt in seq]

        color = MODEL_COLOR[model]

        ax_map.plot(
            x_positions,
            map_vals,
            color=color,
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.6,
            alpha=0.95,
            zorder=3,
        )
        ax_lat.plot(
            x_positions,
            lat_vals,
            color=color,
            linewidth=2.0,
            linestyle="--",
            marker="s",
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.6,
            alpha=0.75,
            zorder=2.5,
        )

    # -----------------------------------------------------------------
    # OPTIMUM ANNOTATION BLOCK -- delete to disable per-model markers.
    # -----------------------------------------------------------------
    if annotate_optimum:
        optima: list[tuple[str, int, float, float]] = []
        for model in model_order:
            seq = sequences[model]
            idx = _optimum_index(seq)
            if idx is None:
                continue
            lat_opt, map_opt, _ = seq[idx]
            optima.append((model, idx, lat_opt, map_opt))
            ax_map.scatter(
                idx,
                map_opt,
                s=320,
                facecolors="none",
                edgecolors=MODEL_COLOR[model],
                linewidths=2.4,
                zorder=4,
            )

        if optima:
            lines = [
                f"{MODEL_DISPLAY[m]}: оптимум на {PRECISION_CATEGORIES[i]} "
                f"(mAP={mp:.3f}, {lat:.2f} мс)"
                for m, i, lat, mp in optima
            ]
            ax_map.text(
                0.015,
                0.015,
                "Оптимум по правилу ΔLat > ΔmAP:\n" + "\n".join(lines),
                transform=ax_map.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                bbox={
                    "boxstyle": "round,pad=0.5",
                    "facecolor": "#fff8dc",
                    "edgecolor": "#444444",
                    "linewidth": 0.8,
                },
                zorder=5,
            )
    # ---------------------------- END OPTIMUM ANNOTATION BLOCK -------------

    ax_map.set_xticks(x_positions)
    ax_map.set_xticklabels(PRECISION_CATEGORIES)
    ax_map.set_xlabel("Точность вычислений (от FP32 к смешанной — в порядке уменьшения битности)")
    ax_map.set_ylabel("mAP@[0.5:0.95] на COCO val2017  (сплошная линия)")
    ax_lat.set_ylabel("Латентность, мс (Pre + Inference + Post, RTX 3070)  (пунктирная линия)")

    ax_map.grid(visible=True, which="both", axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax_map.grid(visible=True, which="both", axis="x", linestyle=":", linewidth=0.5, alpha=0.4)

    ax_map.set_title(
        "Зависимость mAP и латентности от формата вычислений\n"
        "Оптимум — последний шаг, на котором ΔЛатентность > ΔmAP"
    )

    # Two-part legend: model colour + line-style meaning.
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            color=MODEL_COLOR[m],
            linewidth=2.5,
            marker="o",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=MODEL_DISPLAY[m],
        )
        for m in model_order
    ]
    style_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="mAP  (левая ось)",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=2.0,
            linestyle="--",
            marker="s",
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="Латентность  (правая ось)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#444444",
            markeredgewidth=2.0,
            markersize=14,
            label="Оптимум модели",
        ),
    ]

    legend_models = ax_map.legend(
        handles=model_handles,
        title="Модель",
        loc="upper right",
        framealpha=0.92,
    )
    ax_map.add_artist(legend_models)
    ax_map.legend(
        handles=style_handles,
        title="Метрика",
        loc="center right",
        framealpha=0.92,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Per-model precision sweep:")
    for model in model_order:
        seq = sequences[model]
        idx = _optimum_index(seq)
        opt_label = PRECISION_CATEGORIES[idx] if idx is not None else "n/a"
        logger.info(
            "  %-10s  optimum=%-6s  points=%s",
            MODEL_DISPLAY[model],
            opt_label,
            ", ".join(
                f"{cat}={'-' if math.isnan(pt[1]) else f'{pt[1]:.3f}'}"
                for cat, pt in zip(PRECISION_CATEGORIES, seq, strict=True)
            ),
        )

    typer.echo(f"Saved: {out_path}")


if __name__ == "__main__":
    app()
