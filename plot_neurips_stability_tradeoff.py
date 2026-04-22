#!/usr/bin/env python3
"""Create a single-panel NeurIPS-style stability-efficiency teaser plot.

Example:
    python plot_neurips_stability_tradeoff.py \
        --comparable SLDM:50:88.09 AccGeoLDM:16:51.02 MOLTD:12:92.53 \
        --baseline GeoLDM-4:4:12.43 GeoLDM-5:5:27.96 GeoLDM-8:8:70.78 \
        --ours FlashMol-4:4:83.89 FlashMol-5:5:87.05 FlashMol-8:8:94.87 \
        --output neurips_stability_tradeoff_flashmol.pdf
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COMPARABLE = (
    "SLDM:50:88.09",
    "AccGeoLDM:16:51.02",
    "MOLTD:12:92.53",
)


@dataclass(frozen=True)
class Point:
    label: str
    nfe: float
    stability: float


def parse_point(raw_value: str) -> Point:
    parts = raw_value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid point '{raw_value}'. Expected format 'label:nfe:stability'."
        )

    label, nfe, stability = parts
    try:
        return Point(label=label, nfe=float(nfe), stability=float(stability))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid numeric values in '{raw_value}'."
        ) from exc


def parse_group(raw_values: Sequence[str]) -> List[Point]:
    return [parse_point(value) for value in raw_values]


def sort_points(points: Sequence[Point]) -> List[Point]:
    return sorted(points, key=lambda point: point.nfe)


def padded_limits(values: Iterable[float], pad_ratio: float) -> Tuple[float, float]:
    values = list(values)
    lower = min(values)
    upper = max(values)
    if np.isclose(lower, upper):
        delta = max(abs(lower) * 0.1, 1.0)
        return lower - delta, upper + delta

    padding = (upper - lower) * pad_ratio
    return lower - padding, upper + padding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a single-panel NFEs vs molecular stability teaser figure."
    )
    parser.add_argument(
        "--comparable",
        nargs="+",
        default=list(DEFAULT_COMPARABLE),
        help="Comparable models as repeated 'label:nfe:stability' values.",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        required=True,
        help="Baseline model points as repeated 'label:nfe:stability' values.",
    )
    parser.add_argument(
        "--ours",
        nargs="+",
        required=True,
        help="Our model points as repeated 'label:nfe:stability' values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurips_stability_tradeoff_flashmol.pdf"),
        help="Output path. Use .pdf for paper-ready vector output.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster DPI when saving bitmap output.",
    )
    return parser


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, which="major", color="#cfd3db", linewidth=0.75, alpha=0.55)
    ax.grid(True, which="minor", axis="y", color="#d9dde4", linewidth=0.55, alpha=0.4)
    ax.set_axisbelow(True)

    for spine_name in ("top", "right", "bottom", "left"):
        ax.spines[spine_name].set_linewidth(0.95)
        ax.spines[spine_name].set_color("#bcc1c9")

    ax.tick_params(axis="both", labelsize=11, colors="#2f3744", width=0.8)
    ax.xaxis.label.set_color("#1d2530")
    ax.yaxis.label.set_color("#1d2530")


def draw_flashmol(ax: plt.Axes, points: Sequence[Point], color: str) -> None:
    xs = np.array([point.nfe for point in points], dtype=float)
    ys = np.array([point.stability for point in points], dtype=float)

    ax.plot(xs, ys, color=color, linewidth=3.0, alpha=0.96, zorder=5)
    ax.scatter(xs, ys, s=760, color=color, alpha=0.12, linewidth=0, zorder=5)
    ax.scatter(
        xs,
        ys,
        s=250,
        color=color,
        edgecolor="white",
        linewidth=1.8,
        alpha=0.98,
        zorder=6,
    )


def draw_baseline_family(ax: plt.Axes, points: Sequence[Point], color: str) -> None:
    xs = np.array([point.nfe for point in points], dtype=float)
    ys = np.array([point.stability for point in points], dtype=float)

    ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.72, zorder=3)
    ax.scatter(
        xs,
        ys,
        s=145,
        color=color,
        edgecolor="white",
        linewidth=1.1,
        alpha=0.9,
        zorder=4,
    )


def draw_comparables(ax: plt.Axes, points: Sequence[Point], color: str) -> None:
    xs = np.array([point.nfe for point in points], dtype=float)
    ys = np.array([point.stability for point in points], dtype=float)
    ax.scatter(
        xs,
        ys,
        s=150,
        color=color,
        edgecolor="white",
        linewidth=1.1,
        alpha=0.86,
        zorder=4,
    )

def label_offsets() -> dict[str, Tuple[float, float, str, str]]:
    return {
        "FlashMol-4": (8, -2, "left", "bold"),
        "FlashMol-5": (8, 0, "left", "bold"),
        "FlashMol-8": (10, 8, "left", "bold"),
        "GeoLDM-4": (0, -16, "center", "normal"),
        "GeoLDM-5": (0, -16, "center", "normal"),
        "GeoLDM-8": (8, 4, "left", "normal"),
        "AccGeoLDM": (8, -2, "left", "normal"),
        "MOLTD": (0, 10, "center", "normal"),
        "SLDM": (-8, 0, "right", "normal"),
    }


def annotate_points(ax: plt.Axes, points: Sequence[Point], color: str, emphasis: bool) -> None:
    offsets = label_offsets()
    for point in points:
        dx, dy, ha, weight = offsets.get(point.label, (6, 6, "left", "normal"))
        ax.annotate(
            point.label,
            (point.nfe, point.stability),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=11 if emphasis else 9.6,
            fontweight=weight,
            color=color,
            alpha=0.98 if emphasis else 0.9,
            zorder=8,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    comparable_points = sort_points(parse_group(args.comparable))
    baseline_points = sort_points(parse_group(args.baseline))
    our_points = sort_points(parse_group(args.ours))
    all_points = comparable_points + baseline_points + our_points

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")

    colors = {
        "flashmol": "#8b2f8f",
        "baseline": "#7a8cab",
        "comparable": "#7a8d84",
        "flashmol_text": "#6f1f77",
        "baseline_text": "#5f6979",
        "comparable_text": "#616c66",
    }

    style_axes(ax)
    ax.set_xscale("log")
    draw_baseline_family(ax, baseline_points, colors["baseline"])
    draw_comparables(ax, comparable_points, colors["comparable"])
    draw_flashmol(ax, our_points, colors["flashmol"])

    annotate_points(ax, baseline_points, colors["baseline_text"], emphasis=False)
    annotate_points(ax, comparable_points, colors["comparable_text"], emphasis=False)
    annotate_points(ax, our_points, colors["flashmol_text"], emphasis=True)

    all_ys = [point.stability for point in all_points]
    y_min, y_max = padded_limits(all_ys, pad_ratio=0.09)
    ax.set_ylim(max(0.0, y_min), min(100.0, max(100.0, y_max)))
    ax.set_xlim(3.7, 55.0)

    ticks = [4, 8, 12, 16, 32, 50]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_ylabel("Molecular Stability")
    ax.set_xlabel("NFEs (log scale)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {args.output.resolve()}")


if __name__ == "__main__":
    main()
