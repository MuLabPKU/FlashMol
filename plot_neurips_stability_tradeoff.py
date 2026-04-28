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
from matplotlib.lines import Line2D


DEFAULT_COMPARABLE = (
    "GeoLDM*-4:4:0.0",
    "GeoLDM*-5:5:0.08",
    "GeoLDM*-8:8:3.67",
    "SLDM:50:88.09",
    "AccGeoLDM-16:16:51.02",
    "AccGeoLDM-32:32:77.02",
    "AccGeoLDM-63:63:84.24",
    "AccGeoLDM-125:125:88.50",
    "AccGeoLDM-250:250:89.74",
    "AccGeoLDM-500:500:88.93",
    "GeoBFN:100:87.2",
    "GeoLDM:1000:89.4",
    "EquiFM:200:88.3",
    "GeoRCG (EDM):50:89.08",
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


def split_label(label: str) -> Tuple[str, str | None]:
    if "-" not in label:
        return label, None

    prefix, suffix = label.rsplit("-", 1)
    if suffix.replace(".", "", 1).isdigit():
        return prefix, suffix
    return label, None


def group_points_by_method(points: Sequence[Point]) -> dict[str, List[Point]]:
    grouped: dict[str, List[Point]] = {}
    for point in points:
        method_name, _ = split_label(point.label)
        grouped.setdefault(method_name, []).append(point)

    return {method: sort_points(method_points) for method, method_points in grouped.items()}


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
    ax.grid(True, which="major", color="#cfd3db", linewidth=1.1, alpha=0.55)
    ax.grid(True, which="minor", axis="y", color="#d9dde4", linewidth=0.8, alpha=0.4)
    ax.set_axisbelow(True)

    for spine_name in ("top", "right", "bottom", "left"):
        ax.spines[spine_name].set_linewidth(1.1)
        ax.spines[spine_name].set_color("#bcc1c9")
    ax.spines["bottom"].set_zorder(0)
    ax.spines["left"].set_zorder(0)

    ax.tick_params(axis="both", labelsize=22, colors="#2f3744", width=1.0)
    ax.xaxis.label.set_color("#1d2530")
    ax.yaxis.label.set_color("#1d2530")


def draw_flashmol(ax: plt.Axes, points: Sequence[Point], color: str) -> None:
    xs = np.array([point.nfe for point in points], dtype=float)
    ys = np.array([point.stability for point in points], dtype=float)

    ax.plot(xs, ys, color=color, linewidth=3.4, alpha=0.98, zorder=6)
    ax.scatter(xs, ys, s=900, color=color, alpha=0.14, linewidth=0, zorder=5)
    ax.scatter(
        xs,
        ys,
        s=280,
        color=color,
        edgecolor="white",
        linewidth=2.0,
        alpha=0.99,
        zorder=7,
    )


def draw_baseline_family(ax: plt.Axes, points: Sequence[Point], color: str) -> None:
    xs = np.array([point.nfe for point in points], dtype=float)
    ys = np.array([point.stability for point in points], dtype=float)

    ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.72, zorder=3)
    ax.scatter(
        xs,
        ys,
        s=135,
        color=color,
        edgecolor="white",
        linewidth=1.1,
        alpha=0.82,
        zorder=4,
    )


def draw_method_series(
    ax: plt.Axes,
    points: Sequence[Point],
    color: str,
    *,
    connect_points: bool,
    line_width: float = 2.2,
    marker_size: float = 190,
    marker: str = "o",
    zorder: int = 4,
) -> None:
    xs = np.array([point.nfe for point in points], dtype=float)
    ys = np.array([point.stability for point in points], dtype=float)
    if connect_points and len(points) > 1:
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=line_width,
            alpha=0.78,
            zorder=zorder - 1,
        )
    ax.scatter(
        xs,
        ys,
        s=marker_size,
        color=color,
        marker=marker,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.82,
        zorder=zorder,
    )


def label_offsets() -> dict[str, Tuple[float, float, str, str]]:
    return {
        "FlashMol-4": (16, -18, "left", "bold"),
        "FlashMol-5": (16, -3, "left", "bold"),
        "FlashMol-8": (16, 15, "left", "bold"),
        "GeoLDM-4": (-14, -18, "right", "normal"),
        "GeoLDM-5": (14, -18, "left", "normal"),
        "GeoLDM-8": (14, 10, "left", "normal"),
        "GeoLDM": (-14, 11, "right", "normal"),
        "GeoLDM*-4": (-18, 16, "right", "normal"),
        "GeoLDM*-5": (14, -38, "left", "normal"),
        "GeoLDM*-8": (10, 10, "left", "normal"),
        "AccGeoLDM-16": (14, -10, "left", "normal"),
        "AccGeoLDM-32": (14, 10, "left", "normal"),
        "AccGeoLDM-125": (12, -12, "left", "normal"),
        "AccGeoLDM-500": (12, -16, "left", "normal"),
    }


def should_label_point(point: Point) -> bool:
    return point.label in {
        "FlashMol-4",
        "FlashMol-5",
        "FlashMol-8",
        "GeoLDM-4",
        "GeoLDM-5",
        "GeoLDM-8",
        "GeoLDM",
        "GeoLDM*-4",
        "GeoLDM*-5",
        "GeoLDM*-8",
        "AccGeoLDM-16",
        "AccGeoLDM-32",
        "AccGeoLDM-125",
        "AccGeoLDM-500",
    }


def annotate_points(ax: plt.Axes, points: Sequence[Point], color: str, emphasis: bool) -> None:
    offsets = label_offsets()
    for point in points:
        if not should_label_point(point):
            continue
        dx, dy, ha, weight = offsets.get(point.label, (6, 6, "left", "normal"))
        zorder = 8
        display_label = "GeoLDM*" if point.label == "GeoLDM" else point.label
        ax.annotate(
            display_label,
            (point.nfe, point.stability),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=22 if emphasis else 19,
            fontweight=weight,
            color=color,
            alpha=0.98 if emphasis else 0.9,
            arrowprops=(
                {
                    "arrowstyle": "-",
                    "color": color,
                    "lw": 0.8,
                    "alpha": 0.5,
                    "shrinkA": 0,
                    "shrinkB": 3,
                }
                if point.label in {"GeoLDM*-4", "GeoLDM*-5"}
                else None
            ),
            zorder=zorder,
        )


def add_legend(ax: plt.Axes, colors: dict[str, str]) -> None:
    legend_handles = [
        Line2D([], [], linestyle="none", label="Methods (NFE sweeps)"),
        Line2D([0], [0], color=colors["FlashMol"], lw=4.0, marker="o", markersize=10, label="FlashMol"),
        Line2D([0], [0], color=colors["AccGeoLDM"], lw=2.6, marker="o", markersize=8, label="AccGeoLDM"),
        Line2D([0], [0], color=colors["GeoLDM"], lw=2.0, marker="o", markersize=8, label="GeoLDM"),
        Line2D([0], [0], color=colors["GeoLDM*"], lw=2.0, marker="o", markersize=8, label="GeoLDM*"),
        Line2D([], [], linestyle="none", label=""),
        Line2D([], [], linestyle="none", label="Baselines"),
        Line2D([0], [0], color=colors["SLDM"], lw=0, marker="o", markersize=9, markerfacecolor=colors["SLDM"], label="SLDM", alpha=0.85),
        Line2D([0], [0], color=colors["GeoRCG (EDM)"], lw=0, marker="o", markersize=9, markerfacecolor=colors["GeoRCG (EDM)"], label="GeoRCG (EDM)", alpha=0.85),
        Line2D([0], [0], color=colors["GeoBFN"], lw=0, marker="o", markersize=9, markerfacecolor=colors["GeoBFN"], label="GeoBFN", alpha=0.85),
        Line2D([0], [0], color=colors["EquiFM"], lw=0, marker="o", markersize=9, markerfacecolor=colors["EquiFM"], label="EquiFM", alpha=0.85),
        Line2D([0], [0], color=colors["MOLTD"], lw=0, marker="o", markersize=9, markerfacecolor=colors["MOLTD"], label="MolTD", alpha=0.85),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#d3d8df",
        fontsize=16,
        handlelength=2.0,
        borderpad=0.55,
        labelspacing=0.45,
    )
    legend.get_texts()[0].set_fontweight("bold")
    legend.get_texts()[6].set_fontweight("bold")
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
            "axes.titlesize": 28,
            "axes.labelsize": 26,
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(13.0, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")

    colors = {
        "FlashMol": "#8b2f8f",
        "GeoLDM": "#2b6cb0",
        "GeoLDM*": "#63b3ed",
        "AccGeoLDM": "#157f6b",
        "GeoBFN": "#b7791f",
        "EquiFM": "#718096",
        "GeoRCG (EDM)": "#805ad5",
        "MOLTD": "#dd6b20",
        "SLDM": "#c05621",
    }

    style_axes(ax)
    ax.set_xscale("log")
    draw_baseline_family(ax, baseline_points, colors["GeoLDM"])
    annotate_points(ax, baseline_points, colors["GeoLDM"], emphasis=False)

    for method_name, method_points in group_points_by_method(comparable_points).items():
        draw_method_series(
            ax,
            method_points,
            colors.get(method_name, "#616c66"),
            connect_points=len(method_points) > 1,
            line_width=2.5 if len(method_points) > 1 else 2.0,
            marker_size=170 if len(method_points) > 1 else 125,
            marker="o",
        )
        annotate_points(ax, method_points, colors.get(method_name, "#616c66"), emphasis=False)

    draw_flashmol(ax, our_points, colors["FlashMol"])
    annotate_points(ax, our_points, colors["FlashMol"], emphasis=True)
    add_legend(ax, colors)

    all_ys = [point.stability for point in all_points]
    y_min, y_max = padded_limits(all_ys, pad_ratio=0.09)
    ax.set_ylim(max(0.0, y_min), min(100.0, max(100.0, y_max)))
    ax.set_xlim(3.7, 1200.0)

    ticks = [4, 8, 16, 32, 63, 125, 250, 500, 1000]
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
