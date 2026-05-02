#!/usr/bin/env python3
"""Plot Valid&Unique vs Molecular Stability trade-offs with epoch-colored points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Record:
    step_num: int
    epoch: int
    stability: float
    valid_unique: float


RECORDS = (
    Record(step_num=5, epoch=5, stability=87.05, valid_unique=78.83),
    Record(step_num=5, epoch=10, stability=91.83, valid_unique=76.99),
    Record(step_num=5, epoch=15, stability=93.62, valid_unique=74.95),
    Record(step_num=8, epoch=5, stability=94.87, valid_unique=87.51),
    Record(step_num=8, epoch=10, stability=96.62, valid_unique=86.51),
    Record(step_num=8, epoch=15, stability=96.87, valid_unique=85.40),
)


def padded_limits(values: Iterable[float], pad_ratio: float) -> Tuple[float, float]:
    values = list(values)
    lower = min(values)
    upper = max(values)
    if np.isclose(lower, upper):
        delta = max(abs(lower) * 0.1, 1.0)
        return lower - delta, upper + delta

    padding = (upper - lower) * pad_ratio
    return lower - padding, upper + padding


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, which="major", color="#cfd3db", linewidth=0.75, alpha=0.55)
    ax.grid(True, which="minor", axis="y", color="#d9dde4", linewidth=0.55, alpha=0.4)
    ax.set_axisbelow(True)

    for spine_name in ("top", "right", "bottom", "left"):
        ax.spines[spine_name].set_linewidth(0.95)
        ax.spines[spine_name].set_color("#bcc1c9")

    ax.tick_params(axis="both", labelsize=19, colors="#2f3744", width=0.8)
    ax.xaxis.label.set_color("#1d2530")
    ax.yaxis.label.set_color("#1d2530")


def label_offsets() -> dict[tuple[int, int], tuple[float, float, str]]:
    return {
        (5, 5): (-10, -8, "right"),
        (5, 10): (8, 0, "left"),
        (5, 15): (8, 8, "left"),
        (8, 5): (8, 0, "left"),
        (8, 10): (-10, -8, "right"),
        (8, 15): (-10, 8, "right"),
    }


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 26,
            "axes.labelsize": 24,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")
    style_axes(ax)

    step_styles = {
        5: {"line": "#b98bcf", "marker": "o"},
        8: {"line": "#9a61b8", "marker": "o"},
    }
    epoch_colors = {
        5: "#cda5df",
        10: "#9d60bd",
        15: "#5f2f86",
    }

    offsets = label_offsets()

    for step_num in (5, 8):
        records = sorted(
            [record for record in RECORDS if record.step_num == step_num],
            key=lambda record: record.epoch,
        )
        xs = np.array([record.valid_unique for record in records], dtype=float)
        ys = np.array([record.stability for record in records], dtype=float)

        ax.plot(xs, ys, color=step_styles[step_num]["line"], linewidth=1.8, alpha=0.72, zorder=2)

        for record in records:
            point_color = epoch_colors[record.epoch]
            ax.scatter(
                record.valid_unique,
                record.stability,
                s=520,
                color=point_color,
                alpha=0.13,
                linewidth=0,
                zorder=3,
            )
            ax.scatter(
                record.valid_unique,
                record.stability,
                s=180,
                color=point_color,
                edgecolor="white",
                linewidth=1.5,
                alpha=0.98,
                zorder=4,
            )

            dx, dy, ha = offsets[(record.step_num, record.epoch)]
            ax.annotate(
                f"{record.step_num}-step",
                (record.valid_unique, record.stability),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=18,
                color="#4b3f58",
                alpha=0.95,
                zorder=5,
            )

    x_values = [record.valid_unique for record in RECORDS]
    y_values = [record.stability for record in RECORDS]
    ax.set_xlim(*padded_limits(x_values, pad_ratio=0.08))
    ax.set_ylim(*padded_limits(y_values, pad_ratio=0.08))

    ax.set_xlabel("Valid & Unique")
    ax.set_ylabel("Molecular Stability")

    epoch_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=epoch_colors[epoch],
            markeredgecolor="white",
            markeredgewidth=1.2,
            markersize=8.5,
            label=f"Epoch {epoch}",
        )
        for epoch in (5, 10, 15)
    ]
    legend = ax.legend(
        handles=epoch_handles,
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        borderpad=0.6,
        handletextpad=0.5,
        fontsize=17,
    )
    legend.get_frame().set_edgecolor("#d6dbe3")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")

    output_path = Path("valid_unique_stability_tradeoff_epochs.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()
