#!/usr/bin/env python3
"""Render Figure 2 for the PostTrainBench^0 research blog."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / "public/figures/premise-check.png"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.7,
            "savefig.dpi": 220,
        }
    )

    labels = ["Qwen2.5-3B", "Qwen2.5-7B"]
    base = np.array([44.0764, 55.7229])
    best = np.array([47.5807, 57.7214])
    oracle = np.array([52.1800, 62.2043])
    improved = np.array([51, 49])
    total = 192
    y = np.arange(len(labels))

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.55),
        gridspec_kw={"width_ratios": [1.34, 1], "wspace": 0.42},
    )

    for index in y:
        left.plot(
            [base[index], oracle[index]],
            [index, index],
            color="#CBD5E1",
            linewidth=3,
            solid_capstyle="round",
            zorder=1,
        )
    left.scatter(base, y, s=76, color="#64748B", label="Base checkpoint", zorder=3)
    left.scatter(best, y, s=88, color="#2563EB", label="Best single checkpoint", zorder=3)
    left.scatter(oracle, y, s=100, marker="*", color="#F59E0B", label="Task-wise oracle", zorder=3)

    for index in y:
        left.annotate(
            f"+{best[index] - base[index]:.2f}",
            (best[index], index),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#1D4ED8",
            fontsize=8.5,
        )
        left.annotate(
            f"oracle gap +{oracle[index] - best[index]:.2f}",
            (oracle[index], index),
            xytext=(-4, -20),
            textcoords="offset points",
            ha="right",
            va="top",
            color="#B45309",
            fontsize=8.5,
        )

    left.set_yticks(y, labels)
    left.set_ylim(1.28, -0.22)
    left.set_xlim(41, 65)
    left.set_xlabel("Seven-task mean score (0–100)")
    left.set_title("A  Better joint checkpoints exist, but task optima disagree", loc="left", fontweight="bold")
    left.legend(frameon=False, loc="center", bbox_to_anchor=(0.56, 0.49))
    left.grid(axis="y", visible=False)

    remaining = total - improved
    right.barh(y, improved, color="#2563EB", height=0.42)
    right.barh(y, remaining, left=improved, color="#E2E8F0", height=0.42)
    for index in y:
        right.text(
            improved[index] / 2,
            index,
            f"{improved[index]}\nimproved",
            va="center",
            ha="center",
            fontsize=8.5,
            color="white",
            fontweight="bold",
        )
        right.text(
            improved[index] + remaining[index] / 2,
            index,
            f"{remaining[index]}\nat or below base",
            va="center",
            ha="center",
            fontsize=8.5,
            color="#334155",
        )

    right.set_yticks(y, labels)
    right.set_ylim(1.22, -0.22)
    right.set_xlim(0, total)
    right.set_xlabel("Randomly perturbed checkpoints")
    right.set_title("B  Most nearby candidates do not improve the mean", loc="left", fontweight="bold")
    right.grid(axis="y", visible=False)

    fig.suptitle(
        "Premise check: local improvement is possible, but joint optimization is non-trivial",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.15, top=0.77)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
