#!/usr/bin/env python3
"""Render the Qwen2.5-3B weight-neighborhood analysis used as Figure 3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "public" / "weight-space-data.json"
OUTPUT = ROOT / "public" / "figures" / "qwen25-3b-weight-projection"
BASE_JOINT_SCORE = 44.0764

TASKS = ["countdown", "gsm8k", "math500", "olympiadbench", "mbpp", "rocstories", "uspto50k"]
LABELS = {
    "countdown": "Countdown",
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "olympiadbench": "Olympiad",
    "mbpp": "MBPP",
    "rocstories": "ROCStories",
    "uspto50k": "USPTO-50K",
}
COLORS = {
    "countdown": "#2563EB",
    "gsm8k": "#7C3AED",
    "math500": "#DB2777",
    "olympiadbench": "#DC2626",
    "mbpp": "#EA580C",
    "rocstories": "#059669",
    "uspto50k": "#0891B2",
}
OFFSETS = {
    "countdown": (8, -16),
    "gsm8k": (-45, -17),
    "math500": (8, 7),
    "olympiadbench": (8, 7),
    "mbpp": (8, -4),
    "rocstories": (8, -4),
    "uspto50k": (-58, 7),
}


def main() -> None:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    if payload.get("model") != "Qwen2.5-3B-Instruct":
        raise ValueError(f"unexpected model: {payload.get('model')}")

    rows = payload["points"]
    points = np.asarray([row["pca"] for row in rows], dtype=float)
    joint = np.asarray([row["scores"]["joint"] * 100 for row in rows], dtype=float)
    task_scores = {
        task: np.asarray([row["scores"][task] for row in rows], dtype=float)
        for task in TASKS
    }
    best_joint = int(np.argmax(joint))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
        }
    )
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.9), gridspec_kw={"wspace": 0.22})

    scatter = left.scatter(
        points[:, 0], points[:, 1], c=joint, cmap="viridis", s=18,
        alpha=0.82, linewidths=0, rasterized=True,
    )
    left.scatter(0, 0, marker="P", s=88, facecolor="white", edgecolor="#111827", linewidth=1.0, zorder=5)
    left.scatter(
        points[best_joint, 0], points[best_joint, 1], marker="*", s=210,
        facecolor="#FACC15", edgecolor="#111827", linewidth=0.9, zorder=6,
    )
    left.annotate(
        f"best joint  {joint[best_joint]:.2f}", points[best_joint], xytext=(10, 14),
        textcoords="offset points", fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": "#64748B", "lw": 0.8},
    )
    left.set_title("(a) Joint objective", loc="left", fontsize=10, pad=7)
    left.set_xlabel("weight-difference PCA axis 1")
    left.set_ylabel("weight-difference PCA axis 2")
    left.set_aspect("equal", adjustable="box")
    colorbar = figure.colorbar(scatter, ax=left, shrink=0.82, pad=0.02)
    colorbar.set_label("seven-task mean score (0–100)")
    colorbar.ax.axhline(BASE_JOINT_SCORE, color="white", linewidth=1.2, linestyle="--")

    right.scatter(points[:, 0], points[:, 1], s=13, color="#CBD5E1", alpha=0.48, linewidths=0, rasterized=True)
    right.scatter(0, 0, marker="P", s=88, facecolor="white", edgecolor="#111827", linewidth=1.0, zorder=5)
    for task in TASKS:
        best = int(np.argmax(task_scores[task]))
        endpoint = points[best]
        color = COLORS[task]
        right.annotate(
            "", xy=endpoint, xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5, "mutation_scale": 11},
        )
        right.scatter(endpoint[0], endpoint[1], marker="*", s=135, facecolor=color, edgecolor="white", linewidth=0.7, zorder=6)
        right.annotate(
            LABELS[task], endpoint, xytext=OFFSETS[task], textcoords="offset points",
            fontsize=8.1, fontweight="bold", color=color,
        )
    right.set_title("(b) Task specialists", loc="left", fontsize=10, pad=7)
    right.set_xlabel("weight-difference PCA axis 1")
    right.set_ylabel("weight-difference PCA axis 2")
    right.set_aspect("equal", adjustable="box")

    figure.suptitle("Qwen2.5-3B-Instruct · 1,000 evaluated perturbations", fontsize=13, fontweight="bold", y=0.99)
    figure.text(
        0.5, 0.935,
        "Coordinates use checkpoint weight differences only; scores do not determine the projection.",
        ha="center", color="#64748B", fontsize=8.5,
    )
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.86, wspace=0.22)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
