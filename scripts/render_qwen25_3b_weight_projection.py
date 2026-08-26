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
    "countdown": (-72, 9),
    "gsm8k": (8, -2),
    "math500": (8, -18),
    "olympiadbench": (8, 8),
    "mbpp": (8, -17),
    "rocstories": (8, -7),
    "uspto50k": (-58, 8),
}


def smooth_surface(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate only where measured candidates provide local support."""
    pad = np.maximum(np.ptp(points, axis=0) * 0.08, 0.25)
    lower = points.min(axis=0) - pad
    upper = points.max(axis=0) + pad
    gx, gy = np.meshgrid(np.linspace(lower[0], upper[0], 88), np.linspace(lower[1], upper[1], 80))
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    pair = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    pair.sort(axis=1)
    width = max(float(np.median(pair[:, 12])) * 1.05, 0.20)
    squared = np.sum((grid[:, None, :] - points[None, :, :]) ** 2, axis=2)
    weights = np.exp(-squared / (2 * width**2))
    support = weights.sum(axis=1)
    surface = weights @ values / np.maximum(support, 1e-9)
    nearest = np.sqrt(squared.min(axis=1))
    mask = nearest > 1.65 * width
    return gx, gy, np.ma.masked_where(mask.reshape(gx.shape), surface.reshape(gx.shape))


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
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
        }
    )
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.9), gridspec_kw={"width_ratios": [1.08, 1]})

    gx, gy, surface = smooth_surface(points, joint)
    levels = np.linspace(float(np.percentile(joint, 5)), float(np.percentile(joint, 95)), 12)
    image = left.contourf(
        gx, gy, surface, levels=levels, cmap="viridis", alpha=0.90, extend="both",
    )
    left.contour(gx, gy, surface, levels=[BASE_JOINT_SCORE], colors="white", linewidths=1.1, linestyles="--")
    left.scatter(
        points[:, 0], points[:, 1], c=joint, cmap="viridis", vmin=levels[0], vmax=levels[-1],
        s=17, marker="o", alpha=0.62, edgecolor="white", linewidth=0.28, rasterized=True,
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
    left.set_title("(a)  Joint objective", loc="left", fontsize=10.5, pad=7)
    left.set_xlabel("parameter projection 1")
    left.set_ylabel("parameter projection 2")
    left.set_aspect("equal", adjustable="box")
    colorbar = figure.colorbar(image, ax=left, shrink=0.82, pad=0.02)
    colorbar.set_label("seven-task mean score (0–100)")

    right.scatter(points[:, 0], points[:, 1], s=13, color="#CBD5E1", alpha=0.42, linewidths=0, rasterized=True)
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
    right.set_title("(b)  Task specialists", loc="left", fontsize=10.5, pad=7)
    right.set_xlabel("parameter projection 1")
    right.set_ylabel("parameter projection 2")
    right.set_aspect("equal", adjustable="box")

    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.13, top=0.94, wspace=0.20)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
