#!/usr/bin/env python3
"""Render the current four-hour Agent search trajectories from a frozen snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STATUS_LABELS = {
    "valid": "valid submission",
    "valid-after-corrected-audit": "valid submission",
    "search-valid-finalization-failed": "search result; final replay failed",
    "no-submission": "search result; no submission",
    "running": "running",
}
WINDOW_MINUTES = 240.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    base = float(payload["base_score"])
    runs = payload["runs"]

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "normal"})
    figure, axis = plt.subplots(figsize=(13.5, 7.2), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors

    for index, run in enumerate(runs):
        points = list(run["points"])
        if points[-1]["minute"] < WINDOW_MINUTES:
            points.append(
                {
                    "attempt_id": "window-end",
                    "minute": WINDOW_MINUTES,
                    "score": points[-1]["score"],
                }
            )
        status = run["status"]
        linestyle = "-" if status in {"valid", "valid-after-corrected-audit"} else "--"
        if status == "running":
            linestyle = ":"
        label = f'{run["label"]} ({STATUS_LABELS[status]})'
        axis.step(
            [point["minute"] for point in points],
            [100 * point["score"] for point in points],
            where="post",
            linewidth=2.2,
            linestyle=linestyle,
            color=colors[index],
            label=label,
        )
        endpoint = points[-1]
        marker = "o" if status in {"valid", "valid-after-corrected-audit"} else "X"
        if status == "running":
            marker = ">"
        axis.scatter(
            [endpoint["minute"]],
            [100 * endpoint["score"]],
            color=colors[index],
            marker=marker,
            s=62,
            zorder=5,
        )
        if status != "running":
            axis.annotate(
                f'{100 * run["best_score"]:.2f}',
                (endpoint["minute"], 100 * endpoint["score"]),
                xytext=(-8, 6),
                textcoords="offset points",
                ha="right",
                va="bottom",
                color=colors[index],
            )

    axis.axhline(100 * base, color="#333333", linestyle=(0, (4, 3)), linewidth=1.2)
    axis.text(238, 100 * base - 0.12, f"base {100 * base:.2f}", ha="right", va="top")
    axis.set_xlim(0, WINDOW_MINUTES)
    axis.set_ylim(100 * (base - 0.004), 100 * (max(run["best_score"] for run in runs) + 0.009))
    axis.set_xticks(range(0, 241, 30))
    axis.set_xlabel("Minutes since Agent window started")
    axis.set_ylabel("Best full seven-task mean observed (0–100)")
    axis.set_title(
        "PostTrainBench⁰ four-hour Agent search trajectories\n"
        "Running best; full seven-task evaluations only"
    )
    axis.grid(alpha=0.22)
    axis.legend(loc="upper left", ncol=2, frameon=False)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=220)
    figure.savefig(args.output_prefix.with_suffix(".svg"))
    plt.close(figure)


if __name__ == "__main__":
    main()
