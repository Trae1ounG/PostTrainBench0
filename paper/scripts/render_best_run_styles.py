#!/usr/bin/env python3
"""Render best-run search trajectories and operator profiles from retained traces."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
TRACES = PAPER / "data" / "traces"
OUT = PAPER / "figures"
ANALYSIS = PAPER / "data" / "analysis"

BASE_SCORE = 44.07642857142857
MODEL_SPECS = [
    ("Kimi K2.6", lambda run: run["agent_variant"] == "kimi-k2.6"),
    ("GPT-5.5 high", lambda run: run["agent_variant"] == "gpt-5.5 [high]"),
    ("GPT-5.5 xhigh", lambda run: run["agent_variant"] == "gpt-5.5 [xhigh]"),
    ("GPT-5.6 high", lambda run: run["agent_variant"] == "gpt-5.6-sol [high]"),
    ("GPT-5.6 medium", lambda run: run["agent_variant"] == "gpt-5.6-sol [medium]"),
    ("GPT-5.6 xhigh", lambda run: run["agent_variant"] == "gpt-5.6-sol [xhigh]"),
    ("Claude Sonnet 4.6", lambda run: run["agent_variant"] == "claude-4.6-sonnet-medium"),
    ("Claude Opus 4.8", lambda run: run["agent_variant"] == "claude-opus-4-8-high"),
    ("DeepSeek V4 Pro", lambda run: run["agent_variant"] == "ali-deepseek-v4-pro"),
    ("GLM-5.1", lambda run: run["agent_variant"] == "glm-5.1"),
    ("MiniMax M2.7", lambda run: run["agent_variant"] == "Minimax-M2.7-highspeed"),
    ("Qwen3.7-Max", lambda run: run["agent_variant"] == "openai_qwen3.7-max"),
]
COLORS = ["#D97706", "#2A7F62", "#59A14F", "#72A0C1", "#4E79A7", "#1F5FAE",
          "#7553A6", "#B64E73", "#677489", "#8C8C8C", "#9C755F", "#76B7B2"]
SINGLE = "#77AADD"
COMPOSED = "#44AA99"
INK = "#182230"
GRID = "#E5E9F0"


def select_runs() -> list[dict]:
    inventory = json.loads((TRACES / "agent_run_inventory_20260826_1922.json").read_text())["runs"]
    valid = [run for run in inventory if run.get("audit_verdict") == "valid"]
    selected = []
    for display_name, predicate in MODEL_SPECS:
        candidates = [run for run in valid if predicate(run)]
        if not candidates:
            raise RuntimeError(f"No valid run found for {display_name}")
        best = max(candidates, key=lambda run: run["best_observed_full_suite_score"])
        selected.append({**best, "display_name": display_name})
    return selected


def load_trajectories() -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    path = TRACES / "accepted_agent_trajectory_20260826_1922.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["run_id"]].append(
                {"minute": float(row["elapsed_minutes"]), "best": float(row["running_best"])}
            )
    return {run_id: {"runId": run_id, "points": points} for run_id, points in grouped.items()}


def render() -> None:
    selected = select_runs()
    trajectories = load_trajectories()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "selection_rule": "highest-scoring audit-valid Qwen2.5-3B-Instruct run per agent model",
        "base_score": BASE_SCORE,
        "runs": [
            {
                "model": run["display_name"],
                "run_id": run["run_id"],
                "best_score": run["best_observed_full_suite_score"] * 100,
                "one_direction_candidates": run["one_term_candidates"],
                "composed_candidates": run["multi_term_candidates"],
                "selected_direction_count": run["best_observed_term_count"],
                "submission_minute": 240.0 - run["submission_margin_seconds"] / 60.0,
            }
            for run in selected
        ],
    }
    (ANALYSIS / "best_run_search_styles.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "normal",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": "#7A8494",
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    fig = plt.figure(figsize=(7.15, 6.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.48)

    ax = fig.add_subplot(grid[0])
    for color, run in zip(COLORS, selected):
        trajectory = trajectories.get(run["run_id"])
        if trajectory is None:
            raise RuntimeError(f"Missing trajectory for {run['run_id']}")
        minutes = np.array([0.0] + [point["minute"] for point in trajectory["points"]])
        running_best = np.array(
            [BASE_SCORE] + [point["best"] * 100 for point in trajectory["points"]]
        )
        submission_minute = float(
            np.clip(240.0 - run["submission_margin_seconds"] / 60.0, minutes[-1], 240.0)
        )
        solid_minutes = np.append(minutes, submission_minute)
        solid_best = np.append(running_best, running_best[-1])
        ax.step(
            solid_minutes,
            solid_best,
            where="post",
            lw=1.7,
            color=color,
            label=run["display_name"],
        )
        if submission_minute < 240.0:
            ax.plot(
                [submission_minute, 240.0],
                [running_best[-1], running_best[-1]],
                lw=1.0,
                ls=(0, (2, 2)),
                color=color,
                alpha=0.35,
            )
        ax.scatter(
            submission_minute,
            running_best[-1],
            s=19,
            marker="s",
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )

    ax.axhline(BASE_SCORE, color="#7A8494", lw=0.8, ls=(0, (4, 3)))
    ax.text(238, BASE_SCORE + 0.10, "base 44.08", ha="right", va="bottom", color="#667085")
    ax.set_xlim(0, 240)
    ax.set_ylim(43.8, 50.0)
    ax.set_xticks([0, 60, 120, 180, 240])
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Best score so far")
    ax.grid(axis="both", alpha=0.9)
    ax.legend(
        ncol=3,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.08),
        handlelength=2.2,
        columnspacing=1.15,
        fontsize=5.9,
    )
    ax.text(-0.075, 1.04, "a", transform=ax.transAxes, weight="bold", fontsize=9)

    ax = fig.add_subplot(grid[1])
    names = [run["display_name"] for run in selected]
    single = np.array([run["one_term_candidates"] for run in selected], dtype=float)
    composed = np.array([run["multi_term_candidates"] for run in selected], dtype=float)
    denominators = single + composed
    single_pct = single / denominators * 100
    composed_pct = composed / denominators * 100
    y = np.arange(len(selected))

    ax.barh(y, single_pct, color=SINGLE, height=0.58, label="One-direction candidates")
    ax.barh(
        y,
        composed_pct,
        left=single_pct,
        color=COMPOSED,
        height=0.58,
        label="Composed candidates",
    )
    for idx, run in enumerate(selected):
        score = run["best_observed_full_suite_score"] * 100
        terms = run["best_observed_term_count"]
        ax.text(102.2, idx, f"k={terms} · {score:.2f}", va="center", color=INK)

    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 126)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of evaluated candidate programs (%)")
    ax.grid(axis="x", alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=2,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.08),
        handlelength=1.4,
        columnspacing=1.6,
    )
    ax.text(-0.075, 1.04, "b", transform=ax.transAxes, weight="bold", fontsize=9)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"fig6b-best-run-search-styles.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    render()
