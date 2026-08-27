#!/usr/bin/env python3
"""Render the static export of the website's default trajectory summary."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


BASE_SCORE = 0.440764
TIME_LIMIT_MINUTES = 240.0
COLORS = [
    "#2457ff", "#de5b3f", "#078b71", "#8a4bd0", "#d39400", "#1686b0",
    "#cc3d7e", "#667085", "#76a000", "#6b4f3f", "#111827", "#00a3a3",
    "#e11d48", "#f97316", "#4f46e5", "#059669",
]


def agent_group(run: dict) -> str:
    if run["agent"] == "es":
        return "es-conservative" if "conservative" in run["runId"] else "es-original"
    effort = re.search(r"\[(medium|high|xhigh)\]", run["label"])
    return f'{run["agent"]}:{effort.group(1)}' if effort else run["agent"]


def clean_agent_name(agent: str) -> str:
    model, _, effort = agent.partition(":")
    names = {
        "claude-4.6-sonnet-medium": "Sonnet 4.6 medium",
        "claude-opus-4-8-high": "Opus 4.8 high",
        "ali-deepseek-v4-pro": "DeepSeek V4 Pro",
        "gpt-5.4-pro-2026-03-05": "GPT-5.4 Pro",
        "gpt-5.6-sol": "GPT-5.6 Sol",
        "kimi-k2.6": "Kimi K2.6",
        "glm-5.1": "GLM-5.1",
        "openai_qwen3.7-max": "Qwen3.7-Max",
        "Minimax-M2.7-highspeed": "MiniMax M2.7",
        "randopt": "RandOPT",
        "es-conservative": "Evolution strategy · conservative",
        "es-original": "Evolution strategy · original",
    }
    return f"{names.get(model, model)} {effort}".strip()


def score_of(run: dict) -> float | None:
    return run["observedBest"] if run["observedBest"] is not None else run["finalScore"]


def eligible_runs(runs: list[dict], *, include_baselines: bool) -> list[dict]:
    return [
        run for run in runs
        if "smoke" not in run["runId"]
        and run["agent"] != "gpt-5.5-2026-04-24"
        and (include_baselines or run["kind"] == "agent")
    ]


def top_three_per_setting(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(agent_group(run), run["harness"])].append(run)
    selected: list[dict] = []
    for group in grouped.values():
        scored = [run for run in group if score_of(run) is not None]
        selected.extend(sorted(scored, key=score_of, reverse=True)[:3])
    return selected


def value_at(run: dict, minute: float) -> float:
    value = BASE_SCORE
    for point in run["points"]:
        if point["minute"] > minute:
            break
        value = point["best"]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    runs = payload["runs"]

    # Match InteractiveTrajectory.tsx: colors see all groups, while the default
    # plot hides baselines and keeps the best three runs per Agent + harness.
    all_groups = sorted({agent_group(run) for run in eligible_runs(runs, include_baselines=True)})
    color_by_agent = {group: COLORS[index % len(COLORS)] for index, group in enumerate(all_groups)}
    retained = top_three_per_setting(eligible_runs(runs, include_baselines=False))
    retained_by_agent: dict[str, list[dict]] = defaultdict(list)
    for run in retained:
        retained_by_agent[agent_group(run)].append(run)

    import matplotlib.pyplot as plt
    import numpy as np

    minutes = np.linspace(0, TIME_LIMIT_MINUTES, 121)
    summaries = []
    for agent in sorted(retained_by_agent):
        agent_runs = retained_by_agent[agent]
        values = np.array([[value_at(run, minute) for minute in minutes] for run in agent_runs])
        summaries.append({
            "agent": agent,
            "runs": agent_runs,
            "center": np.median(values, axis=0),
            "low": np.min(values, axis=0),
            "high": np.max(values, axis=0),
        })

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "normal",
        "axes.labelcolor": "#1f2937",
        "xtick.color": "#475467",
        "ytick.color": "#475467",
    })
    figure, axis = plt.subplots(figsize=(15.5, 7.6), constrained_layout=True)

    for summary in summaries:
        agent = summary["agent"]
        color = color_by_agent[agent]
        run_count = len(summary["runs"])
        if run_count > 1:
            axis.fill_between(
                minutes, 100 * summary["low"], 100 * summary["high"],
                step="post", color=color, alpha=0.09, linewidth=0,
            )
        axis.step(
            minutes, 100 * summary["center"], where="post", linewidth=1.9,
            color=color, label=f"{clean_agent_name(agent)}  ·  n={run_count}",
        )
        axis.scatter(
            [TIME_LIMIT_MINUTES], [100 * summary["center"][-1]], color=color,
            s=28, zorder=5, edgecolors="white", linewidths=0.7,
        )

    all_values = [BASE_SCORE]
    for summary in summaries:
        all_values.extend(summary["low"])
        all_values.extend(summary["high"])

    axis.axhline(100 * BASE_SCORE, color="#344054", linestyle=(0, (4, 3)), linewidth=1.1)
    axis.text(238, 100 * BASE_SCORE - 0.13, f"Base  {100 * BASE_SCORE:.2f}", ha="right", va="top", color="#344054")
    axis.set_xlim(0, TIME_LIMIT_MINUTES)
    axis.set_ylim(100 * min(all_values) - 0.45, 100 * max(all_values) + 0.65)
    axis.set_xticks(range(0, 241, 30))
    axis.set_xlabel("Minutes since Agent window started")
    axis.set_ylabel("Best seven-task mean observed so far (0–100)")
    axis.grid(axis="both", color="#d0d5dd", alpha=0.45, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="center left", bbox_to_anchor=(1.015, 0.5), ncol=1, frameon=False,
        fontsize=9.1, handlelength=2.5, labelspacing=0.9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
