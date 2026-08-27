#!/usr/bin/env python3
"""Render the paper figures from the retained analysis traces."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.interpolate import RBFInterpolator, griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
OUT = PAPER / "figures"
SUMMARY = PAPER / "data" / "agent_run_summary.csv"
TRACES = PAPER / "data" / "traces"
WEIGHTS = TRACES / "weight_space_candidates.json"
TRAJECTORIES = TRACES / "accepted_agent_trajectory_20260826_1922.csv"
INVENTORY = TRACES / "agent_run_inventory_20260826_1922.json"
PROJECTED_TRACES = TRACES / "checkpoint_projected_traces.json"

CONFIG_SPECS = [
    ("Kimi K2.6", "kimi-k2.6"),
    ("GPT-5.5 high", "gpt-5.5 [high]"),
    ("GPT-5.5 xhigh", "gpt-5.5 [xhigh]"),
    ("GPT-5.6 high", "gpt-5.6-sol [high]"),
    ("GPT-5.6 medium", "gpt-5.6-sol [medium]"),
    ("GPT-5.6 xhigh", "gpt-5.6-sol [xhigh]"),
    ("DeepSeek V4 Pro", "ali-deepseek-v4-pro"),
    ("Opus 4.8 high", "claude-opus-4-8-high"),
    ("Sonnet 4.6 medium", "claude-4.6-sonnet-medium"),
    ("GLM-5.1", "glm-5.1"),
    ("MiniMax M2.7", "Minimax-M2.7-highspeed"),
    ("Qwen3.7-Max", "openai_qwen3.7-max"),
]

BLUE = "#1F5FAE"
ORANGE = "#D97706"
GREEN = "#2A7F62"
PURPLE = "#7553A6"
RED = "#BA3B46"
GRAY = "#667085"
LIGHT = "#D8DEE8"
PALE = "#EEF2F7"
INK = "#182230"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "normal",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": "#536173",
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#E8ECF1",
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.02, label.lower(), transform=ax.transAxes, weight="bold", fontsize=9)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def load_weight_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    payload = json.loads(WEIGHTS.read_text())
    tasks = payload["tasks"]
    joint = np.array([p["scores"]["joint"] * 100 for p in payload["points"]])
    task_scores = np.array([[p["scores"][task] for task in tasks] for p in payload["points"]])
    return tasks, joint, task_scores


def load_base_task_scores(tasks: list[str]) -> np.ndarray:
    runs = json.loads(INVENTORY.read_text())["runs"]
    source = next(run for run in runs if run.get("task_scores") and run.get("task_changes"))
    return np.array([source["task_scores"][task] - source["task_changes"][task] for task in tasks])


def load_trajectory_runs() -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    labels: dict[str, str] = {}
    with TRAJECTORIES.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = row["run_id"]
            labels[run_id] = row["label"]
            grouped[run_id].append(
                {
                    "minute": float(row["elapsed_minutes"]),
                    "score": float(row["score"]),
                    "best": float(row["running_best"]),
                }
            )
    return [
        {
            "runId": run_id,
            "label": labels[run_id],
            "points": points,
            "evaluations": len(points),
            "observedBest": max(point["best"] for point in points),
        }
        for run_id, points in grouped.items()
    ]


def render_search_space() -> None:
    _, joint, task_scores = load_weight_data()
    base = 44.08
    oracle = task_scores.max(axis=0).mean() * 100
    ordered = np.sort(joint)
    ranks = np.linspace(0, 100, len(ordered), endpoint=False)
    above = ordered > base

    budgets = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000])
    rng = np.random.default_rng(42)
    samples = np.empty((2000, len(budgets)))
    for repeat in range(samples.shape[0]):
        sequence = rng.permutation(joint)
        running = np.maximum.accumulate(sequence)
        samples[repeat] = running[budgets - 1]
    q10, q50, q90 = np.percentile(samples, [10, 50, 90], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), gridspec_kw={"wspace": 0.30})
    ax = axes[0]
    ax.fill_between(ranks, ordered, base, where=above, color=ORANGE, alpha=0.16, interpolate=True)
    ax.plot(ranks[~above], ordered[~above], color="#AAB3C0", lw=1.5)
    ax.plot(ranks[above], ordered[above], color=ORANGE, lw=1.8)
    ax.axhline(base, color=INK, ls="--", lw=0.9)
    first_above = int(np.argmax(above))
    ax.annotate(
        f"{above.sum()} / {len(joint):,} above base",
        (ranks[first_above], base),
        xytext=(25, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.7},
        color=ORANGE,
        fontsize=7,
    )
    ax.text(1.5, base - 0.18, f"base {base:.2f}", color=GRAY, va="top")
    ax.text(98.5, ordered[-1] + 0.12, f"best {ordered[-1]:.2f}", color=ORANGE, ha="right")
    ax.set(xlabel="Candidate percentile", ylabel="Seven-task score", xlim=(0, 100))
    ax.grid(axis="y")
    panel_label(ax, "A")

    ax = axes[1]
    ax.fill_between(budgets, q10, q90, color=BLUE, alpha=0.16, label="10--90% search paths")
    ax.plot(budgets, q50, color=BLUE, lw=1.8, marker="o", ms=3.2, label="median best found")
    ax.axhline(base, color=INK, ls="--", lw=0.9)
    ax.axhline(oracle, color=ORANGE, ls=":", lw=1.2)
    ax.text(950, oracle + 0.10, f"task-wise oracle {oracle:.2f}", ha="right", color=ORANGE)
    ax.text(950, base - 0.12, "base", ha="right", va="top", color=GRAY)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1000], ["1", "4", "16", "64", "256", "1k"])
    ax.set(xlabel="Random candidates evaluated", ylabel="Best score found")
    ax.grid(True)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.88),
        handlelength=1.8,
    )
    panel_label(ax, "B")

    save(fig, "fig2-search-space")


def render_weight_space_landscape() -> None:
    """Reproduce the Neural Thickets-style local accuracy landscape."""
    payload = json.loads(WEIGHTS.read_text())
    coordinates = np.array([point["random"] for point in payload["points"]])
    tasks = payload["tasks"]
    base = load_base_task_scores(tasks)
    base_by_task = dict(zip(tasks, base))
    base_by_task["joint"] = float(np.mean(base))

    x_limit = float(np.quantile(np.abs(coordinates[:, 0]), 0.995)) * 1.05
    y_limit = float(np.quantile(np.abs(coordinates[:, 1]), 0.995)) * 1.05
    grid_x, grid_y = np.meshgrid(np.linspace(-x_limit, x_limit, 220), np.linspace(-y_limit, y_limit, 220))

    def normalized_surface(task: str) -> tuple[np.ndarray, np.ndarray]:
        scores = np.array([point["scores"][task] for point in payload["points"]])
        change = scores - base_by_task[task]
        negative = max(abs(float(np.quantile(change, 0.02))), 1e-8)
        positive = max(abs(float(np.quantile(change, 0.98))), 1e-8)
        normalized = np.where(change < 0, change / negative, change / positive)
        cubic = griddata(coordinates, normalized, (grid_x, grid_y), method="cubic")
        nearest = griddata(coordinates, normalized, (grid_x, grid_y), method="nearest")
        surface = np.where(np.isfinite(cubic), cubic, nearest)
        return np.clip(gaussian_filter(surface, sigma=2.4), -1, 1), scores

    task_panels = [
        ("joint", "Seven-task average"),
        ("countdown", "Countdown"),
        ("gsm8k", "GSM8K"),
        ("math500", "MATH-500"),
        ("olympiadbench", "OlympiadBench"),
        ("mbpp", "MBPP"),
        ("rocstories", "ROCStories"),
        ("uspto50k", "USPTO-50K"),
    ]
    surfaces = {task: normalized_surface(task) for task, _ in task_panels}

    fig, axes_grid = plt.subplots(2, 4, figsize=(7.15, 3.45), gridspec_kw={"wspace": 0.07, "hspace": 0.20})
    axes = axes_grid.ravel()
    image = None
    for ax, (task, title) in zip(axes, task_panels):
        surface, scores = surfaces[task]
        image = ax.imshow(
            surface, origin="lower", extent=(-x_limit, x_limit, -y_limit, y_limit),
            cmap="RdYlBu_r", vmin=-1, vmax=1, interpolation="bilinear", aspect="equal",
        )
        best = int(np.argmax(scores))
        gain = (scores[best] - base_by_task[task]) * 100
        ax.add_patch(plt.Circle((0, 0), np.sqrt(2), fill=False, color=INK, ls="--", lw=0.65, alpha=0.75))
        ax.scatter(0, 0, marker="*", s=35, color=INK, edgecolor="white", linewidth=0.4, zorder=4)
        ax.scatter(*coordinates[best], marker="*", s=43, color="#F4C542", edgecolor=INK, linewidth=0.55, zorder=5)
        ax.set_title(title, fontsize=7.5, pad=3, fontweight="normal")
        ax.text(
            0.04, 0.05, rf"$\Delta_{{\max}}={gain:+.1f}$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 1.0},
        )
        ax.set(xlim=(-x_limit, x_limit), ylim=(-y_limit, y_limit))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(True)
        ax.spines[:].set_linewidth(0.45)

    legend = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=INK, markeredgecolor="white", markersize=7, label="Frozen base"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#F4C542", markeredgecolor=INK, markersize=8, label="Best for this panel"),
        Line2D([0], [0], color=INK, ls="--", lw=0.8, label="Typical perturbation range"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.43, -0.025), ncol=3, frameon=False, fontsize=6.2)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.014, pad=0.018)
    colorbar.set_ticks([-1, 0, 1])
    colorbar.set_ticklabels(["lower", "base", "higher"])
    colorbar.set_label("Relative to base")
    save(fig, "fig3-weight-space-landscape")


def render_weight_space_walks() -> None:
    """Overlay complete retained checkpoint walks on their empirical score fields."""
    payload = json.loads(PROJECTED_TRACES.read_text())
    runs = [run for run in payload["runs"] if run.get("status") == "valid" and run.get("submission", {}).get("candidate_path")]
    fig, axes = plt.subplots(1, len(runs), figsize=(7.15, 2.25), gridspec_kw={"wspace": 0.10})
    axes = np.atleast_1d(axes)
    for index, (ax, run) in enumerate(zip(axes, runs)):
        rows = run["rows"]
        xy = np.array([[row["x"], row["y"]] for row in rows]) * 1000
        scores = np.array([row["score"] * 100 for row in rows])
        unique_xy, inverse = np.unique(xy, axis=0, return_inverse=True)
        unique_score = np.array([scores[inverse == group].mean() for group in range(len(unique_xy))])
        pad = np.maximum(np.ptp(unique_xy, axis=0) * 0.10, 0.35)
        low, high = unique_xy.min(axis=0) - pad, unique_xy.max(axis=0) + pad
        grid_x, grid_y = np.meshgrid(np.linspace(low[0], high[0], 150), np.linspace(low[1], high[1], 150))
        coordinate_scale = np.maximum(np.std(unique_xy, axis=0), 0.25)
        normalized_xy = unique_xy / coordinate_scale
        normalized_grid = np.column_stack([grid_x.ravel() / coordinate_scale[0], grid_y.ravel() / coordinate_scale[1]])
        try:
            interpolator = RBFInterpolator(
                normalized_xy,
                unique_score,
                kernel="thin_plate_spline",
                smoothing=0.45,
                neighbors=min(40, len(unique_xy)),
            )
            field = interpolator(normalized_grid).reshape(grid_x.shape)
        except Exception:
            field = griddata(unique_xy, unique_score, (grid_x, grid_y), method="nearest")
        field = gaussian_filter(field, sigma=2.6)
        field = np.clip(field, np.quantile(unique_score, 0.03), np.quantile(unique_score, 0.97))
        center = 41.08
        scale = max(abs(float(np.quantile(field - center, 0.05))), abs(float(np.quantile(field - center, 0.95))), 0.5)
        ax.imshow(
            field, origin="lower", extent=(low[0], high[0], low[1], high[1]),
            cmap="RdYlBu_r", norm=TwoSlopeNorm(vmin=center - scale, vcenter=center, vmax=center + scale),
            interpolation="bilinear", aspect="auto", alpha=0.78,
        )
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        progress = np.linspace(0, 1, len(segments))
        ax.add_collection(LineCollection(segments, cmap="viridis", array=progress, linewidth=1.15, alpha=0.88, zorder=3))
        stride = max(2, len(rows) // 12)
        for start in range(0, len(rows) - 1, stride):
            end = min(start + stride, len(rows) - 1)
            delta = xy[end] - xy[start]
            ax.annotate(
                "", xy=tuple(xy[end]), xytext=tuple(xy[start]),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.42, "mutation_scale": 5.2, "alpha": 0.60},
                zorder=4,
            )
        best = int(np.argmax(scores))
        ax.scatter(0, 0, marker="*", s=52, color=INK, edgecolor="white", linewidth=0.6, zorder=5)
        ax.scatter(xy[best, 0], xy[best, 1], marker="*", s=67, color="#F4C542",
                   edgecolor=INK, linewidth=0.55, zorder=6)
        ax.set_title(run["label"], loc="left", fontsize=7.8, pad=3, fontweight="normal")
        panel_label(ax, chr(ord("A") + index))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(True)
        ax.spines[:].set_linewidth(0.45)
        ax.grid(False)
        ax.margins(0.12)
    time_map = mpl.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(0, 1))
    time_bar = fig.colorbar(time_map, ax=axes, orientation="horizontal", fraction=0.045, pad=0.10, aspect=55)
    time_bar.set_ticks([0, 1])
    time_bar.set_ticklabels(["first evaluation", "last evaluation"])
    time_bar.set_label("Search order", labelpad=-1)
    legend = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=INK, markeredgecolor="white", markersize=7, label="Frozen base"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#F4C542", markeredgecolor=INK, markersize=8, label="Best checkpoint"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.50, -0.07), ncol=2, frameon=False, fontsize=6.2)
    save(fig, "fig7-weight-space-walks")


def render_task_conflict() -> None:
    tasks, _, task_scores = load_weight_data()
    base = load_base_task_scores(tasks)
    short = ["Countdown", "GSM8K", "MATH", "Olympiad", "MBPP", "ROCStories", "USPTO"]
    specialist_indices = [int(np.argmax(task_scores[:, index])) for index in range(len(tasks))]
    specialist_delta = (task_scores[specialist_indices] - base) * 100
    improved_counts = (task_scores > base).sum(axis=1)
    histogram = np.bincount(improved_counts, minlength=8)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), gridspec_kw={"width_ratios": [1.42, 0.85], "wspace": 0.56})
    ax = axes[0]
    limit = float(np.ceil(np.abs(specialist_delta).max()))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    im = ax.imshow(specialist_delta, cmap="RdBu", norm=norm, aspect="auto")
    for row in range(7):
        for col in range(7):
            value = specialist_delta[row, col]
            ax.text(col, row, f"{value:+.1f}", ha="center", va="center", fontsize=6.2,
                    color="white" if abs(value) > limit * 0.55 else INK,
                    weight="semibold" if row == col else "normal")
    ax.set_xticks(range(7), short, rotation=38, ha="right")
    ax.set_yticks(range(7), [f"best for {name}" for name in short])
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Score change")
    cb.ax.tick_params(length=2)
    panel_label(ax, "A")

    ax = axes[1]
    bars = ax.bar(range(8), histogram, color=[PALE] + [BLUE] * 6 + [ORANGE], edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, histogram):
        if count:
            ax.text(bar.get_x() + bar.get_width() / 2, count + 6, str(int(count)), ha="center", va="bottom", fontsize=6.5)
    ax.set(xlabel="Tasks improved by one candidate", ylabel="Candidates", xticks=range(8))
    ax.set_ylim(0, histogram.max() * 1.17)
    ax.grid(axis="y")
    ax.text(0.98, 0.93, "1,000 candidates", transform=ax.transAxes, ha="right", color=GRAY)
    panel_label(ax, "B")

    save(fig, "fig4-task-conflict")


def render_task_breakdown() -> None:
    """Per-task changes for every audit-valid configuration with task scores."""
    weight_payload = json.loads(WEIGHTS.read_text())
    tasks = weight_payload["tasks"]
    base = load_base_task_scores(tasks)
    random_best = max(weight_payload["points"][:500], key=lambda point: point["scores"]["joint"])

    inventory = json.loads(INVENTORY.read_text())["runs"]
    agent_rows = []
    for label, variant in CONFIG_SPECS:
        accepted = [
            run for run in inventory
            if run.get("audit_verdict") == "valid"
            and run.get("task_scores")
            and run.get("agent_variant") == variant
        ]
        if accepted:
            scores = np.mean([[run["task_scores"][task] for task in tasks] for run in accepted], axis=0)
            agent_rows.append((label, scores, len(accepted)))
    agent_rows.sort(key=lambda row: float(np.mean(row[1])), reverse=True)
    rows = [("Random-500", np.array([random_best["scores"][task] for task in tasks]), 1), *agent_rows]

    values = np.vstack([scores - base for _, scores, _ in rows]) * 100
    labels = [f"{label}  (n={count})" for label, _, count in rows]
    task_labels = ["Countdown", "GSM8K", "MATH-500", "Olympiad", "MBPP", "ROCStories", "USPTO"]

    fig, ax = plt.subplots(figsize=(7.15, 4.55))
    image = ax.imshow(values, cmap="RdBu", norm=TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12), aspect="auto")
    ax.set_xticks(np.arange(len(tasks)), task_labels, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(length=0)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if abs(values[row, col]) >= 7 else INK
            ax.text(col, row, f"{values[row, col]:+.1f}", ha="center", va="center", fontsize=6.5, color=color)
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Change from base (points)")
    save(fig, "fig4-task-breakdown")


def read_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SUMMARY.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "base_score": float(row["base_score"]),
                    "n": int(row["n"]),
                    "mean_score": float(row["mean_score"]),
                    "min_score": float(row["min_score"]),
                    "max_score": float(row["max_score"]),
                }
            )
    return rows


def render_main_result() -> None:
    """Render the shared-target result that anchors the first page."""
    rows = read_summary()
    left, right = "Qwen2.5-3B-Instruct", "Qwen3-4B-Base"
    by_target = {
        target: {row["agent_model"]: row for row in rows if row["target_model"] == target}
        for target in (left, right)
    }
    shared = sorted(set(by_target[left]) & set(by_target[right]))
    gains = {
        target: {
            agent: float(by_target[target][agent]["mean_score"])
            - float(by_target[target][agent]["base_score"])
            for agent in shared
        }
        for target in (left, right)
    }
    macro = {agent: (gains[left][agent] + gains[right][agent]) / 2 for agent in shared}
    order = sorted(shared, key=macro.get, reverse=True)
    labels = {
        "DeepSeek V4 Pro": "DeepSeek V4 Pro",
        "GPT-5.4 Pro": "GPT-5.4 Pro",
        "GPT-5.5 xhigh": "GPT-5.5 xhigh",
        "GPT-5.6 xhigh": "GPT-5.6 xhigh",
        "Kimi K2.6": "Kimi K2.6",
        "MiniMax M2.7": "MiniMax M2.7",
        "Qwen3.7-Max": "Qwen3.7-Max",
        "GLM-5.1": "GLM-5.1",
    }

    x = np.arange(len(order))
    values = np.array([macro[agent] for agent in order])
    fig, ax = plt.subplots(figsize=(7.15, 2.35))
    bars = ax.bar(x, values, width=0.67, color="#A96F50", edgecolor="white", linewidth=0.6)
    for index, agent in enumerate(order):
        ax.scatter(index - 0.11, gains[left][agent], s=27, marker="o", color=BLUE,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(index + 0.11, gains[right][agent], s=31, marker="D", color=ORANGE,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(index, values[index] + 0.14, f"{values[index]:.2f}", ha="center",
                va="bottom", fontsize=6.8, weight="semibold", color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [labels[agent] for agent in order], rotation=24, ha="right")
    ax.set_ylabel("Mean gain over base (points)")
    ax.set_ylim(0, max(values.max(), max(gains[right].values())) + 1.0)
    ax.grid(axis="y")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=27, marker="o", color=BLUE, label="Qwen2.5-3B-Instruct")
    ax.scatter([], [], s=31, marker="D", color=ORANGE, label="Qwen3-4B-Base")
    ax.legend(frameon=False, ncol=2, loc="upper right", handletextpad=0.4,
              columnspacing=1.3)
    ax.text(
        0.01,
        0.97,
        "bar: macro-average across the two targets",
        transform=ax.transAxes,
        va="top",
        color=GRAY,
        fontsize=6.6,
    )
    save(fig, "fig1-main-result")


def render_target_dependence() -> None:
    rows = read_summary()
    left, right = "Qwen2.5-3B-Instruct", "Qwen3-4B-Base"
    by_target = {target: {r["agent_model"]: r for r in rows if r["target_model"] == target} for target in (left, right)}
    shared = sorted(set(by_target[left]) & set(by_target[right]))
    short = {
        "DeepSeek V4 Pro": "DeepSeek V4", "GPT-5.4 Pro": "GPT-5.4", "GPT-5.5 xhigh": "GPT-5.5",
        "GPT-5.6 xhigh": "GPT-5.6", "Kimi K2.6": "Kimi K2.6", "MiniMax M2.7": "MiniMax M2.7",
        "Qwen3.7-Max": "Qwen3.7-Max", "GLM-5.1": "GLM-5.1",
    }
    gains = {
        target: {agent: float(by_target[target][agent]["mean_score"]) - float(by_target[target][agent]["base_score"]) for agent in shared}
        for target in (left, right)
    }
    ranks = {
        target: {agent: rank + 1 for rank, agent in enumerate(sorted(shared, key=gains[target].get, reverse=True))}
        for target in (left, right)
    }
    rho = spearmanr([gains[left][agent] for agent in shared], [gains[right][agent] for agent in shared]).statistic

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85), gridspec_kw={"width_ratios": [1.16, 1.0], "wspace": 0.60})
    ax = axes[0]
    palette = [BLUE, ORANGE, GREEN, PURPLE, RED, "#4F7C8A", "#9A6B3D", "#6B7280"]
    for color, agent in zip(palette, sorted(shared, key=ranks[left].get)):
        y0, y1 = ranks[left][agent], ranks[right][agent]
        ax.plot([0, 1], [y0, y1], color=color, lw=1.6, alpha=0.9)
        ax.scatter([0, 1], [y0, y1], s=25, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(-0.04, y0, f"{short[agent]}  (+{gains[left][agent]:.2f})", ha="right", va="center", fontsize=5.9)
        ax.text(1.04, y1, f"+{gains[right][agent]:.2f}", ha="left", va="center", fontsize=6.2)
    ax.set_xlim(-0.85, 1.23)
    ax.set_ylim(8.65, 0.35)
    ax.set_xticks([0, 1], ["Qwen2.5-3B\nprimary", "Qwen3-4B\ntarget ablation"])
    ax.set_yticks(range(1, 9), [])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y")
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.text(0.98, 0.02, rf"Spearman $\rho={rho:.2f}$", transform=ax.transAxes, ha="right", color=GRAY)
    panel_label(ax, "A")

    ax = axes[1]
    repeated = [row for row in rows if row["n"] >= 2]
    repeated.sort(key=lambda row: float(row["max_score"]) - float(row["min_score"]), reverse=True)
    repeated = repeated[:10]
    y_pos = np.arange(len(repeated))
    colors = [BLUE if row["target_model"] == left else ORANGE for row in repeated]
    for pos, row, color in zip(y_pos, repeated, colors):
        lo, mean, hi = float(row["min_score"]), float(row["mean_score"]), float(row["max_score"])
        ax.plot([lo, hi], [pos, pos], color=LIGHT, lw=3.0, solid_capstyle="round")
        ax.scatter([lo, hi], [pos, pos], s=9, color="#AAB3C0", zorder=2)
        ax.plot(mean, pos, marker="|", ms=10, mew=1.7, color=color, zorder=3)
    labels = [f"{row['agent_model']} · {'Q2.5' if row['target_model'] == left else 'Q3'}" for row in repeated]
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Run score: min, mean, max")
    ax.grid(axis="x")
    ax.tick_params(axis="y", labelsize=6.1, length=0)
    ax.text(0.98, 0.03, "blue: primary   orange: target ablation", transform=ax.transAxes, ha="right", color=GRAY, fontsize=6.2)
    panel_label(ax, "B")

    save(fig, "fig3-target-dependence")


def family_name(agent: str) -> str:
    return {
        "kimi-k2.6": "Kimi K2.6",
        "gpt-5.6-sol": "GPT-5.6",
        "claude-opus-4-8-high": "Opus 4.8",
    }.get(agent, agent)


def render_trace_cases() -> None:
    runs = load_trajectory_runs()
    inventory = json.loads(INVENTORY.read_text())["runs"]
    valid = {
        run["run_id"]: run
        for run in inventory
        if run.get("audit_verdict") == "valid" and run.get("agent_model") not in {"es", "randopt"}
    }
    groups: dict[str, list[dict]] = defaultdict(list)
    labels = {variant: label for label, variant in CONFIG_SPECS}
    for run in runs:
        record = valid.get(run["runId"])
        if record and record.get("agent_variant") in labels and run.get("points"):
            run["submissionMinute"] = float(
                np.clip(
                    240.0 - record["submission_margin_seconds"] / 60.0,
                    run["points"][-1]["minute"],
                    240.0,
                )
            )
            groups[record["agent_variant"]].append(run)
    representatives: list[tuple[str, str, dict, int]] = []
    for label, variant in CONFIG_SPECS:
        family_runs = groups.get(variant, [])
        if not family_runs:
            continue
        endpoints = np.array([run["observedBest"] for run in family_runs])
        median = np.median(endpoints)
        chosen = min(family_runs, key=lambda run: abs(run["observedBest"] - median))
        representatives.append((label, variant, chosen, len(family_runs)))

    palette = [ORANGE, GREEN, "#59A14F", "#72A0C1", "#4E79A7", BLUE,
               RED, PURPLE, "#B279A2", GRAY, "#9C755F", "#76B7B2"]
    fig, axes = plt.subplots(4, 3, figsize=(7.15, 6.95), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.11, "hspace": 0.24})
    for index, (ax, (label, _, run, repeat_count), color) in enumerate(zip(axes.flat, representatives, palette)):
        points = run["points"]
        minute = np.array([point["minute"] for point in points])
        score = np.array([point["score"] * 100 for point in points])
        best = np.array([point["best"] * 100 for point in points])
        submission_minute = run["submissionMinute"]
        updates = int(np.sum(np.diff(best) > 1e-9))
        ax.scatter(minute, score, s=8, color="#B7C0CC", alpha=0.55, linewidths=0, rasterized=True)
        ax.step(
            np.append(minute, submission_minute),
            np.append(best, best[-1]),
            where="post",
            color=color,
            lw=1.55,
        )
        if submission_minute < 240.0:
            ax.plot(
                [submission_minute, 240.0],
                [best[-1], best[-1]],
                color=color,
                lw=0.9,
                ls=(0, (2, 2)),
                alpha=0.35,
            )
        ax.scatter(
            submission_minute,
            best[-1],
            s=23,
            marker="s",
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.axhline(44.08, color=INK, ls="--", lw=0.8)
        ax.set(xlim=(0, 240), ylim=(39.5, 50.0))
        ax.grid(True)
        letter = chr(ord("a") + index)
        ax.text(0.0, 1.02, letter, transform=ax.transAxes, weight="bold", fontsize=8.2)
        ax.set_title(label, x=0.085, ha="left", fontsize=7.2, pad=2)
        ax.text(0.97, 0.95, f"{repeat_count} runs · {run['evaluations']} evals · {best[-1]:.2f}",
                transform=ax.transAxes, ha="right", va="top", color=GRAY, fontsize=5.4)
    fig.supxlabel("Elapsed minutes", y=-0.005, fontsize=8)
    fig.supylabel("Seven-task score", x=0.01, fontsize=8)
    axes[-1, -1].plot([], [], color="#B7C0CC", marker="o", ls="", ms=3, label="evaluated candidate")
    axes[-1, -1].plot([], [], color=INK, lw=1.55, label="running best")
    handles, legend_labels = axes[-1, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=2,
        handlelength=1.4,
        fontsize=5.5,
    )

    save(fig, "fig5-trace-cases")


def render_agent_behavior() -> None:
    """Summarize long-horizon use and checkpoint composition."""
    inventory = json.loads(INVENTORY.read_text())["runs"]
    agent_runs = [
        run
        for run in inventory
        if run["agent_model"] not in {"es", "randopt"}
        and run.get("formal_status") == "valid"
        and run.get("attempts_succeeded", 0) > 0
    ]
    trajectory_runs = load_trajectory_runs()
    valid_run_ids = {run["run_id"] for run in agent_runs}
    t90 = []
    for run in trajectory_runs:
        if run["runId"] not in valid_run_ids or not run.get("points"):
            continue
        base = 44.07642857142857
        points = run["points"]
        final = max(point["best"] * 100 for point in points)
        if final <= base:
            continue
        threshold = base + 0.9 * (final - base)
        hit = next(point["minute"] for point in points if point["best"] * 100 >= threshold - 1e-9)
        t90.append(hit)
    t90 = np.sort(np.array(t90))

    best_one = np.array([run["best_one_term_score"] * 100 for run in agent_runs])
    best_multi = np.array([run["best_multi_term_score"] * 100 for run in agent_runs])
    composed_wins = best_multi > best_one
    median_advantage = float(np.median(best_multi - best_one))

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    fraction = np.arange(1, len(t90) + 1) / len(t90)
    ax.step(t90, fraction * 100, where="post", color=ORANGE, lw=1.9)
    ax.axvline(120, color=GRAY, ls="--", lw=0.8)
    ax.axvline(180, color=GRAY, ls="--", lw=0.8)
    ax.scatter(np.median(t90), 50, s=27, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.text(np.median(t90) + 5, 48, f"median {np.median(t90):.0f} min", color=ORANGE, fontsize=6.6)
    ax.set(xlabel="Minutes to 90% of final gain", ylabel="Agent runs reached (%)", xlim=(0, 240), ylim=(0, 104))
    ax.set_xticks([0, 60, 120, 180, 240])
    ax.grid(axis="y")
    panel_label(ax, "A")

    ax = axes[1]
    colors = np.where(composed_wins, GREEN, "#AAB3C0")
    for one, multi, color in zip(best_one, best_multi, colors):
        ax.plot([one, multi], [one, multi], alpha=0)
        ax.scatter(one, multi, s=25, color=color, edgecolor="white", linewidth=0.5)
    low = min(best_one.min(), best_multi.min()) - 0.4
    high = max(best_one.max(), best_multi.max()) + 0.4
    ax.plot([low, high], [low, high], color=INK, ls="--", lw=0.8)
    ax.set(xlabel="Best one-direction score", ylabel="Best composed score", xlim=(low, high), ylim=(low, high))
    ax.grid(True)
    ax.text(0.04, 0.96, f"composition wins in {composed_wins.sum()}/{len(agent_runs)} runs\nmedian difference {median_advantage:+.2f}",
            transform=ax.transAxes, va="top", color=GREEN, fontsize=6.5)
    panel_label(ax, "B")

    save(fig, "fig6-agent-behavior")


def main() -> None:
    render_main_result()
    render_search_space()
    render_weight_space_landscape()
    render_target_dependence()
    render_task_conflict()
    render_task_breakdown()
    render_trace_cases()
    render_agent_behavior()
    render_weight_space_walks()


if __name__ == "__main__":
    main()
