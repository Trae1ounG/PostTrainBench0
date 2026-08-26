from pathlib import Path

import matplotlib.pyplot as plt


QWEN25 = [
    ("GPT-5.5 high", [49.2307, 48.3879]),
    ("Kimi K2.6", [49.25, 47.5143]),
    ("GPT-5.5 xhigh", [48.1657, 48.125]),
    ("GPT-5.6 xhigh", [48.6443, 47.3207]),
    ("GPT-5.6 high", [47.2793, 46.9757]),
    ("GPT-5.4 Pro", [46.8271, 45.8436]),
    ("GPT-5.6 medium", [48.8271, 47.6529]),
    ("DeepSeek V4 Pro", [47.1107, 46.945]),
    ("Opus 4.8 high", [47.1329, 46.0964]),
    ("Sonnet 4.6 medium", [47.9164, 45.7171]),
    ("GLM-5.1", [46.19, 45.8857]),
    ("MiniMax M2.7", [47.1207, 45.7457]),
    ("Qwen3.7-Max", [46.1543, 46.08]),
]

QWEN3 = [
    ("MiniMax M2.7", [48.9357, 49.0679]),
    ("Kimi K2.6", [51.9436, 45.8807]),
    ("GPT-5.4 Pro", [49.8921, 47.1979]),
    ("DeepSeek V4 Pro", [46.2893, 50.2079]),
    ("GPT-5.6 xhigh", [43.29, 53.19]),
    ("GLM-5.1", [46.6714, 44.7421]),
    ("GPT-5.5 xhigh", [47.4157, 42.9629]),
    ("Qwen3.7-Max", [44.6236, 44.38]),
]


def panel(ax, rows, title, base, limits):
    positions = list(range(len(rows)))
    for y, (name, scores) in zip(positions, rows):
        average = sum(scores) / len(scores)
        ax.hlines(y, min(scores), max(scores), color="#cbd5e1", linewidth=2.4, zorder=1)
        ax.scatter(scores, [y] * len(scores), s=36, color="#2563eb", edgecolor="white", linewidth=0.6, zorder=3)
        ax.vlines(average, y - 0.20, y + 0.20, color="#111827", linewidth=2, zorder=4)
        ax.text(limits[1] + 0.18, y, f"{average:.2f}", va="center", fontsize=8.5, color="#475569")
    ax.axvline(base, color="#475569", linewidth=1.25, linestyle="--")
    ax.set_yticks(positions, [name for name, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(*limits)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Best complete seven-task score in one run (0–100)")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(14.4, 8.1), gridspec_kw={"wspace": 0.42})
    panel(axes[0], QWEN25, "Qwen2.5-3B-Instruct", 44.0764, (43.5, 50.0))
    panel(axes[1], QWEN3, "Qwen3-4B-Base", 41.0821, (40.5, 54.0))
    figure.suptitle("Top two score-bearing runs per Agent setting", fontsize=18, fontweight="bold", y=0.985)
    figure.text(0.5, 0.945, "Runs without a final submission count by their best complete seven-task score.", ha="center", color="#64748b", fontsize=10)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#2563eb", label="One retained run"),
        plt.Line2D([], [], marker="|", markersize=14, linestyle="", color="#111827", label="Mean of retained runs"),
        plt.Line2D([], [], linestyle="--", color="#475569", label="Base checkpoint"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    figure.subplots_adjust(top=0.88, bottom=0.12, left=0.13, right=0.94)
    output = Path(__file__).resolve().parents[1] / "public" / "figures" / "historical-agent-runs"
    figure.savefig(output.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
