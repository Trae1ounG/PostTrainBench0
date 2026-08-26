# Paper draft

This directory contains the first ICLR-style research-preview draft for
PostTrainBench⁰. It intentionally reports the current scores as visible
development results rather than a final leaderboard.

Regenerate the quantitative figures and build with:

```bash
python3 scripts/render_paper_figures.py
latexmk -pdf main.tex
```

The plotting script reads the immutable files in `data/traces/` plus
`data/score_bearing_runs.csv` and its derived `data/agent_run_summary.csv`.
The comparison keeps the three highest complete seven-task runs per exact
Qwen2.5 setting and the two available same-protocol Qwen3 runs. Submission and
audit status remain separate from score-bearing completion. The script writes vector PDF figures for the paper and
high-resolution PNG previews from the same source. Trace inputs are retained
separately from derived figures so later analyses can be regenerated without
overwriting the original exports. The paper and blog share
`../docs/assets/system-overview.png` as the benchmark pipeline figure.

The current draft reports visible-development experiments. A submission version
still requires the planned hidden-split and equal-evaluation-budget experiments.
