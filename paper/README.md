# Paper draft

This directory contains the first ICLR-style research-preview draft for
PostTrainBench⁰. It intentionally reports the current scores as visible
development results rather than a final leaderboard.

Regenerate the quantitative figures and build with:

```bash
python3 scripts/render_paper_figures.py
latexmk -pdf main.tex
```

The public repository contains `data/score_bearing_runs.csv` and its derived
`data/agent_run_summary.csv`. Raw trace records are retained in a private
research archive and are not part of the public release.
The comparison keeps the three highest complete seven-task runs per exact
Qwen2.5 setting and the two available same-protocol Qwen3 runs. Submission and
audit status remain separate from score-bearing completion. The plotting
scripts write local vector PDF figures and high-resolution PNG previews, but
compiled PDFs are not committed to the public repository. The paper and blog share
`../docs/assets/system-overview.png` as the benchmark pipeline figure.

The current draft reports visible-development experiments. A submission version
still requires the planned hidden-split and equal-evaluation-budget experiments.
