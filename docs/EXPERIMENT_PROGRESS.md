# Experiment progress

Snapshot date: 2026-08-24.

## Current project status

The four-hour, seven-task ES/RandOPT protocol is implemented end to end. An
agent can read the shared prompt and starters, create candidate checkpoints,
request score-only evaluations, inspect attempt history, and leave one final
checkpoint. The trusted runtime can reload that checkpoint in a fresh process
and audit the trace for gradient use, evaluator access, model substitution, and
ensembling.

This is a benchmark proposal under validation, not a finished leaderboard.

## Premise experiment

The controlled Qwen3-4B study sampled 500 nearby candidates.

- Base seven-task score: 41.08.
- Candidates above the base: 62 of 500 (12.4%).
- Best observed single checkpoint: 46.03.
- No sampled checkpoint improved all seven tasks.

The result shows that the local search problem is non-empty, but it also shows
why finite sampling matters: useful candidates exist without being uniformly
easy to encounter, and task-wise improvements do not point in one direction.

## Four-hour agent experiments

The current blog snapshot contains 51 complete runs:

- 34 runs on Qwen2.5-3B-Instruct, base score 44.08.
- 17 runs on Qwen3-4B-Base, base score 41.08.

The grouped values are stored in
[`data/derived/blog_run_summary.csv`](../data/derived/blog_run_summary.csv).
Several agents find improvements, but the ordering is not stable. The clearest
counterexample is GPT-5.6 xhigh on Qwen3-4B: two runs reach 43.29 and 53.19, a
9.90-point span under the same nominal agent setting.

## What is verified

- Fixed-checkpoint evaluation can replay exactly after a fresh reload.
- The tested target-model neighborhoods contain better joint checkpoints.
- Agents produce inspectably different search code and trajectories.
- Search outcomes can vary substantially between nominally repeated runs.

## What remains unresolved

- Candidate counts are not yet matched across agents.
- Most agent-target pairs have too few repeats for a reliable ranking.
- Search and final scoring use the same queryable 200-example task views.
- A best-of-run score mixes algorithm quality, system throughput, and luck.
- Parameter projections are descriptive and cannot establish high-dimensional
  cluster structure by themselves.

The next controlled protocol should match complete-evaluation budgets, add a
one-shot hidden test suite, repeat every pair, and confirm incumbents through
local-neighborhood tests or strategy replay. See
[`RANDOMNESS_AND_VALIDITY.md`](RANDOMNESS_AND_VALIDITY.md) for the reasoning.
