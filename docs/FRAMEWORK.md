# PostTrainBench⁰ framework boundary

PostTrainBench⁰ separates the agent workspace from the trusted evaluator.

## Agent-visible area

The agent sees:

- the shared task prompt;
- a writable temporary workspace;
- the pinned base-model reference;
- the ES and RandOpt starter implementations;
- commands for evaluating a candidate on any chosen task subset;
- the returned per-task and average scores;
- the remaining wall-clock time;
- automatic retention of the best candidate with a completed full-suite evaluation.

The benchmark does not prescribe the search algorithm. An agent may sample,
combine, rescale, or otherwise edit candidate weights as long as it remains
gradient-free. At timeout, the trusted controller materializes one best eligible
checkpoint.

## Trusted area

The agent cannot edit:

- evaluator code or task examples;
- the base-model source;
- attempt records and score records;
- the controller and final-selection broker;
- the post-run compliance auditor.

Evaluation is deterministic for a fixed checkpoint: the same 200 examples per
task, greedy decoding, fixed generation seed, and exact restoration of base
weights before constructing a new candidate.

## Persistent run record

Every evaluation attempt gets its own append-only record containing the
candidate definition, requested tasks, status, duration, and returned scores.
The final retained candidate is recorded separately. Full command and file-operation
traces are retained in the controlled workspace for the zeroth-order compliance audit; this
GitHub repository stores only de-identified summaries and derived figures.

## Current implementation stage

The working implementation supports native Codex and OpenCode CLI harnesses,
resident one-GPU evaluators, task-subset evaluation, a four-hour agent window,
one final checkpoint, and post-run trace auditing. Infrastructure-specific
launchers and credentials are deliberately outside this repository.
