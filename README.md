# PostTrainBench⁰

**Can LLM agents automate LLM post-training without gradients?**

[Research blog](https://trae1oung.github.io/posttrainbench0/) ·
[Protocol](docs/PROTOCOL.md) ·
[Instruction template](prompt.txt)

PostTrainBench⁰ is a long-horizon benchmark for agent-driven, zeroth-order LLM
post-training. An Agent receives one frozen base checkpoint, a clean writable
workspace, two small starting methods, and a score-only evaluator. It may write
and revise any gradient-free search code during the time budget. The trusted
runtime retains one checkpoint: the highest-scoring candidate that completed
the full task suite before the deadline.

This `main` branch is the benchmark runtime. The bilingual research blog and
its interactive figures live on the [`website`](https://github.com/Trae1ounG/PostTrainBench0/tree/website) branch;
historical protocol iterations and raw experiment operations remain in the
separate private research archive.

## What the operator configures

Copy [`configs/example.json`](configs/example.json) and set:

- `paths.base_model`: the exact local model-weight directory;
- `paths.evaluation_data`: the trusted task-data directory;
- `paths.randopt_source`: a checkout of the RandOPT evaluator/data handlers;
- `paths.runs_root`: the one parent directory for all append-only episodes;
- `agent.harness`, `agent.model`, and `agent.cli_path`;
- credential files or environment-variable names, which stay in the trusted
  CLI control process and are not written into the Agent workspace;
- wall-clock time, GPU count, sample count, and task list.

The configuration is explicit by design. The launcher does not search the
machine for checkpoints, datasets, credentials, or spare output directories.

## What the Agent sees

```text
/home/agent/                    # the only writable root
├── prompt.txt                  # common rendered instruction
├── episode.json                # base-model path, tasks, budget, GPU count
├── timer.sh
├── bin/{evaluate,evaluate-batch,results,status}
└── starter/{agent_client.py,randopt.py,es.py}

/models/base/                   # full base checkpoint, read-only
```

It does **not** see task examples or labels, evaluator code, attempt records,
the CLI credential home, the best-candidate store, final outputs, or host
filesystem paths. Agent-issued commands run in a no-network Bubblewrap
namespace. See [the protocol](docs/PROTOCOL.md) for the exact mount table.

## Starter code and evaluation API

- [`starter/randopt.py`](starter/randopt.py) samples independent deterministic
  full-model noise directions.
- [`starter/es.py`](starter/es.py) uses antithetic probes and score differences
  to update a center candidate.

They are deliberately short and optional. The Agent may edit, combine, or
replace them. Candidate files describe `base + Σ scale × direction(seed)`;
actual model mutation and scoring happen only in the trusted evaluator.

```bash
bin/evaluate candidates/a.json countdown gsm8k
bin/evaluate-batch candidates/a.json candidates/b.json --tasks math500 mbpp
bin/results
bin/status
```

Every call creates a permanent attempt record. Only a completed full-suite
evaluation may become the retained result; no manual submit command is needed.

## Run one episode

The current reference runtime targets a Linux GPU Trial with Bubblewrap, Ray,
vLLM, and a prepared RandOPT checkout.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[gpu,test]'

# Edit paths and Agent settings first.
cp configs/example.json configs/local.json

# Safe preflight: creates the exact run tree but does not start a GPU or Agent.
posttrainbench0 --config configs/local.json --init-only

# Use a new run_id after the preflight; run IDs are never overwritten.
posttrainbench0 --config configs/local.json
```

Harness adapters are under [`agents/`](agents/). Codex, Cursor Agent, and
OpenCode receive the same prompt and filesystem contract; their scripts only
translate the shared inputs into native CLI arguments.

## Outputs

One run is stored under `RUNS_ROOT/RUN_ID/` with separate `workspace/`,
`attempts/`, `logs/`, `best/`, and `final/` directories. The final controller
step rebuilds the best candidate from the immutable base, materializes one
checkpoint, reloads it in a fresh one-GPU evaluator, and checks that every task
score is unchanged. `audit.json` separately records the trace-based no-gradient
and isolation verdict, so a numerical score is never confused with an accepted
benchmark run.

## Tests

```bash
python -m pytest -q
```

The tests cover configuration validation, append-only initialization,
agent-visible files, score-only evaluation, full-suite best retention, and the
absence of gradient frameworks from the two starter implementations.
