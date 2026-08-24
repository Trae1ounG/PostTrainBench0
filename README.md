<div align="center">

# PostTrainBench⁰

### Can LLM agents automate LLM post-training without gradients?

[Research blog](https://trae1oung.github.io/posttrainbench0/) ·
[Task instruction](prompt.txt) ·
[Full protocol](docs/PROTOCOL.md) ·
[Website source](https://github.com/Trae1ounG/PostTrainBench0/tree/website)

</div>

PostTrainBench⁰ is a long-horizon research task: give an LLM Agent a frozen
language model, a score-only evaluator, and a fixed GPU/time budget, then ask it
to produce **one checkpoint with a better average score across several tasks**.

The Agent may write and revise any search program, but it cannot use gradients,
change the evaluator, inspect answers, download another model, or ensemble
multiple checkpoints. This keeps the central post-training loop—form a
hypothesis, run an experiment, read feedback, and revise—without an explicit
training step.

![PostTrainBench⁰ system overview](docs/assets/system-overview.png)

## The benchmark in one minute

| | Contract |
|---|---|
| **Input** | One immutable base checkpoint, one clean workspace, the shared instruction, two optional starter methods, and score-only evaluation commands |
| **Agent's job** | Write a gradient-free search program, construct candidate checkpoints, decide what to evaluate, and use returned scores to plan the next experiment |
| **Objective** | Maximize the equal-weight mean over the configured task suite |
| **Budget** | One wall-clock window and a fixed number of GPUs; reasoning, code execution, and evaluation all count |
| **Output** | The highest-scoring single checkpoint that completed the full task suite before the deadline |
| **Acceptance** | Reload that checkpoint in a fresh evaluator, replay every task, and pass the trace/isolation audit |

Eight GPUs increase evaluation throughput; they never form an ensemble. A
score always belongs to one independently loadable checkpoint.

## Who controls what?

| Agent controls | Trusted runtime controls |
|---|---|
| Search code and experimental notes | Read-only base-model mount |
| Candidate definitions and combinations | Hidden task examples, labels, and scorers |
| Evaluation order and task subsets | Append-only attempt history |
| Serial or parallel candidate scheduling | Best full-suite checkpoint retention |
| Whether to use, modify, or replace the starters | Deadline, final materialization, fresh replay, and compliance audit |

Inside an episode, the Agent sees only:

- `/home/agent`: its writable root containing the rendered instruction,
  episode metadata, timer, public evaluation commands, and starter code;
- `/models/base`: the complete base checkpoint, mounted read-only.

The evaluator implementation, raw task data, trusted logs, credentials,
best-candidate store, final output, and host paths are not mounted. Agent-issued
commands execute in a no-network Bubblewrap namespace. The exact mount table is
documented in [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## What happens during one episode?

1. **Configure.** The operator names the exact base-model directory, evaluation
   data, run root, Agent harness, Agent model, budget, and GPU count.
2. **Isolate.** The runtime creates a new append-only `RUN_ID`, maps its clean
   workspace to `/home/agent`, and verifies the filesystem/network boundary.
3. **Search.** Codex, Cursor Agent, or OpenCode receives the same
   [`prompt.txt`](prompt.txt) and works autonomously until the deadline.
4. **Evaluate.** Every public evaluation call produces a permanent attempt
   record. Task-subset scores may guide search; only a complete suite can become
   the current best.
5. **Finalize.** The controller materializes the best candidate, reloads it in
   a fresh one-GPU evaluator, replays all tasks, and writes a separate audit
   verdict.

No `submit` command is required. The trusted controller continuously retains
the highest completed full-suite score, so an Agent can keep searching until
the time limit.

## Task suite

The reference evaluator uses 200 fixed examples per task, greedy decoding,
generation seed 42, and task-specific output limits.

| Task | Capability | Returned score |
|---|---|---|
| Countdown | Constrained arithmetic | Valid expression reaching the target |
| GSM8K | Math word problems | Normalized exact answer |
| MATH-500 | Competition mathematics | Normalized mathematical answer |
| OlympiadBench | Olympiad mathematics | Normalized final answer |
| MBPP | Code generation | Isolated test execution pass rate |
| ROCStories | Story ordering | Ordering score |
| USPTO-50K | Reaction classification | Exact class accuracy |

The joint score is the equal-weight mean of these task scores. The task list and
sample count are explicit in each run configuration.

## Configure a run

Start from [`configs/example.json`](configs/example.json). The important
fields are intentionally explicit:

```json
{
  "run_id": "qwen25-3b-kimi-k26-run01",
  "paths": {
    "base_model": "/models/Qwen2.5-3B-Instruct",
    "evaluation_data": "/datasets/posttrainbench0/visible200",
    "randopt_source": "/opt/RandOPT",
    "runs_root": "/workspace/posttrainbench0-runs",
    "prompt": "./prompt.txt",
    "starter": "./starter"
  },
  "agent": {
    "harness": "opencode",
    "model": "provider/model-name",
    "cli_path": "/opt/opencode/bin/opencode"
  },
  "evaluation": { "samples_per_task": 200 },
  "runtime": { "hours": 4, "num_gpus": 8 }
}
```

The launcher does not search the machine for a model, dataset, credential, or
output directory. Credentials stay in a separate trusted control home and are
never copied into the Agent workspace or public run manifest.

## Agent-facing interface

Every harness receives the same instruction and these four commands:

| Command | Purpose |
|---|---|
| `bin/evaluate CANDIDATE.json [TASK ...]` | Score one candidate on all tasks or a selected subset |
| `bin/evaluate-batch A.json B.json ... --tasks TASK ...` | Evaluate several candidates concurrently |
| `bin/results` | Read completed attempt results, including calls whose shell output was interrupted |
| `bin/status` | Read remaining time and the current best full-suite score |

A candidate is a compact deterministic program of the form
`base + Σ scale × direction(seed)`. The trusted evaluator reconstructs it from
the immutable base on every call, preventing accidental BF16 drift between
attempts.

Two short, optional starting points are included:

| File | Starting behavior |
|---|---|
| [`starter/randopt.py`](starter/randopt.py) | Sample independent deterministic full-model directions and retain the best score |
| [`starter/es.py`](starter/es.py) | Compare positive/negative direction pairs and update a stateful search center from score differences |

The Agent may use, modify, combine, or replace both files. Their purpose is to
provide working evaluator calls—not to prescribe the final algorithm.

## Run it

The reference runtime targets a Linux GPU Trial with Bubblewrap, Ray, vLLM,
and a prepared RandOPT checkout.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[gpu,test]'

cp configs/example.json configs/local.json
# Edit every path and Agent setting in configs/local.json.

# Creates and checks the isolated run tree without starting the Agent/GPU job.
posttrainbench0 --config configs/local.json --init-only

# Use a new run_id after preflight; run IDs are never overwritten.
posttrainbench0 --config configs/local.json
```

Harness adapters live in [`agents/`](agents/). They only translate the shared
prompt and filesystem contract into the native Codex, Cursor Agent, or OpenCode
CLI invocation.

## What a completed run contains

| Path | Contents |
|---|---|
| `run.json` | Immutable trusted run manifest |
| `workspace/` | Agent-authored code, notes, candidates, and public instruction |
| `attempts/` | Append-only record for every evaluation call |
| `best/` | History of improvements and current best full-suite candidate |
| `logs/` | Native Agent trace and controller events |
| `audit.json` | Separate no-gradient and isolation verdict |
| `final/checkpoint/` | One materialized checkpoint |
| `final/result.json` | Fresh-replay task scores and final joint score |

Raw score and compliance status are separate fields: a numerically strong run
is not accepted when its trace or isolation audit fails.

## Repository boundaries

| Location | Purpose |
|---|---|
| `main` | Lightweight executable benchmark runtime and protocol |
| [`website`](https://github.com/Trae1ounG/PostTrainBench0/tree/website) | Bilingual research blog, interactive traces, and publication figures |
| Private research archive | Historical protocol versions, raw traces, infrastructure logs, and internal experiment records |

## Tests

```bash
python -m pytest -q
```

The tests cover configuration validation, append-only initialization,
Agent-visible files, score-only evaluation, full-suite best retention, prompt
consistency, and the absence of gradient frameworks from the two starters.
