# PostTrainBench⁰ protocol

## One run

The operator provides a JSON config containing four concrete locations:

1. the immutable base-model directory;
2. the hidden seven-task data directory;
3. an empty parent directory for append-only runs;
4. the Agent CLI executable and its trusted credentials.

The runtime creates `RUNS_ROOT/RUN_ID` exactly once. Existing run IDs are never
overwritten.

```text
RUN_ID/
├── run.json                    trusted immutable run manifest
├── isolation.json              result of the pre-run isolation probe
├── workspace/                  the only Agent-writable root
│   ├── prompt.txt              rendered common instruction
│   ├── episode.json            public model, task, GPU and deadline metadata
│   ├── timer.sh
│   ├── bin/
│   │   ├── evaluate
│   │   ├── evaluate-batch
│   │   ├── results
│   │   └── status
│   └── starter/
│       ├── agent_client.py
│       ├── randopt.py
│       └── es.py
├── attempts/                   trusted append-only evaluation records
├── best/                       trusted best-full-suite pointer and history
├── logs/                       native Agent trace and controller events
├── control-home/               trusted CLI session and credentials
├── audit.json                  trace and workspace compliance report
└── final/                      one materialized and replayed checkpoint
```

## Visibility boundary

The networked Agent CLI runs in a filtered filesystem view. It can see the
writable workspace and the full base checkpoint mounted read-only at
`/models/base`. When the CLI asks to execute a command, the command is brokered
into a second Bubblewrap namespace with no network access. The evaluator is a
Unix-socket service: the Agent can call the four public commands but cannot read
the evaluator process, data, attempt store, or final checkpoint directory.

| Surface | Agent access | Purpose |
|---|---|---|
| `/home/agent` | read/write | code, candidate manifests, notes and temporary files |
| `/models/base` | read-only | exact starting checkpoint |
| `bin/evaluate*` | execute | score one or several candidates; raw examples stay hidden |
| `bin/results`, `bin/status` | execute | completed feedback and remaining time |
| evaluation data and scorer | none | trusted task execution |
| `attempts/`, `best/`, `final/`, `logs/` | none | append-only records and final selection |

The isolation probe must prove that the workspace is the only writable mount,
the base model is read-only, the evaluator files are invisible, and general
network access is blocked before the Agent window begins.

## Candidate and final selection

A candidate is a compact deterministic program:

```json
{
  "format": "zerograd-noise-program-v1",
  "candidate_id": "candidate-017",
  "terms": [
    {"seed": 123, "scale": 0.001},
    {"seed": 987, "scale": -0.0004}
  ]
}
```

Every evaluation reconstructs the candidate from the immutable base anchor;
candidate deltas never accumulate by accident. A task-subset evaluation can
guide search but is not final-eligible. Whenever a candidate completes all
configured tasks, the controller compares its equal-weight mean with the
current best and writes a new immutable best-history entry when it improves.
At the end of the run, the controller materializes that candidate once, starts
a fresh one-GPU evaluator, reloads the checkpoint, and requires exactly equal
task scores.

The post-run auditor scans the native tool trace and all Agent-authored text
files for gradient APIs, evaluator discovery, host-path access, external
network commands, model replacement, and writes to the read-only base mount.
It reports `valid`, `review_required`, or `invalid`; raw score and compliance
status remain separate fields.

## Task suite

The current evaluator uses 200 fixed examples per task, greedy decoding,
generation seed 42, and task-specific output limits from the RandOPT handlers.

| Task | Type | Returned score |
|---|---|---|
| Countdown | constrained arithmetic | valid expression reaching the target |
| GSM8K | math word problems | normalized exact answer |
| MATH-500 | competition mathematics | normalized mathematical answer |
| OlympiadBench | olympiad mathematics | normalized final answer |
| MBPP | code generation | isolated test execution pass rate |
| ROCStories | story ordering | ordering score |
| USPTO-50K | reaction classification | exact class accuracy |
