# Randomness and validity

PostTrainBench⁰ uses deterministic evaluation for a fixed checkpoint. The same
checkpoint is loaded from scratch, evaluated on the same 200 examples per task,
and decoded greedily with a fixed generation seed. If this replay changes the
score, the run has an implementation or restoration problem and should not be
interpreted as benchmark variance.

The observed instability instead comes from the search process.

## Five sources of variation

### 1. Direction sampling

A random seed identifies a concrete high-dimensional weight direction. Useful
directions are not uniformly distributed, so two finite samples can contain
very different opportunities even when both agents use the same code.

### 2. Scale and composition

An agent chooses how far to move along a direction and whether to compose it
with previous directions. A direction that helps at one scale can be destructive
at another. Composition also makes later candidates depend on earlier choices.

### 3. Incumbent path dependence

Once an early candidate becomes the incumbent, a local method may spend the
rest of the run around it. A chance early improvement can therefore be amplified
into a different trajectory rather than averaged away.

### 4. Candidate count

The reported endpoint is the best complete score observed in a run. A maximum
is an extreme statistic: evaluating more candidates raises the expected maximum
even if candidate quality is unchanged. API latency, code efficiency, batching,
and task-subset screening therefore affect the score through throughput.

### 5. Development-view adaptation

The agent repeatedly observes scores from the same 200-example task views. Its
final candidate can adapt to those examples through selection even without
gradients. This is search-time overfitting and is distinct from hidden-test
generalization.

## What current results can support

- The end-to-end task is executable and auditable.
- Better nearby checkpoints exist for the tested target models.
- Agents write materially different search code and follow different paths.
- A selected checkpoint can be reloaded and replayed exactly.

## What current results cannot support

- A stable ordering of frontier agents from one best-of-run score.
- A claim that a higher endpoint came from better reasoning rather than more
  candidate evaluations or a luckier direction pool.
- A claim that gains on the queryable 200-example views transfer to hidden data.
- A claim that two-dimensional projections reveal stable clusters in the full
  parameter space.

## Conditions for a stronger benchmark

The next protocol should match full-evaluation counts, repeat each
agent-target pair, separate a queryable development suite from a one-shot hidden
test suite, and report endpoint quality, search efficiency, repeat stability,
and trace behavior separately. Incumbents should also be confirmed by local
neighborhood tests or by replaying the agent's learned search policy.

Until these controls are in place, PostTrainBench⁰ should be presented as a
benchmark proposal and validity study, not as a final leaderboard.
