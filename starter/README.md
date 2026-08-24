# Agent-visible starter files

The runtime copies this directory into each fresh Agent workspace.

- `randopt.py` samples independent deterministic noise directions and keeps the
  best score it observes.
- `es.py` evaluates positive and negative probes, then adds score-weighted
  directions to a center candidate.
- `agent_client.py` is injected by the trusted runtime at initialization. It is
  the only bridge to the hidden evaluator.

Both methods are examples, not required algorithms. The Agent may edit or
replace them. Neither starter imports a gradient framework.
