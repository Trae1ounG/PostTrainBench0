# Starter methods

`randopt.py` and `es.py` are the two small starting methods shown to every
agent. They use only Python's standard library and the score-only
`agent_client` module mounted by the trusted benchmark runtime.

- `randopt.py` samples independent weight directions and records the best
  evaluated single candidate.
- `es.py` evaluates positive and negative perturbations, then updates a center
  from their score differences.

They are not complete training recipes and do not contain the evaluator. An
agent may modify or replace them during its run. Neither starter performs
backpropagation or imports a gradient framework.
