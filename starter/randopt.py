#!/usr/bin/env python3
"""Readable random-search starter using only evaluator scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from agent_client import call


def write_candidate(path: Path, candidate_id: str, seed: int, scale: float) -> None:
    path.write_text(
        json.dumps(
            {
                "format": "zerograd-noise-program-v1",
                "candidate_id": candidate_id,
                "terms": [{"seed": seed, "scale": scale}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, action="append")
    parser.add_argument("--task", action="append", default=[])
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")

    scales = args.scale or [0.0005, 0.001, 0.002]
    root = Path("candidates/randopt")
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    paths = []
    for index in range(args.samples):
        path = root / f"candidate-{index:04d}.json"
        write_candidate(path, f"randopt-{index:04d}", rng.randrange(2**31), rng.choice(scales))
        paths.append(str(path))

    best = None
    history = []
    for offset in range(0, len(paths), args.batch_size):
        batch = paths[offset : offset + args.batch_size]
        response = call(
            {"operation": "evaluate_batch", "candidates": batch, "tasks": args.task}
        )
        history.extend(response["results"])
        for result in response["results"]:
            if best is None or result["average_score"] > best["average_score"]:
                best = result

    assert best is not None
    (root / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "best.json").write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
