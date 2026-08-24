#!/usr/bin/env python3
"""Antithetic evolution-strategy starter without gradient computation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from agent_client import call


def write_candidate(path: Path, candidate_id: str, terms: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {"format": "zerograd-noise-program-v1", "candidate_id": candidate_id, "terms": terms},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", action="append", default=[])
    args = parser.parse_args()
    if args.iterations <= 0 or args.pairs <= 0 or args.sigma <= 0:
        raise ValueError("iterations, pairs, and sigma must be positive")

    root = Path("candidates/es")
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    center_terms: list[dict] = []
    best = None
    history = []

    for iteration in range(args.iterations):
        pair_records = []
        paths = []
        for pair in range(args.pairs):
            direction_seed = rng.randrange(2**31)
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                candidate_id = f"es-{iteration:03d}-{pair:03d}-{label}"
                path = root / f"{candidate_id}.json"
                write_candidate(
                    path,
                    candidate_id,
                    [*center_terms, {"seed": direction_seed, "scale": sign * args.sigma}],
                )
                paths.append(str(path))
            pair_records.append((direction_seed, paths[-2], paths[-1]))

        response = call(
            {"operation": "evaluate_batch", "candidates": paths, "tasks": args.task}
        )
        by_path = {item["candidate"]: item for item in response["results"]}
        history.extend(response["results"])
        for item in response["results"]:
            if best is None or item["average_score"] > best["average_score"]:
                best = item
        denominator = 2.0 * len(pair_records) * args.sigma
        for direction_seed, plus_path, minus_path in pair_records:
            coefficient = (
                args.learning_rate
                * (by_path[plus_path]["average_score"] - by_path[minus_path]["average_score"])
                / denominator
            )
            center_terms.append({"seed": direction_seed, "scale": coefficient})

        center_path = root / f"center-{iteration:03d}.json"
        write_candidate(center_path, f"es-center-{iteration:03d}", center_terms)
        center_result = call(
            {"operation": "evaluate", "candidate": str(center_path), "tasks": args.task}
        )
        center_item = {
            "candidate": str(center_path),
            "average_score": center_result["average_score"],
            "task_scores": center_result["task_scores"],
        }
        history.append(center_item)
        if best is None or center_item["average_score"] > best["average_score"]:
            best = center_item

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
