from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any


def _score_one_answer(
    sender: Any,
    handler: Any,
    response: str,
    ground_truth: object,
) -> None:
    """Score one untrusted answer without sharing process state with the batch."""

    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            try:
                score = float(handler.compute_reward(response, ground_truth))
            except BaseException:
                score = 0.0
        sender.send(score)
    except BaseException:
        # The coordinator treats a missing child result as a wrong answer.
        pass
    finally:
        sender.close()


def _stop_process(process: Any) -> None:
    process.join(timeout=0.1)
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.2)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.2)


def score_responses_isolated(
    handler: Any,
    responses: list[str],
    ground_truths: list[object],
    *,
    answer_timeout_seconds: float = 10.0,
    max_workers: int = 16,
) -> list[float]:
    """Score answers in short-lived child processes with bounded concurrency."""

    if len(responses) != len(ground_truths):
        raise ValueError("responses and ground_truths must have the same length")
    if answer_timeout_seconds <= 0:
        raise ValueError("answer_timeout_seconds must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if not responses:
        return []

    try:
        context = multiprocessing.get_context("fork")
    except ValueError as exc:
        raise RuntimeError("MBPP scoring requires multiprocessing fork support") from exc

    scores = [0.0] * len(responses)
    next_index = 0
    active: dict[int, tuple[Any, Any, float]] = {}

    while next_index < len(responses) or active:
        while next_index < len(responses) and len(active) < max_workers:
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_score_one_answer,
                args=(sender, handler, responses[next_index], ground_truths[next_index]),
            )
            process.start()
            sender.close()
            active[next_index] = (
                process,
                receiver,
                time.monotonic() + answer_timeout_seconds,
            )
            next_index += 1

        progressed = False
        now = time.monotonic()
        for index, (process, receiver, deadline) in list(active.items()):
            score: float | None = None
            finished = False
            if receiver.poll():
                try:
                    score = float(receiver.recv())
                except (EOFError, OSError, TypeError, ValueError):
                    score = 0.0
                finished = True
            elif not process.is_alive() or now >= deadline:
                score = 0.0
                finished = True

            if not finished:
                continue
            scores[index] = score if score is not None else 0.0
            receiver.close()
            _stop_process(process)
            del active[index]
            progressed = True

        if not progressed and active:
            time.sleep(0.01)

    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--randopt-source", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    payload = json.load(sys.stdin)
    responses = payload.get("responses")
    ground_truths = payload.get("ground_truths")
    if not isinstance(responses, list) or not all(
        isinstance(response, str) for response in responses
    ):
        raise ValueError("responses must be a list of strings")
    if len(responses) != args.samples:
        raise ValueError(f"received {len(responses)} responses, expected {args.samples}")
    if not isinstance(ground_truths, list) or len(ground_truths) != args.samples:
        raise ValueError(f"received invalid ground truths; expected {args.samples}")

    sys.path.insert(0, str(args.randopt_source.resolve()))
    from data_handlers import get_dataset_handler

    handler = get_dataset_handler("mbpp")
    rewards = score_responses_isolated(handler, responses, ground_truths)
    score = sum(rewards) / len(rewards)
    print("ZEROGRAD_MBPP_RESULT=" + json.dumps({"score": score}, sort_keys=True))


if __name__ == "__main__":
    main()
