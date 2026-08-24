from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Mapping


_ATTEMPT_DIR = re.compile(r"^attempt-(\d{4})$")


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    path: Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _next_attempt_number(attempts_root: Path) -> int:
    numbers = []
    for path in attempts_root.iterdir():
        match = _ATTEMPT_DIR.fullmatch(path.name)
        if match and path.is_dir():
            numbers.append(int(match.group(1)))
    return (max(numbers) + 1) if numbers else 1


def create_attempt(
    *,
    run_root: Path,
    tasks: tuple[str, ...],
    candidate_path: str,
    note: str = "",
    created_at: str | None = None,
) -> AttemptRecord:
    """Create one structured evaluation attempt without touching older attempts."""

    run = _read_json(run_root / "run.json")
    configured_tasks = tuple(run["benchmark"]["tasks"])
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("attempt tasks must be non-empty and unique")
    unknown_tasks = sorted(set(tasks) - set(configured_tasks))
    if unknown_tasks:
        raise ValueError(f"tasks are not part of this run: {unknown_tasks}")
    candidate = Path(candidate_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("candidate_path must be relative to the agent workspace")

    attempts_root = run_root / "attempts"
    attempt_number = _next_attempt_number(attempts_root)
    while True:
        attempt_id = f"attempt-{attempt_number:04d}"
        attempt_root = attempts_root / attempt_id
        try:
            attempt_root.mkdir()
            break
        except FileExistsError:
            attempt_number += 1
    (attempt_root / "logs").mkdir()
    (attempt_root / "artifacts").mkdir()

    attempt = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "run_id": run["run_id"],
        "agent": run["agent"],
        "resources": run["resources"],
        "base_model": run["base_model"],
        "tasks": list(tasks),
        "candidate_path": candidate.as_posix(),
        "note": note,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    (attempt_root / "attempt.json").write_text(
        json.dumps(attempt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return AttemptRecord(attempt_id=attempt_id, path=attempt_root)


def record_attempt_result(
    *,
    attempt_root: Path,
    status: str,
    task_scores: Mapping[str, float] | None = None,
    elapsed_seconds: float,
    error: str | None = None,
    finished_at: str | None = None,
) -> Path:
    """Write one result file; a completed attempt is never overwritten."""

    result_path = attempt_root / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    if status not in {"succeeded", "failed", "rejected"}:
        raise ValueError("status must be succeeded, failed, or rejected")
    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be non-negative")

    attempt = _read_json(attempt_root / "attempt.json")
    scores = dict(task_scores or {})
    if status == "succeeded":
        if set(scores) != set(attempt["tasks"]):
            raise ValueError("a successful result must contain every requested task")
        average_score = sum(scores.values()) / len(scores)
    else:
        average_score = None

    result = {
        "schema_version": 1,
        "attempt_id": attempt["attempt_id"],
        "run_id": attempt["run_id"],
        "agent": attempt["agent"],
        "tasks": attempt["tasks"],
        "status": status,
        "task_scores": scores,
        "average_score": average_score,
        "elapsed_seconds": elapsed_seconds,
        "error": error,
        "finished_at": finished_at or datetime.now(timezone.utc).isoformat(),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result_path


def mark_incomplete_attempts_interrupted(run_root: Path) -> list[str]:
    """Close attempts that lost their evaluator process before committing a result."""

    interrupted: list[str] = []
    for attempt_root in sorted((run_root / "attempts").glob("attempt-*")):
        if not (attempt_root / "attempt.json").is_file():
            continue
        if (attempt_root / "result.json").exists():
            continue
        record_attempt_result(
            attempt_root=attempt_root,
            status="failed",
            elapsed_seconds=0.0,
            error="Interrupted before the evaluator committed a result; not reused.",
        )
        interrupted.append(attempt_root.name)
    return interrupted
