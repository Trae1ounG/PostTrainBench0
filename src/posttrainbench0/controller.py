from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socketserver
import threading
import time
from typing import Mapping, Protocol

from .attempts import create_attempt, record_attempt_result
from .candidate_io import LoadedCandidate, load_candidate


class TrustedEvaluator(Protocol):
    def evaluate_batch(self, candidates: list[LoadedCandidate], tasks: tuple[str, ...]) -> list[Mapping[str, float]]: ...


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class EpisodeController:
    def __init__(self, run_root: Path, evaluator: TrustedEvaluator) -> None:
        self.root = run_root
        self.run = _read(run_root / "run.json")
        self.workspace = Path(self.run["paths"]["workspace"])
        self.tasks = tuple(self.run["benchmark"]["tasks"])
        self.deadline = int(self.run["budget"]["deadline_epoch"])
        self.evaluator = evaluator
        self._evaluation_lock = threading.Lock()
        self._best_lock = threading.Lock()
        self.events = Path(self.run["paths"]["logs"]) / "agent_events.jsonl"

    def _remaining(self) -> int:
        return max(0, self.deadline - int(time.time()))

    def _select_tasks(self, requested: object) -> tuple[str, ...]:
        if requested in (None, []):
            return self.tasks
        if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
            raise ValueError("tasks must be a list of task names")
        tasks = tuple(requested)
        unknown = sorted(set(tasks) - set(self.tasks))
        if not tasks or len(set(tasks)) != len(tasks) or unknown:
            raise ValueError(f"invalid tasks; unknown={unknown}")
        return tasks

    def _event(self, value: dict) -> None:
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **value}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _retain_if_best(self, candidate: LoadedCandidate, attempt_id: str, scores: dict[str, float]) -> bool:
        if set(scores) != set(self.tasks):
            return False
        average = sum(scores.values()) / len(scores)
        best_root = Path(self.run["paths"]["best"])
        current_path = best_root / "current.json"
        with self._best_lock:
            current = _read(current_path) if current_path.is_file() else None
            if current is not None and average <= float(current["average_score"]):
                return False
            history_root = best_root / "history" / attempt_id
            history_root.mkdir(parents=True)
            shutil.copy2(candidate.source_path, history_root / "candidate.json")
            record = {
                "attempt_id": attempt_id,
                "candidate_id": candidate.candidate.candidate_id,
                "average_score": average,
                "task_scores": scores,
            }
            (history_root / "best.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary = current_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(current_path)
            candidate_target = best_root / "candidate.json"
            temporary_candidate = candidate_target.with_suffix(".tmp")
            shutil.copy2(candidate.source_path, temporary_candidate)
            temporary_candidate.replace(candidate_target)
        self._event({"event": "best_updated", **record})
        return True

    def _evaluate(self, candidate_paths: list[str], tasks: tuple[str, ...]) -> dict:
        if self._remaining() <= 0:
            raise RuntimeError("episode time budget has expired")
        attempts = [
            create_attempt(run_root=self.root, tasks=tasks, candidate_path=path, note="score-only evaluation")
            for path in candidate_paths
        ]
        candidates = [load_candidate(self.workspace, path) for path in candidate_paths]
        for candidate, attempt in zip(candidates, attempts):
            shutil.copy2(candidate.source_path, attempt.path / "artifacts" / "candidate.json")
        started = time.monotonic()
        try:
            evaluated = self.evaluator.evaluate_batch(candidates, tasks)
            if len(evaluated) != len(candidates):
                raise RuntimeError("evaluator returned the wrong number of results")
            elapsed = time.monotonic() - started
            results = []
            for path, candidate, attempt, raw_scores in zip(candidate_paths, candidates, attempts, evaluated):
                scores = {task: float(raw_scores[task]) for task in tasks}
                result_path = record_attempt_result(
                    attempt_root=attempt.path,
                    status="succeeded",
                    task_scores=scores,
                    elapsed_seconds=elapsed,
                )
                result = _read(result_path)
                retained = self._retain_if_best(candidate, attempt.attempt_id, scores)
                item = {
                    "attempt_id": attempt.attempt_id,
                    "candidate": path,
                    "task_scores": scores,
                    "average_score": result["average_score"],
                    "retained_as_best": retained,
                }
                results.append(item)
                self._event({"event": "evaluation", "status": "succeeded", **item})
            return {"ok": True, "results": results, "elapsed_seconds": elapsed, "remaining_seconds": self._remaining()}
        except Exception as exc:
            elapsed = time.monotonic() - started
            for attempt in attempts:
                if not (attempt.path / "result.json").exists():
                    record_attempt_result(
                        attempt_root=attempt.path,
                        status="failed",
                        elapsed_seconds=elapsed,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            self._event({"event": "evaluation", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            raise

    def _results(self) -> dict:
        rows = []
        for root in sorted(Path(self.run["paths"]["attempts"]).glob("attempt-*")):
            if not (root / "attempt.json").is_file():
                continue
            attempt = _read(root / "attempt.json")
            result = _read(root / "result.json") if (root / "result.json").is_file() else {"status": "running"}
            rows.append({
                "attempt_id": attempt["attempt_id"],
                "candidate": attempt["candidate_path"],
                "tasks": attempt["tasks"],
                "status": result["status"],
                "task_scores": result.get("task_scores", {}),
                "average_score": result.get("average_score"),
            })
        best = _read(Path(self.run["paths"]["best"]) / "current.json") if (Path(self.run["paths"]["best"]) / "current.json").is_file() else None
        return {"ok": True, "attempts": rows, "best_full_suite": best, "remaining_seconds": self._remaining()}

    def handle(self, request: dict) -> dict:
        operation = request.get("operation")
        if operation == "status":
            return {"ok": True, "run_id": self.run["run_id"], "tasks": list(self.tasks), "remaining_seconds": self._remaining(), "best_full_suite": self._results()["best_full_suite"]}
        if operation == "results":
            return self._results()
        with self._evaluation_lock:
            if operation == "evaluate":
                response = self._evaluate([str(request["candidate"])], self._select_tasks(request.get("tasks")))
                return {"ok": True, **response["results"][0], "elapsed_seconds": response["elapsed_seconds"], "remaining_seconds": response["remaining_seconds"]}
            if operation == "evaluate_batch":
                paths = request.get("candidates")
                if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
                    raise ValueError("candidates must be a non-empty list")
                max_batch = int(self.run["resources"]["gpus"])
                if len(paths) > max_batch:
                    raise ValueError(f"at most {max_batch} candidates may be evaluated concurrently")
                return self._evaluate(paths, self._select_tasks(request.get("tasks")))
        raise ValueError("unknown operation")


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class UnixControllerServer:
    def __init__(self, socket_path: Path, controller: EpisodeController) -> None:
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        socket_path.unlink(missing_ok=True)

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    response = controller.handle(json.loads(self.rfile.readline()))
                except Exception as exc:
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write(json.dumps(response).encode() + b"\n")

        self.socket_path = socket_path
        self.server = _ThreadingUnixServer(str(socket_path), Handler)
        socket_path.chmod(0o600)

    def serve_forever(self) -> None:
        self.server.serve_forever()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.socket_path.unlink(missing_ok=True)
