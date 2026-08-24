from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


TASKS = (
    "countdown",
    "gsm8k",
    "math500",
    "olympiadbench",
    "mbpp",
    "rocstories",
    "uspto50k",
)


def _path(value: str, *, label: str, must_exist: bool = True) -> Path:
    path = Path(os.path.expandvars(value)).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    base_model_path: Path
    data_root: Path
    randopt_source: Path
    runs_root: Path
    harness: str
    agent_model: str
    cli_path: Path
    credential_files: dict[str, Path]
    pass_environment: tuple[str, ...]
    harness_environment: dict[str, str]
    hours: float
    num_gpus: int
    samples_per_task: int
    tasks: tuple[str, ...]
    prompt_path: Path
    starter_path: Path

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        paths = raw["paths"]
        agent = raw["agent"]
        evaluation = raw["evaluation"]
        runtime = raw["runtime"]
        harness = str(agent["harness"])
        if harness not in {"codex", "cursor", "opencode"}:
            raise ValueError("agent.harness must be codex, cursor, or opencode")
        tasks = tuple(evaluation.get("tasks", TASKS))
        unknown = sorted(set(tasks) - set(TASKS))
        if not tasks or len(set(tasks)) != len(tasks) or unknown:
            raise ValueError(f"invalid evaluation.tasks; unknown={unknown}")
        hours = float(runtime["hours"])
        num_gpus = int(runtime["num_gpus"])
        samples = int(evaluation["samples_per_task"])
        if hours <= 0 or num_gpus <= 0 or samples <= 0:
            raise ValueError("hours, num_gpus, and samples_per_task must be positive")
        credentials = {
            destination: _path(source, label=f"credential {destination}")
            for destination, source in agent.get("credential_files", {}).items()
        }
        if any(Path(destination).is_absolute() or ".." in Path(destination).parts for destination in credentials):
            raise ValueError("credential destinations must be relative to the trusted control home")
        return cls(
            run_id=str(raw["run_id"]),
            base_model_path=_path(paths["base_model"], label="base model"),
            data_root=_path(paths["evaluation_data"], label="evaluation data"),
            randopt_source=_path(paths["randopt_source"], label="RandOPT source"),
            runs_root=_path(paths["runs_root"], label="runs root", must_exist=False),
            harness=harness,
            agent_model=str(agent["model"]),
            cli_path=_path(agent["cli_path"], label="agent CLI"),
            credential_files=credentials,
            pass_environment=tuple(agent.get("pass_environment", [])),
            harness_environment={str(k): str(v) for k, v in agent.get("environment", {}).items()},
            hours=hours,
            num_gpus=num_gpus,
            samples_per_task=samples,
            tasks=tasks,
            prompt_path=_path(paths["prompt"], label="prompt"),
            starter_path=_path(paths["starter"], label="starter directory"),
        )
