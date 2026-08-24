from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import time

from .config import RunConfig


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class EpisodeLayout:
    root: Path
    workspace: Path
    attempts: Path
    logs: Path
    best: Path
    final: Path


def _render_prompt(template: str, config: RunConfig) -> str:
    replacements = {
        "{base_model_path}": "/models/base",
        "{task_list}": "\n".join(f"- {task}" for task in config.tasks),
        "{num_hours}": f"{config.hours:g}",
        "{num_gpus}": str(config.num_gpus),
    }
    rendered = template
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    if re.search(r"\{[A-Za-z_][A-Za-z0-9_]*\}", rendered):
        raise ValueError("prompt contains an unresolved placeholder")
    return rendered.rstrip() + "\n"


def _write_commands(workspace: Path) -> None:
    commands = workspace / "bin"
    commands.mkdir()
    template = """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/starter/agent_client.py" OPERATION "$@"
"""
    for name, operation in (
        ("evaluate", "evaluate"),
        ("evaluate-batch", "evaluate-batch"),
        ("results", "results"),
        ("status", "status"),
    ):
        target = commands / name
        target.write_text(template.replace("OPERATION", operation), encoding="utf-8")
        target.chmod(0o755)


def initialize(config: RunConfig) -> EpisodeLayout:
    if not RUN_ID.fullmatch(config.run_id):
        raise ValueError("run_id may contain only letters, digits, '.', '_' and '-'")
    root = config.runs_root / config.run_id
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {root}")
    layout = EpisodeLayout(
        root=root,
        workspace=root / "workspace",
        attempts=root / "attempts",
        logs=root / "logs",
        best=root / "best",
        final=root / "final",
    )
    for directory in (layout.workspace, layout.attempts, layout.logs, layout.best, layout.final):
        directory.mkdir(parents=True, exist_ok=False)

    shutil.copytree(config.starter_path, layout.workspace / "starter")
    shutil.copy2(Path(__file__).with_name("agent_client.py"), layout.workspace / "starter" / "agent_client.py")
    _write_commands(layout.workspace)
    (layout.workspace / "prompt.txt").write_text(
        _render_prompt(config.prompt_path.read_text(encoding="utf-8"), config),
        encoding="utf-8",
    )
    started_at = int(time.time())
    deadline = started_at + round(config.hours * 3600)
    public = {
        "schema_version": 1,
        "run_id": config.run_id,
        "base_model": {"path": "/models/base", "read_only": True},
        "tasks": list(config.tasks),
        "joint_score": "equal_weight_mean",
        "harness": config.harness,
        "agent_model": config.agent_model,
        "num_gpus": config.num_gpus,
        "started_at_epoch": started_at,
        "deadline_epoch": deadline,
    }
    (layout.workspace / "episode.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    timer = layout.workspace / "timer.sh"
    timer.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "deadline=$(python3 -c 'import json; print(json.load(open(\"episode.json\"))[\"deadline_epoch\"])')\n"
        "remaining=$((deadline - $(date +%s)))\n"
        "(( remaining > 0 )) || { echo expired; exit 0; }\n"
        "printf '%02d:%02d:%02d remaining\\n' $((remaining/3600)) $(((remaining%3600)/60)) $((remaining%60))\n",
        encoding="utf-8",
    )
    timer.chmod(0o755)

    run = {
        "schema_version": 1,
        "run_id": config.run_id,
        "agent": {"model": config.agent_model, "harness": config.harness},
        "benchmark": {"tasks": list(config.tasks), "objective": "equal_weight_mean"},
        "base_model": {"source": str(config.base_model_path), "container_path": "/models/base", "read_only": True},
        "budget": {"hours": config.hours, "started_at_epoch": started_at, "deadline_epoch": deadline},
        "resources": {"gpus": config.num_gpus},
        "paths": {name: str(getattr(layout, name)) for name in ("workspace", "attempts", "logs", "best", "final")},
        "mounts": [
            {"source": str(layout.workspace), "target": "/home/agent", "mode": "rw"},
            {"source": str(config.base_model_path), "target": "/models/base", "mode": "ro"},
        ],
        "trusted_only": [str(layout.attempts), str(layout.logs), str(layout.best), str(layout.final), str(config.data_root)],
    }
    (layout.root / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (layout.logs / "agent_events.jsonl").touch(mode=0o600)
    return layout
