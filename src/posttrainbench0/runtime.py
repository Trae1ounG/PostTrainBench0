from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

from .candidate_io import load_candidate
from .audit import audit_episode
from .config import RunConfig
from .controller import EpisodeController, UnixControllerServer
from .episode import EpisodeLayout
from .evaluators.randopt_vllm import RandOptVllmBackend
from .isolation import CommandBroker, prepare_agent_shell, prepare_harness_launcher


def _read_jsonl(path: Path) -> list[object]:
    rows: list[object] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"type": "unparsed", "line_number": line_number, "raw": line})
    return rows


def _build_backend(config: RunConfig, *, num_engines: int | None = None) -> RandOptVllmBackend:
    return RandOptVllmBackend(
        model_path=config.base_model_path,
        data_root=config.data_root,
        randopt_source=config.randopt_source,
        samples_per_task=config.samples_per_task,
        num_engines=config.num_gpus if num_engines is None else num_engines,
        global_seed=42,
    )


def _prepare_control_home(config: RunConfig, layout: EpisodeLayout) -> Path:
    control_home = layout.root / "control-home"
    control_home.mkdir(mode=0o700)
    for directory in ("config", "data", "state", "cache"):
        (control_home / directory).mkdir(mode=0o700)
    for destination, source in config.credential_files.items():
        target = control_home / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o600)
    solve_source = Path(__file__).resolve().parents[2] / "agents" / config.harness / "solve.sh"
    if not solve_source.is_file():
        raise FileNotFoundError(solve_source)
    solve_target = control_home / "solve.sh"
    shutil.copy2(solve_source, solve_target)
    solve_target.chmod(0o700)
    return control_home


def _launch_agent(config: RunConfig, layout: EpisodeLayout, shell: Path, control_dir: Path) -> dict:
    control_home = _prepare_control_home(config, layout)
    harness_launcher = prepare_harness_launcher(
        run_root=layout.root,
        workspace=layout.workspace,
        base_model=config.base_model_path,
        control_dir=control_dir,
        control_home=control_home,
        cli_path=config.cli_path,
        harness=config.harness,
    )
    trace = layout.logs / f"{config.harness}.trace.jsonl"
    stderr_path = layout.logs / f"{config.harness}.stderr.log"
    environment = {
        "HOME": str(control_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PTB0_AGENT_MODEL": config.agent_model,
        **config.harness_environment,
    }
    for name in config.pass_environment:
        if name not in os.environ:
            raise RuntimeError(f"requested environment variable is not set: {name}")
        environment[name] = os.environ[name]
    command = [str(harness_launcher), f"/run/{config.harness}/solve.sh"]
    broker_path = control_dir / "command-broker.sock"
    with trace.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr, CommandBroker(broker_path, shell):
        process = subprocess.Popen(command, cwd=layout.workspace, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        deadline = int(json.loads((layout.workspace / "episode.json").read_text())["deadline_epoch"])
        timed_out = False
        while process.poll() is None:
            if time.time() >= deadline:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                break
            time.sleep(0.25)
        try:
            return_code = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return_code = process.wait(timeout=5)
    return {
        "harness": config.harness,
        "agent_model": config.agent_model,
        "return_code": return_code,
        "timed_out": timed_out,
        "trajectory": _read_jsonl(trace),
    }


def run(config: RunConfig, layout: EpisodeLayout) -> dict:
    control_dir = Path("/tmp") / f"ptb0-{config.run_id}"
    control_dir.mkdir(mode=0o700)
    socket_path = control_dir / "evaluator.sock"
    isolation = prepare_agent_shell(
        run_root=layout.root,
        workspace=layout.workspace,
        base_model=config.base_model_path,
        control_dir=control_dir,
    )
    backend = _build_backend(config)
    server: UnixControllerServer | None = None
    server_thread: threading.Thread | None = None
    try:
        backend.start()
        controller = EpisodeController(layout.root, backend)
        server = UnixControllerServer(socket_path, controller)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        agent_result = _launch_agent(config, layout, isolation.shell, control_dir)
        (layout.logs / "agent_result.json").write_text(json.dumps(agent_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        if server is not None:
            server.close()
        if server_thread is not None:
            server_thread.join(timeout=5)
        backend.close()
        shutil.rmtree(control_dir, ignore_errors=True)

    best_path = layout.best / "candidate.json"
    if not best_path.is_file():
        raise RuntimeError("no candidate completed the full task suite")
    best = load_candidate(layout.best, "candidate.json")
    current = json.loads((layout.best / "current.json").read_text(encoding="utf-8"))
    with _build_backend(config, num_engines=1) as fresh:
        fresh_scores = {task: float(score) for task, score in fresh.evaluate(best, config.tasks).items()}
        fresh.materialize(best, layout.final / "checkpoint")
    with _build_backend(config, num_engines=1) as replay:
        replay.load_checkpoint(layout.final / "checkpoint")
        replay_scores = {task: float(score) for task, score in replay.evaluate_loaded_checkpoint(config.tasks).items()}
    if fresh_scores != replay_scores:
        raise RuntimeError("reloading the retained checkpoint changed deterministic scores")
    audit = audit_episode(
        workspace=layout.workspace,
        agent_result=agent_result,
        isolation_report=json.loads(isolation.report.read_text(encoding="utf-8")),
        final_submission_count=1,
    )
    (layout.root / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": 1,
        "run_id": config.run_id,
        "candidate_id": best.candidate.candidate_id,
        "search_score": current["average_score"],
        "replay_score": sum(replay_scores.values()) / len(replay_scores),
        "task_scores": replay_scores,
        "checkpoint_reload_equivalent": True,
        "audit_verdict": audit["verdict"],
        "isolation": audit["isolation"],
    }
    (layout.final / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
