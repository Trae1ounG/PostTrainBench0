from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from posttrainbench0.config import RunConfig
from posttrainbench0.controller import EpisodeController
from posttrainbench0.episode import initialize
from posttrainbench0.evaluators.randopt_vllm import RandOptVllmBackend, _LocalVllmEngine


ROOT = Path(__file__).resolve().parents[1]


class FakeEvaluator:
    def evaluate_batch(self, candidates, tasks):
        return [
            {task: float(candidate.candidate.terms[0].scale) for task in tasks}
            for candidate in candidates
        ]


def make_config(tmp_path: Path, *, run_id: str = "test-run") -> RunConfig:
    for name in ("model", "data", "randopt"):
        (tmp_path / name).mkdir()
    cli = tmp_path / "agent-cli"
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    cli.chmod(0o755)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "paths": {
                    "base_model": str(tmp_path / "model"),
                    "evaluation_data": str(tmp_path / "data"),
                    "randopt_source": str(tmp_path / "randopt"),
                    "runs_root": str(tmp_path / "runs"),
                    "prompt": str(ROOT / "prompt.txt"),
                    "starter": str(ROOT / "starter"),
                },
                "agent": {"harness": "opencode", "model": "test/model", "cli_path": str(cli)},
                "evaluation": {"samples_per_task": 2, "tasks": ["countdown", "gsm8k"]},
                "runtime": {"hours": 1, "num_gpus": 2},
            }
        ),
        encoding="utf-8",
    )
    return RunConfig.load(config_path)


def write_candidate(workspace: Path, name: str, scale: float) -> Path:
    path = workspace / name
    path.write_text(
        json.dumps(
            {
                "format": "zerograd-noise-program-v1",
                "candidate_id": name,
                "terms": [{"seed": 1, "scale": scale}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_initialization_creates_only_the_documented_agent_surface(tmp_path: Path) -> None:
    layout = initialize(make_config(tmp_path))
    assert sorted(path.name for path in layout.workspace.iterdir()) == [
        "bin",
        "episode.json",
        "prompt.txt",
        "starter",
        "timer.sh",
    ]
    assert sorted(path.name for path in (layout.workspace / "bin").iterdir()) == [
        "evaluate",
        "evaluate-batch",
        "results",
        "status",
    ]
    assert not (layout.workspace / "run.json").exists()
    with pytest.raises(FileExistsError):
        initialize(make_config(tmp_path))


def test_only_full_suite_candidates_are_retained_and_history_is_append_only(tmp_path: Path) -> None:
    layout = initialize(make_config(tmp_path))
    write_candidate(layout.workspace, "weak.json", 0.2)
    write_candidate(layout.workspace, "strong.json", 0.8)
    controller = EpisodeController(layout.root, FakeEvaluator())
    subset = controller.handle({"operation": "evaluate", "candidate": "strong.json", "tasks": ["countdown"]})
    assert subset["retained_as_best"] is False
    first = controller.handle({"operation": "evaluate", "candidate": "weak.json", "tasks": []})
    second = controller.handle({"operation": "evaluate", "candidate": "strong.json", "tasks": []})
    assert first["retained_as_best"] is True
    assert second["retained_as_best"] is True
    current = json.loads((layout.best / "current.json").read_text())
    assert current["average_score"] == pytest.approx(0.8)
    assert len(list((layout.best / "history").iterdir())) == 2
    assert len(list(layout.attempts.iterdir())) == 3


def test_prompt_and_starters_match_the_public_contract() -> None:
    prompt = " ".join((ROOT / "prompt.txt").read_text(encoding="utf-8").split())
    assert "RandOpt and ES" in prompt
    assert "no active submission command is required" in prompt
    assert "bin/submit" not in prompt
    forbidden = {"torch", "jax", "tensorflow"}
    for path in (ROOT / "starter").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint(forbidden), path.name


def test_main_branch_has_no_blog_application() -> None:
    assert not (ROOT / "app" / "page.tsx").exists()
    assert not (ROOT / "package.json").exists()


def test_reference_data_and_container_contract_are_shipped() -> None:
    manifest = json.loads((ROOT / "data" / "visible200" / "data_manifest.json").read_text())
    assert manifest["rows_per_task"] == 200
    assert set(manifest["tasks"]) == {
        "countdown",
        "gsm8k",
        "math500",
        "olympiadbench",
        "mbpp",
        "rocstories",
        "uspto50k",
    }
    for relative in manifest["tasks"].values():
        assert (ROOT / "data" / "visible200" / relative).exists()
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.11-slim-bookworm" in dockerfile
    assert "vllm==0.8.5" in dockerfile
    assert "80ec7f6d97e0e3e56b9d58fcf22094a73dd489f6" in dockerfile


def test_local_engine_process_uses_direct_ipc(tmp_path: Path, monkeypatch) -> None:
    fake_package = tmp_path / "vllm"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text(
        "class SamplingParams:\n"
        "    def __init__(self, **values): self.values = values\n"
        "class LLM:\n"
        "    def __init__(self, **values): self.values = values\n"
        "    def collective_rpc(self, method, args=()): return [method, list(args)]\n"
        "    def generate(self, prompts, params, use_tqdm=False):\n"
        "        return [{'prompt': item, 'sampling': params.values} for item in prompts]\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    engine = _LocalVllmEngine(gpu_devices="0", kwargs={})
    try:
        assert engine.call(
            "collective_rpc", method="apply_noise_program", args=([{"seed": 1}],)
        )[0] == "apply_noise_program"
        generated = engine.call(
            "generate", prompts=["hello"], sampling={"temperature": 0.0}
        )
        assert generated == [{"prompt": "hello", "sampling": {"temperature": 0.0}}]
    finally:
        engine.close()


def test_mbpp_scoring_always_uses_the_networkless_sandbox(tmp_path: Path, monkeypatch) -> None:
    backend = RandOptVllmBackend(
        model_path=tmp_path,
        data_root=tmp_path,
        randopt_source=tmp_path,
        samples_per_task=1,
        num_engines=1,
    )
    backend.task_data["mbpp"] = [{"ground_truth": "tests"}]
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout='ZEROGRAD_MBPP_RESULT={"score": 1.0}\n',
            stderr="",
        )

    monkeypatch.setattr("posttrainbench0.evaluators.randopt_vllm.shutil.which", lambda _: "/usr/bin/bwrap")
    monkeypatch.setattr("posttrainbench0.evaluators.randopt_vllm.subprocess.run", fake_run)
    output = SimpleNamespace(outputs=[SimpleNamespace(text="def answer(): pass")])
    assert backend._score_mbpp_outputs([output]) == 1.0
    assert "--unshare-all" in captured["command"]
    assert "--share-net" not in captured["command"]
    assert captured["environment"] == {"PATH": "/usr/local/bin:/usr/bin:/bin"}
