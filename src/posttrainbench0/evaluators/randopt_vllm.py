from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Mapping

from ..candidate_io import LoadedCandidate


TASK_PATHS = {
    "countdown": "countdown/countdown.json",
    "gsm8k": "gsm8k/train.parquet",
    "math500": "math-500/test.jsonl",
    "olympiadbench": "olympiadbench/OE_TO_maths_en_COMP.parquet",
    "mbpp": "mbpp_full",
    "rocstories": "rocstories/train.parquet",
    "uspto50k": "uspto_50k/train.parquet",
}

CUDA_CONTEXT_FAILURE_MARKERS = (
    "CUDA error: an illegal memory access was encountered",
    "CUDA error: device-side assert triggered",
)


class RandOptVllmBackend:
    """Trusted seven-task evaluator backed by resident vLLM Ray actors."""

    def __init__(
        self,
        *,
        model_path: Path,
        data_root: Path,
        randopt_source: Path,
        samples_per_task: int,
        num_engines: int,
        precision: str = "bfloat16",
        tensor_parallel_size: int = 1,
        global_seed: int = 42,
        sandbox_tools_root: Path | None = None,
        placement_group_timeout_seconds: float = 120,
    ) -> None:
        self.model_path = model_path.resolve()
        self.data_root = data_root.resolve()
        self.randopt_source = randopt_source.resolve()
        self.samples_per_task = samples_per_task
        self.num_engines = num_engines
        self.precision = precision
        self.tensor_parallel_size = tensor_parallel_size
        self.global_seed = global_seed
        self.sandbox_tools_root = (
            sandbox_tools_root.resolve() if sandbox_tools_root is not None else None
        )
        if placement_group_timeout_seconds <= 0:
            raise ValueError("placement_group_timeout_seconds must be positive")
        self.placement_group_timeout_seconds = placement_group_timeout_seconds
        self.engines = []
        self.placement_groups = []
        self.handlers = {}
        self.task_data = {}
        self.prompts = {}
        self._ray = None
        self._sampling_params = None

    def start(self) -> None:
        if self.engines:
            raise RuntimeError("backend is already started")
        import sys

        sys.path.insert(0, str(self.randopt_source))
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        from data_handlers import get_dataset_handler

        worker_env = {
            key: os.environ[key]
            for key in ("PYTHONPATH", "TOKENIZERS_PARALLELISM", "VLLM_USE_V1")
            if key in os.environ
        }
        runtime_env = {"env_vars": worker_env} if worker_env else None
        if os.environ.get("RAY_ADDRESS"):
            ray.init(
                address="auto",
                ignore_reinit_error=True,
                runtime_env=runtime_env,
            )
        else:
            ray.init(
                address="local",
                ignore_reinit_error=True,
                runtime_env=runtime_env,
            )
        required_gpus = self.num_engines * self.tensor_parallel_size
        visible_gpus = int(ray.cluster_resources().get("GPU", 0))
        if visible_gpus < required_gpus:
            raise RuntimeError(f"Ray exposes {visible_gpus} GPUs; {required_gpus} required")

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        is_instruct = any(
            marker in str(self.model_path).lower() for marker in ("instruct", "chat", "it")
        )

        def format_prompt(messages) -> str:
            if is_instruct and tokenizer.chat_template:
                return tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            return "\n".join(message["content"] for message in messages) + "\n"

        for task, relative_path in TASK_PATHS.items():
            path = self.data_root / relative_path
            if not path.exists():
                raise FileNotFoundError(path)
            handler = get_dataset_handler(task)
            rows = handler.load_data(
                str(path), split="train", max_samples=self.samples_per_task
            )
            if len(rows) != self.samples_per_task:
                raise ValueError(
                    f"{task} loaded {len(rows)} examples, expected {self.samples_per_task}"
                )
            self.handlers[task] = handler
            self.task_data[task] = rows
            self.prompts[task] = [format_prompt(row["messages"]) for row in rows]

        class ZeroGradLLM(LLM):
            def __init__(self, *args, **kwargs):
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                super().__init__(*args, **kwargs)

        bundles = [{"GPU": 1, "CPU": 0} for _ in range(self.tensor_parallel_size)]
        self.placement_groups = [
            placement_group(bundles) for _ in range(self.num_engines)
        ]
        try:
            ray.get(
                [group.ready() for group in self.placement_groups],
                timeout=self.placement_group_timeout_seconds,
            )
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                )
                for group in self.placement_groups
            ]
            kwargs = {
                "model": str(self.model_path),
                "tensor_parallel_size": self.tensor_parallel_size,
                "distributed_executor_backend": "ray",
                "worker_extension_cls": "posttrainbench0.evaluators.randopt_worker.PostTrainBenchWorker",
                "dtype": self.precision,
                "enforce_eager": True,
                "gpu_memory_utilization": 0.75,
                "disable_log_stats": True,
            }
            # vLLM creates a local torch.distributed TCP store even when each
            # engine uses one GPU. Starting all eight constructors at once can
            # make two processes select the same free port before either binds
            # it. Initialize each resident engine fully before starting the
            # next; this time is outside the Agent's wall-clock window.
            for strategy in strategies:
                engine = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    scheduling_strategy=strategy,
                )(ZeroGradLLM).remote(**kwargs)
                self.engines.append(engine)
                ray.get(
                    engine.collective_rpc.remote("store_base_weights", args=())
                )
        except Exception:
            self.close()
            raise
        self._ray = ray
        self._sampling_params = SamplingParams

    def _require_started(self) -> None:
        if not self.engines or self._ray is None or self._sampling_params is None:
            raise RuntimeError("backend.start() must be called first")

    @staticmethod
    def _serialized_terms(candidate: LoadedCandidate) -> list[dict]:
        return [
            {"seed": term.seed, "scale": term.scale}
            for term in candidate.candidate.terms
        ]

    def evaluate(self, candidate: LoadedCandidate, tasks: tuple[str, ...]) -> Mapping[str, float]:
        return self.evaluate_batch([candidate], tasks)[0]

    def evaluate_batch(
        self, candidates: list[LoadedCandidate], tasks: tuple[str, ...]
    ) -> list[Mapping[str, float]]:
        self._require_started()
        if not candidates or len(candidates) > len(self.engines):
            raise ValueError(
                f"batch must contain between 1 and {len(self.engines)} candidates"
            )
        unknown = sorted(set(tasks) - set(self.handlers))
        if unknown:
            raise ValueError(f"unknown tasks: {unknown}")

        ray = self._ray
        active = self.engines[: len(candidates)]
        try:
            ray.get(
                [
                    engine.collective_rpc.remote(
                        "apply_noise_program", args=(self._serialized_terms(candidate),)
                    )
                    for engine, candidate in zip(active, candidates)
                ]
            )
            scores = self._score_active_engines(active, tasks)
            ray.get(
                [
                    engine.collective_rpc.remote("reset_to_base_weights", args=())
                    for engine in active
                ]
            )
            return scores
        except Exception as error:
            if self._recover_if_cuda_context_failed(error):
                raise
            try:
                ray.get(
                    [
                        engine.collective_rpc.remote("reset_to_base_weights", args=())
                        for engine in active
                    ]
                )
            except Exception as reset_error:
                self._recover_if_cuda_context_failed(reset_error)
            raise

    def _recover_if_cuda_context_failed(self, error: Exception) -> bool:
        if not any(marker in str(error) for marker in CUDA_CONTEXT_FAILURE_MARKERS):
            return False
        try:
            self.close()
            self.start()
        except Exception as recovery_error:
            raise RuntimeError(
                "resident evaluator CUDA context failed and backend restart failed"
            ) from recovery_error
        return True

    def _score_active_engines(self, engines, tasks: tuple[str, ...]) -> list[dict[str, float]]:
        results: list[dict[str, float]] = [{} for _ in engines]
        for task in tasks:
            params = self._sampling_params(
                temperature=0.0,
                seed=self.global_seed,
                max_tokens=self.handlers[task].default_max_tokens,
            )
            outputs = self._ray.get(
                [
                    engine.generate.remote(self.prompts[task], params, use_tqdm=False)
                    for engine in engines
                ]
            )
            if task == "mbpp":
                with ThreadPoolExecutor(max_workers=len(outputs)) as executor:
                    scores = list(executor.map(self._score_mbpp_outputs, outputs))
                for index, score in enumerate(scores):
                    results[index][task] = score
                continue
            for index, output in enumerate(outputs):
                results[index][task] = float(
                    self.handlers[task].postprocess_outputs(output, self.task_data[task])
                )
        return results

    def _score_mbpp_outputs(self, outputs) -> float:
        """Score generated code in a bounded main-thread child process."""

        responses = [output.outputs[0].text for output in outputs]
        ground_truths = [row["ground_truth"] for row in self.task_data["mbpp"]]
        scorer_args = [
            "-m",
            "posttrainbench0.mbpp_scorer",
            "--samples",
            str(self.samples_per_task),
            "--randopt-source",
            str(self.randopt_source),
        ]
        child_env = {
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": os.pathsep.join(
                (str(Path(__file__).resolve().parents[2]), str(self.randopt_source))
            ),
        }
        command = [sys.executable, *scorer_args]
        if self.sandbox_tools_root is not None:
            tools_bin = self.sandbox_tools_root / "root" / "usr" / "bin"
            tools_lib = self.sandbox_tools_root / "root" / "usr" / "lib" / "x86_64-linux-gnu"
            bwrap = tools_bin / "bwrap"
            if not bwrap.is_file():
                raise FileNotFoundError(f"missing Bubblewrap executable: {bwrap}")
            source_root = Path(__file__).resolve().parents[2]
            binds: list[str] = []
            for path in ("/bin", "/usr", "/usr/local", "/lib", "/lib64"):
                if Path(path).exists():
                    binds.extend(("--ro-bind", path, path))
            command = [
                str(bwrap),
                "--unshare-all",
                "--die-with-parent",
                "--new-session",
                "--clearenv",
                *binds,
                "--dir",
                "/dev",
                "--dev-bind",
                "/dev/null",
                "/dev/null",
                "--dev-bind",
                "/dev/urandom",
                "/dev/urandom",
                "--dev-bind",
                "/dev/zero",
                "/dev/zero",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/home",
                "--dir",
                "/opt",
                "--ro-bind",
                str(self.randopt_source),
                "/opt/randopt",
                "--ro-bind",
                str(source_root),
                "/opt/posttrainbench0-src",
                "--setenv",
                "HOME",
                "/home",
                "--setenv",
                "LANG",
                "C.UTF-8",
                "--setenv",
                "PATH",
                "/usr/local/bin:/usr/bin:/bin",
                "--setenv",
                "PYTHONNOUSERSITE",
                "1",
                "--setenv",
                "PYTHONPATH",
                "/opt/posttrainbench0-src:/opt/randopt",
                "--chdir",
                "/tmp",
                "/usr/bin/python3",
                "-m",
                "posttrainbench0.mbpp_scorer",
                "--samples",
                str(self.samples_per_task),
                "--randopt-source",
                "/opt/randopt",
            ]
            child_env = {
                "LD_LIBRARY_PATH": str(tools_lib),
                "PATH": f"{tools_bin}:/usr/local/bin:/usr/bin:/bin",
            }
        process = subprocess.run(
            command,
            input=json.dumps({"responses": responses, "ground_truths": ground_truths}),
            text=True,
            capture_output=True,
            timeout=180,
            env=child_env,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "bounded MBPP scorer failed with exit code "
                f"{process.returncode}: {process.stderr[-500:]}"
            )
        marker = "ZEROGRAD_MBPP_RESULT="
        line = next(
            (item for item in reversed(process.stdout.splitlines()) if item.startswith(marker)),
            None,
        )
        if line is None:
            raise RuntimeError("bounded MBPP scorer did not return a result")
        return float(json.loads(line[len(marker) :])["score"])

    def materialize(self, candidate: LoadedCandidate, output_dir: Path) -> Mapping[str, object]:
        self._require_started()
        if output_dir.exists():
            raise FileExistsError(output_dir)
        output_dir.mkdir(parents=True)
        state_path = output_dir / "model_state.pt"
        engine = self.engines[0]
        ray = self._ray
        ray.get(
            engine.collective_rpc.remote(
                "apply_noise_program", args=(self._serialized_terms(candidate),)
            )
        )
        try:
            metadata = ray.get(
                engine.collective_rpc.remote(
                    "save_posttrainbench0_checkpoint", args=(str(state_path),)
                )
            )[0]
        finally:
            ray.get(engine.collective_rpc.remote("reset_to_base_weights", args=()))

        for filename in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ):
            source = self.model_path / filename
            if source.is_file():
                shutil.copy2(source, output_dir / filename)
        checkpoint = {
            "schema_version": 1,
            "format": "posttrainbench0-vllm-state-v1",
            "base_model_path": str(self.model_path),
            "state_file": state_path.name,
            "candidate_id": candidate.candidate.candidate_id,
            "terms": self._serialized_terms(candidate),
            **metadata,
        }
        (output_dir / "checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return checkpoint

    def load_checkpoint(self, checkpoint_dir: Path) -> Mapping[str, object]:
        self._require_started()
        checkpoint = json.loads(
            (checkpoint_dir / "checkpoint.json").read_text(encoding="utf-8")
        )
        if checkpoint.get("format") != "posttrainbench0-vllm-state-v1":
            raise ValueError("unsupported checkpoint format")
        state_path = checkpoint_dir / checkpoint["state_file"]
        return self._ray.get(
            self.engines[0].collective_rpc.remote(
                "load_posttrainbench0_checkpoint", args=(str(state_path),)
            )
        )[0]

    def evaluate_loaded_checkpoint(self, tasks: tuple[str, ...]) -> Mapping[str, float]:
        """Score the checkpoint already loaded into the first resident engine."""

        self._require_started()
        unknown = sorted(set(tasks) - set(self.handlers))
        if unknown:
            raise ValueError(f"unknown tasks: {unknown}")
        return self._score_active_engines(self.engines[:1], tasks)[0]

    def close(self) -> None:
        if self._ray is None:
            try:
                import ray
            except ImportError:
                return
        else:
            ray = self._ray
        from ray.util.placement_group import remove_placement_group

        for engine in self.engines:
            try:
                ray.kill(engine)
            except Exception:
                pass
        for group in self.placement_groups:
            try:
                remove_placement_group(group)
            except Exception:
                pass
        self.engines = []
        self.placement_groups = []
        if ray.is_initialized():
            ray.shutdown()
        self._ray = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
