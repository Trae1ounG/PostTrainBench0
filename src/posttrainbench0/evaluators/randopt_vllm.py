from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
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


def _engine_process(connection, gpu_devices: str, kwargs: dict) -> None:
    """Own one persistent vLLM engine in a GPU-isolated child process."""

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM, SamplingParams

        engine = LLM(**kwargs)
        engine.collective_rpc("store_base_weights", args=())
        connection.send({"ok": True, "result": "ready"})
        while True:
            request = connection.recv()
            operation = request["operation"]
            if operation == "close":
                connection.send({"ok": True, "result": None})
                return
            if operation == "collective_rpc":
                result = engine.collective_rpc(
                    request["method"], args=tuple(request.get("args", ()))
                )
            elif operation == "generate":
                params = SamplingParams(**request["sampling"])
                result = engine.generate(request["prompts"], params, use_tqdm=False)
            else:
                raise ValueError(f"unknown engine operation: {operation}")
            connection.send({"ok": True, "result": result})
    except BaseException as error:
        try:
            connection.send(
                {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        connection.close()


class _LocalVllmEngine:
    def __init__(self, *, gpu_devices: str, kwargs: dict, startup_timeout: float = 1800) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_engine_process,
            args=(child, gpu_devices, kwargs),
            name=f"posttrainbench0-vllm-{gpu_devices}",
        )
        self._process.start()
        child.close()
        if not parent.poll(startup_timeout):
            self._process.terminate()
            self._process.join(timeout=10)
            raise TimeoutError(f"vLLM engine on GPU {gpu_devices} did not start")
        self._receive()

    def _receive(self):
        try:
            response = self._connection.recv()
        except EOFError as error:
            raise RuntimeError("local vLLM engine exited without a response") from error
        if not response["ok"]:
            raise RuntimeError(
                f"local vLLM engine failed: {response['error']}\n{response['traceback']}"
            )
        return response["result"]

    def call(self, operation: str, **payload):
        if not self._process.is_alive():
            raise RuntimeError("local vLLM engine is not running")
        self._connection.send({"operation": operation, **payload})
        return self._receive()

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self.call("close")
            except Exception:
                self._process.terminate()
        self._process.join(timeout=15)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=5)
        self._connection.close()


class RandOptVllmBackend:
    """Trusted seven-task evaluator backed by local resident vLLM processes."""

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
        self.engines = []
        self.handlers = {}
        self.task_data = {}
        self.prompts = {}

    def start(self) -> None:
        if self.engines:
            raise RuntimeError("backend is already started")
        import sys

        sys.path.insert(0, str(self.randopt_source))
        from transformers import AutoTokenizer

        from data_handlers import get_dataset_handler

        required_gpus = self.num_engines * self.tensor_parallel_size
        configured = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if configured:
            gpu_tokens = [item.strip() for item in configured.split(",") if item.strip()]
        else:
            process = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
                capture_output=True,
            )
            if process.returncode != 0:
                raise RuntimeError(f"nvidia-smi failed: {process.stderr.strip()}")
            gpu_tokens = [line.strip() for line in process.stdout.splitlines() if line.strip()]
        if len(gpu_tokens) < required_gpus:
            raise RuntimeError(
                f"{len(gpu_tokens)} GPUs are visible; {required_gpus} are required"
            )

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

        try:
            kwargs = {
                "model": str(self.model_path),
                "tensor_parallel_size": self.tensor_parallel_size,
                "worker_extension_cls": "posttrainbench0.evaluators.randopt_worker.PostTrainBenchWorker",
                "dtype": self.precision,
                "enforce_eager": True,
                "gpu_memory_utilization": 0.75,
                "disable_log_stats": True,
            }
            if self.tensor_parallel_size > 1:
                kwargs["distributed_executor_backend"] = "mp"
            for index in range(self.num_engines):
                start = index * self.tensor_parallel_size
                devices = ",".join(gpu_tokens[start : start + self.tensor_parallel_size])
                self.engines.append(
                    _LocalVllmEngine(gpu_devices=devices, kwargs=kwargs)
                )
        except Exception:
            self.close()
            raise

    def _require_started(self) -> None:
        if not self.engines:
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

        active = self.engines[: len(candidates)]
        try:
            with ThreadPoolExecutor(max_workers=len(active)) as executor:
                list(
                    executor.map(
                        lambda pair: pair[0].call(
                            "collective_rpc",
                            method="apply_noise_program",
                            args=(self._serialized_terms(pair[1]),),
                        ),
                        zip(active, candidates),
                    )
                )
            scores = self._score_active_engines(active, tasks)
            with ThreadPoolExecutor(max_workers=len(active)) as executor:
                list(
                    executor.map(
                        lambda engine: engine.call(
                            "collective_rpc", method="reset_to_base_weights", args=()
                        ),
                        active,
                    )
                )
            return scores
        except Exception as error:
            if self._recover_if_cuda_context_failed(error):
                raise
            try:
                with ThreadPoolExecutor(max_workers=len(active)) as executor:
                    list(
                        executor.map(
                            lambda engine: engine.call(
                                "collective_rpc", method="reset_to_base_weights", args=()
                            ),
                            active,
                        )
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
            sampling = {
                "temperature": 0.0,
                "seed": self.global_seed,
                "max_tokens": self.handlers[task].default_max_tokens,
            }
            with ThreadPoolExecutor(max_workers=len(engines)) as executor:
                outputs = list(
                    executor.map(
                        lambda engine: engine.call(
                            "generate", prompts=self.prompts[task], sampling=sampling
                        ),
                        engines,
                    )
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
        source_root = Path(__file__).resolve().parents[2]
        if self.sandbox_tools_root is not None:
            tools_bin = self.sandbox_tools_root / "root" / "usr" / "bin"
            tools_lib = self.sandbox_tools_root / "root" / "usr" / "lib" / "x86_64-linux-gnu"
            bwrap = tools_bin / "bwrap"
            if not bwrap.is_file():
                raise FileNotFoundError(f"missing Bubblewrap executable: {bwrap}")
            child_env = {
                "LD_LIBRARY_PATH": str(tools_lib),
                "PATH": f"{tools_bin}:/usr/local/bin:/usr/bin:/bin",
            }
            sandbox_python = "/usr/bin/python3"
        else:
            executable = shutil.which("bwrap")
            if executable is None:
                raise FileNotFoundError("bubblewrap is required to score MBPP safely")
            bwrap = Path(executable)
            sandbox_python = sys.executable
            child_env = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
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
            "--dev",
            "/dev",
            "--proc",
            "/proc",
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
            sandbox_python,
            "-m",
            "posttrainbench0.mbpp_scorer",
            "--samples",
            str(self.samples_per_task),
            "--randopt-source",
            "/opt/randopt",
        ]
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
        engine.call(
            "collective_rpc",
            method="apply_noise_program",
            args=(self._serialized_terms(candidate),),
        )
        try:
            metadata = engine.call(
                "collective_rpc",
                method="save_posttrainbench0_checkpoint",
                args=(str(state_path),),
            )[0]
        finally:
            engine.call(
                "collective_rpc", method="reset_to_base_weights", args=()
            )

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
        return self.engines[0].call(
            "collective_rpc",
            method="load_posttrainbench0_checkpoint",
            args=(str(state_path),),
        )[0]

    def evaluate_loaded_checkpoint(self, tasks: tuple[str, ...]) -> Mapping[str, float]:
        """Score the checkpoint already loaded into the first resident engine."""

        self._require_started()
        unknown = sorted(set(tasks) - set(self.handlers))
        if unknown:
            raise ValueError(f"unknown tasks: {unknown}")
        return self._score_active_engines(self.engines[:1], tasks)[0]

    def close(self) -> None:
        for engine in self.engines:
            try:
                engine.close()
            except Exception:
                pass
        self.engines = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
