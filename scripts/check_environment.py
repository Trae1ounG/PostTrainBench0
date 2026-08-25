#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


EXPECTED_PACKAGES = {
    "torch": "2.6.0",
    "vllm": "0.8.5",
    "transformers": "4.56.0",
    "numpy": "1.26.4",
    "datasets": "4.2.0",
    "pyarrow": "25.0.1",
}

EXPECTED_FILES = {
    "countdown": "countdown/countdown.json",
    "gsm8k": "gsm8k/train.parquet",
    "math500": "math-500/test.jsonl",
    "olympiadbench": "olympiadbench/OE_TO_maths_en_COMP.parquet",
    "mbpp": "mbpp_full/dataset_dict.json",
    "rocstories": "rocstories/train.parquet",
    "uspto50k": "uspto_50k/train.parquet",
}


def package_versions() -> dict[str, str]:
    versions = {}
    for name in EXPECTED_PACKAGES:
        module = importlib.import_module(name)
        versions[name] = str(module.__version__)
    return versions


def data_row_counts(root: Path) -> dict[str, int]:
    import pyarrow.parquet as parquet
    from datasets import load_from_disk

    countdown = json.loads((root / EXPECTED_FILES["countdown"]).read_text(encoding="utf-8"))
    math500 = [
        line
        for line in (root / EXPECTED_FILES["math500"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mbpp = load_from_disk(str(root / "mbpp_full"))
    return {
        "countdown": len(countdown),
        "gsm8k": parquet.read_metadata(root / EXPECTED_FILES["gsm8k"]).num_rows,
        "math500": len(math500),
        "olympiadbench": parquet.read_metadata(root / EXPECTED_FILES["olympiadbench"]).num_rows,
        "mbpp": len(mbpp["train"]),
        "rocstories": parquet.read_metadata(root / EXPECTED_FILES["rocstories"]).num_rows,
        "uspto50k": parquet.read_metadata(root / EXPECTED_FILES["uspto50k"]).num_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    failures = []
    versions = {} if args.data_only else package_versions()
    if not args.data_only:
        if sys.version_info[:2] != (3, 11):
            failures.append(f"Python 3.11 required, found {sys.version.split()[0]}")
        for name, expected in EXPECTED_PACKAGES.items():
            if not versions[name].startswith(expected):
                failures.append(f"{name} {expected} required, found {versions[name]}")
        if shutil.which("bwrap") is None:
            failures.append("bubblewrap (bwrap) is not installed")

    missing = [task for task, relative in EXPECTED_FILES.items() if not (args.data_root / relative).exists()]
    if missing:
        failures.append(f"missing evaluation data: {', '.join(missing)}")
        row_counts = {}
    else:
        row_counts = data_row_counts(args.data_root)
        wrong_counts = {task: count for task, count in row_counts.items() if count != 200}
        if wrong_counts:
            failures.append(f"expected 200 rows per task, found {wrong_counts}")
    manifest_path = args.data_root / "data_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("rows_per_task") != 200:
            failures.append("data manifest must specify 200 rows per task")
    else:
        failures.append("missing data_manifest.json")

    gpu_rows = []
    if not args.skip_gpu and not args.data_only:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
        )
        if process.returncode != 0:
            failures.append("nvidia-smi failed")
        else:
            gpu_rows = [line for line in process.stdout.splitlines() if line.strip()]
            if not gpu_rows:
                failures.append("no NVIDIA GPUs are visible")

    report = {
        "python": sys.version.split()[0],
        "packages": versions,
        "data_root": str(args.data_root.resolve()),
        "data_rows": row_counts,
        "tasks": sorted(EXPECTED_FILES),
        "gpus": gpu_rows,
        "status": "failed" if failures else "ready",
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
