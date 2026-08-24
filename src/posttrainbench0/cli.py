from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RunConfig
from .episode import initialize
from .runtime import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated PostTrainBench0 episode")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init-only", action="store_true", help="create and print the run tree without starting GPUs or an Agent")
    args = parser.parse_args()
    config = RunConfig.load(args.config.resolve())
    layout = initialize(config)
    if args.init_only:
        print(layout.root)
        return
    print(json.dumps(run(config, layout), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
