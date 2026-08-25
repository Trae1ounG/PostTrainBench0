#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 CONFIG MODEL_DIR RUNS_DIR AGENT_CLI_DIR [ENV_FILE]" >&2
  exit 2
fi

CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
MODEL_DIR="$(cd "$2" && pwd)"
RUNS_DIR="$(cd "$3" && pwd)"
AGENT_CLI_DIR="$(cd "$4" && pwd)"
ENV_FILE="${5:-}"

for path in "$CONFIG" "$MODEL_DIR" "$RUNS_DIR" "$AGENT_CLI_DIR"; do
  [[ -e "$path" ]] || { echo "missing path: $path" >&2; exit 1; }
done

docker_args=(
  run --rm
  --gpus all
  --ipc host
  --network host
  --privileged
  --volume "$CONFIG:/run-config/config.json:ro"
  --volume "$MODEL_DIR:/models/base:ro"
  --volume "$RUNS_DIR:/runs"
  --volume "$AGENT_CLI_DIR:/opt/agent-cli:ro"
)
if [[ -n "$ENV_FILE" ]]; then
  ENV_FILE="$(cd "$(dirname "$ENV_FILE")" && pwd)/$(basename "$ENV_FILE")"
  [[ -f "$ENV_FILE" ]] || { echo "missing env file: $ENV_FILE" >&2; exit 1; }
  docker_args+=(--env-file "$ENV_FILE")
fi

exec docker "${docker_args[@]}" posttrainbench0:0.1.0 \
  posttrainbench0 --config /run-config/config.json
