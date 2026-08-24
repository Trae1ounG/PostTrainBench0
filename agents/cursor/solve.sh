#!/usr/bin/env bash
set -euo pipefail

AGENT_HOME="${PTB0_AGENT_HOME:-/home/agent}"
PROMPT_PATH="${PTB0_PROMPT_PATH:-$AGENT_HOME/prompt.txt}"
AGENT_MODEL="${PTB0_AGENT_MODEL:?set PTB0_AGENT_MODEL}"
CURSOR_BIN="${PTB0_CLI_PATH:?set PTB0_CLI_PATH}"
CONTROL_BASH="${PTB0_CONTROL_BASH:-/usr/bin/bash}"
CURSOR_RESUME_SESSION="${PTB0_RESUME_SESSION:-}"
CURSOR_RESUME_PROMPT="${PTB0_RESUME_PROMPT:-Continue the same benchmark run from the existing workspace and score history. Do not repeat completed evaluations. Respect the original deadline.}"
cd "$AGENT_HOME"

common_args=(
  --print
  --force
  --output-format stream-json
  --sandbox disabled
  --trust
  --workspace "$AGENT_HOME"
  --model "$AGENT_MODEL"
)

if [[ -n "$CURSOR_RESUME_SESSION" ]]; then
  exec "$CONTROL_BASH" "$CURSOR_BIN" \
    "${common_args[@]}" \
    --resume="$CURSOR_RESUME_SESSION" \
    "$CURSOR_RESUME_PROMPT"
fi

exec "$CONTROL_BASH" "$CURSOR_BIN" \
  "${common_args[@]}" \
  "$(<"$PROMPT_PATH")"
