#!/usr/bin/env bash
set -euo pipefail

AGENT_HOME="${PTB0_AGENT_HOME:-/home/agent}"
PROMPT_PATH="${PTB0_PROMPT_PATH:-$AGENT_HOME/prompt.txt}"
AGENT_MODEL="${PTB0_AGENT_MODEL:?set PTB0_AGENT_MODEL}"
OPENCODE_BIN="${PTB0_CLI_PATH:?set PTB0_CLI_PATH}"
cd "$AGENT_HOME"

exec "$OPENCODE_BIN" run \
  --model "$AGENT_MODEL" \
  --format json \
  < "$PROMPT_PATH"
