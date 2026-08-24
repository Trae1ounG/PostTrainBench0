#!/usr/bin/env bash
set -euo pipefail

AGENT_HOME="${PTB0_AGENT_HOME:-/home/agent}"
PROMPT_PATH="${PTB0_PROMPT_PATH:-$AGENT_HOME/prompt.txt}"
AGENT_MODEL="${PTB0_AGENT_MODEL:?set PTB0_AGENT_MODEL}"
CODEX_BIN="${PTB0_CLI_PATH:?set PTB0_CLI_PATH}"
CODEX_CONFIG_OVERRIDES="${PTB0_CODEX_CONFIG_OVERRIDES:-}"
CODEX_SANDBOX_MODE="${PTB0_CODEX_SANDBOX_MODE:-danger-full-access}"
CODEX_RESUME_SESSION="${PTB0_RESUME_SESSION:-}"
CODEX_RESUME_PROMPT="${PTB0_RESUME_PROMPT:-Continue the same benchmark run from the existing workspace and score history. Do not repeat completed evaluations. Respect the original deadline.}"
cd "$AGENT_HOME"

config_args=()
while IFS= read -r override; do
  [[ -n "$override" ]] && config_args+=(--config "$override")
done <<< "$CODEX_CONFIG_OVERRIDES"

common_args=(
  --sandbox "$CODEX_SANDBOX_MODE"
  --ask-for-approval never
  --cd "$AGENT_HOME"
  "${config_args[@]}"
)

if [[ -n "$CODEX_RESUME_SESSION" ]]; then
  exec "$CODEX_BIN" \
    "${common_args[@]}" \
    exec resume \
    --json \
    --skip-git-repo-check \
    --model "$AGENT_MODEL" \
    "$CODEX_RESUME_SESSION" \
    "$CODEX_RESUME_PROMPT"
fi

exec "$CODEX_BIN" \
  "${common_args[@]}" \
  exec \
  --json \
  --skip-git-repo-check \
  --model "$AGENT_MODEL" \
  - < "$PROMPT_PATH"
