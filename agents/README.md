# Harness adapters

Every harness receives the same workspace, prompt, model, GPU allocation, and
deadline. A harness adapter may translate those inputs into its native CLI, but
must not add task-specific strategy instructions.

Every adapter receives the same rendered instruction and the same isolated
workspace. The adapter only translates that contract into a native CLI call.

Required environment:

- `PTB0_AGENT_HOME`: clean writable home, normally `/home/agent`;
- `PTB0_PROMPT_PATH`: rendered prompt, normally `/home/agent/prompt.txt`;
- `PTB0_AGENT_MODEL`: model name passed to the harness;
- `PTB0_CLI_PATH`: the mounted CLI executable;
- current directory: `/home/agent`.

The adapter writes its native trace to standard output/error and exits when the
agent finishes. The outer episode runner owns hard timeout and log capture.

Adapters must not enable web search or bypass the harness sandbox. They should
run without interactive approval, with writes limited to the agent workspace,
and without loading user-specific Codex configuration. The outer Trial
container remains responsible for restricting what the process can read and
for disabling non-agent network routes.

The release includes thin adapters for Codex, Cursor Agent, and OpenCode. API
credentials remain in a trusted per-run control home; they are never copied
into the Agent workspace or run manifest.
