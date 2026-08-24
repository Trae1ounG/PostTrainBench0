from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import shutil
import socket
import socketserver
import subprocess
import threading


@dataclass(frozen=True)
class Isolation:
    shell: Path
    report: Path


class _ThreadingServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class CommandBroker:
    """Execute CLI tool commands inside the no-network Agent shell."""

    def __init__(self, socket_path: Path, shell: Path) -> None:
        self.socket_path = socket_path
        launcher = str(shell)

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                request = json.loads(self.rfile.readline())
                argv = request.get("argv")
                if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
                    raise ValueError("argv must be a list of strings")
                process = subprocess.run([launcher, *argv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                response = {"returncode": process.returncode, "output": base64.b64encode(process.stdout).decode()}
                self.wfile.write(json.dumps(response).encode() + b"\n")

        socket_path.unlink(missing_ok=True)
        self.server = _ThreadingServer(str(socket_path), Handler)
        socket_path.chmod(0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "CommandBroker":
        self.thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.socket_path.unlink(missing_ok=True)


def _bwrap() -> str:
    executable = shutil.which("bwrap")
    if executable is None:
        raise FileNotFoundError("bubblewrap (bwrap) is required on the Linux Trial")
    return executable


def _system_binds() -> list[str]:
    result: list[str] = []
    for path in ("/bin", "/usr", "/lib", "/lib64"):
        if Path(path).exists():
            result.extend(("--ro-bind", path, path))
    return result


def prepare_agent_shell(*, run_root: Path, workspace: Path, base_model: Path, control_dir: Path) -> Isolation:
    private = run_root / "isolation"
    private.mkdir(mode=0o700)
    shell = private / "agent-shell"
    command = [
        _bwrap(), "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
        *_system_binds(),
        "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--dir", "/home",
        "--bind", str(workspace), "/home/agent",
        "--dir", "/models", "--ro-bind", str(base_model), "/models/base",
        "--dir", "/run", "--ro-bind", str(control_dir), "/run/posttrainbench0",
        "--setenv", "HOME", "/home/agent",
        "--setenv", "PATH", "/home/agent/bin:/usr/local/bin:/usr/bin:/bin",
        "--setenv", "PYTHONPATH", "/home/agent/starter",
        "--setenv", "PTB0_EVALUATOR_SOCKET", "/run/posttrainbench0/evaluator.sock",
        "--setenv", "TMPDIR", "/tmp", "--setenv", "LANG", "C.UTF-8",
        "--chdir", "/home/agent", "/bin/bash",
    ]
    shell.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + " ".join(shlex.quote(item) for item in command) + ' "$@"\n', encoding="utf-8")
    shell.chmod(0o700)
    probe = subprocess.run(
        [str(shell), "-lc", "test \"$PWD\" = /home/agent; touch probe; rm probe; ! touch /models/base/probe 2>/dev/null; ! test -e /mnt; python3 -c 'import socket; s=socket.socket(); s.settimeout(.2); assert s.connect_ex((\"1.1.1.1\",80)) != 0'"],
        text=True,
        capture_output=True,
        timeout=20,
    )
    report = {
        "backend": "bubblewrap",
        "probe_exit_code": probe.returncode,
        "workspace_is_only_writable_mount": probe.returncode == 0,
        "base_model_is_read_only": probe.returncode == 0,
        "evaluator_not_visible": probe.returncode == 0,
        "general_network_blocked": probe.returncode == 0,
    }
    report_path = run_root / "isolation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if probe.returncode != 0:
        raise RuntimeError(f"isolation probe failed; see {report_path}")
    return Isolation(shell=shell, report=report_path)


def prepare_harness_launcher(
    *,
    run_root: Path,
    workspace: Path,
    base_model: Path,
    control_dir: Path,
    control_home: Path,
    cli_path: Path,
    harness: str,
) -> Path:
    """Give the networked CLI only the Agent workspace and a brokered shell."""

    private = run_root / "isolation"
    command_client = control_dir / "command-client.py"
    command_client.write_text(
        "import base64,json,socket,sys\n"
        "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.connect('/run/posttrainbench0/command-broker.sock')\n"
        "s.sendall(json.dumps({'argv':sys.argv[1:]}).encode()+b'\\n'); data=b''\n"
        "while not data.endswith(b'\\n'):\n data += s.recv(65536)\n"
        "r=json.loads(data); sys.stdout.buffer.write(base64.b64decode(r['output'])); raise SystemExit(r['returncode'])\n",
        encoding="utf-8",
    )
    wrapper = private / "command-wrapper"
    wrapper.write_text("#!/usr/bin/env sh\nexec /usr/bin/python3 /run/posttrainbench0/command-client.py \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o700)
    minimal_etc: list[str] = ["--dir", "/etc"]
    for path in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/ssl/certs/ca-certificates.crt"):
        if Path(path).is_file():
            minimal_etc.extend(("--ro-bind", path, path))
    runtime_root = cli_path.parent
    control_mount = f"/run/{harness}"
    command = [
        _bwrap(), "--unshare-all", "--share-net", "--die-with-parent", "--new-session",
        *_system_binds(), *minimal_etc,
        "--ro-bind", str(wrapper), "/bin/bash",
        *(("--ro-bind", str(wrapper), "/usr/bin/bash") if harness == "codex" else ()),
        "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--dir", "/home",
        "--bind", str(workspace), "/home/agent",
        "--dir", "/models", "--ro-bind", str(base_model), "/models/base",
        "--dir", "/run", "--ro-bind", str(control_dir), "/run/posttrainbench0",
        "--bind", str(control_home), control_mount,
        "--dir", "/opt", "--dir", "/opt/posttrainbench0",
        "--ro-bind", "/usr/bin/bash", "/opt/posttrainbench0/control-bash",
        "--ro-bind", str(runtime_root), "/opt/agent-cli",
        "--setenv", "HOME", control_mount,
        "--setenv", "SHELL", "/bin/bash",
        "--setenv", "PATH", "/opt/agent-cli:/home/agent/bin:/usr/local/bin:/usr/bin:/bin",
        "--setenv", "PTB0_AGENT_HOME", "/home/agent",
        "--setenv", "PTB0_PROMPT_PATH", "/home/agent/prompt.txt",
        "--setenv", "PTB0_CLI_PATH", f"/opt/agent-cli/{cli_path.name}",
        "--setenv", "PTB0_CONTROL_BASH", "/opt/posttrainbench0/control-bash",
        "--chdir", "/home/agent", "/opt/posttrainbench0/control-bash",
    ]
    launcher = private / "agent-harness"
    launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + " ".join(shlex.quote(item) for item in command) + ' "$@"\n', encoding="utf-8")
    launcher.chmod(0o700)
    return launcher
