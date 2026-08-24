from __future__ import annotations

import argparse
import json
import os
import socket
import sys


DEFAULT_SOCKET = "/run/posttrainbench0/evaluator.sock"


def call(request: dict, socket_path: str | None = None) -> dict:
    path = socket_path or os.environ.get("PTB0_EVALUATOR_SOCKET", DEFAULT_SOCKET)
    payload = json.dumps(request, separators=(",", ":")).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(path)
        connection.sendall(payload)
        received = bytearray()
        while not received.endswith(b"\n"):
            chunk = connection.recv(65536)
            if not chunk:
                break
            received.extend(chunk)
    if not received:
        raise RuntimeError("evaluator returned no response")
    response = json.loads(received)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "evaluator request failed"))
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("status", "results"):
        subparsers.add_parser(operation)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("candidate")
    evaluate.add_argument("tasks", nargs="*")
    batch = subparsers.add_parser("evaluate-batch")
    batch.add_argument("candidates", nargs="+")
    batch.add_argument("--tasks", nargs="*", default=[])
    args = parser.parse_args()
    if args.operation == "evaluate":
        request = {"operation": "evaluate", "candidate": args.candidate, "tasks": args.tasks}
    elif args.operation == "evaluate-batch":
        request = {"operation": "evaluate_batch", "candidates": args.candidates, "tasks": args.tasks}
    else:
        request = {"operation": args.operation}
    try:
        response = call(request)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(response, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
