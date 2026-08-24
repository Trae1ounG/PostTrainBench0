from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .contracts import Candidate, NoiseTerm


NOISE_PROGRAM_FORMAT = "zerograd-noise-program-v1"


@dataclass(frozen=True)
class LoadedCandidate:
    source_path: Path
    candidate: Candidate
    raw: dict


def resolve_workspace_file(workspace: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("candidate path must be relative to the agent workspace")
    unresolved = workspace / candidate
    current = workspace
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("candidate path must not contain symbolic links")
    resolved_workspace = workspace.resolve()
    try:
        resolved = unresolved.resolve(strict=True)
    except FileNotFoundError:
        raise ValueError(f"candidate file does not exist: {relative_path!r}") from None
    try:
        resolved.relative_to(resolved_workspace)
    except ValueError as exc:
        raise ValueError("candidate path escapes the agent workspace") from exc
    if not resolved.is_file():
        raise ValueError("candidate path must name a regular JSON file")
    return resolved


def load_candidate(workspace: Path, relative_path: str) -> LoadedCandidate:
    path = resolve_workspace_file(workspace, relative_path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("candidate manifest exceeds 2 MiB")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format") != NOISE_PROGRAM_FORMAT:
        raise ValueError(f"candidate format must be {NOISE_PROGRAM_FORMAT!r}")
    terms_raw = raw.get("terms")
    if not isinstance(terms_raw, list):
        raise ValueError("candidate terms must be a list")

    terms = []
    for index, term in enumerate(terms_raw):
        if not isinstance(term, dict):
            raise ValueError(f"term {index} must be an object")
        seed = term.get("seed")
        scale = term.get("scale")
        target = term.get("target", "all")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError(f"term {index} seed must be an integer in [0, 2^63)")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError(f"term {index} scale must be a number")
        scale = float(scale)
        if not math.isfinite(scale):
            raise ValueError(f"term {index} scale must be finite")
        if not isinstance(target, str) or not target:
            raise ValueError(f"term {index} target must be a non-empty string")
        terms.append(NoiseTerm(seed=seed, scale=scale, target=target))

    candidate_id = raw.get("candidate_id", path.stem)
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id must be a non-empty string")
    return LoadedCandidate(
        source_path=path,
        candidate=Candidate(candidate_id=candidate_id, terms=tuple(terms)),
        raw=raw,
    )
