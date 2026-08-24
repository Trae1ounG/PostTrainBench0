from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol


@dataclass(frozen=True)
class NoiseTerm:
    """One deterministic Gaussian direction multiplied by ``scale``."""

    seed: int
    scale: float
    target: str = "all"


@dataclass(frozen=True)
class Candidate:
    """A single model represented as base weights plus deterministic noise terms."""

    candidate_id: str
    terms: tuple[NoiseTerm, ...] = ()
    parent_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def perturbed(
        self,
        *,
        candidate_id: str,
        seed: int,
        scale: float,
        target: str = "all",
        metadata: Mapping[str, object] | None = None,
    ) -> "Candidate":
        return Candidate(
            candidate_id=candidate_id,
            terms=(*self.terms, NoiseTerm(seed=seed, scale=scale, target=target)),
            parent_id=self.candidate_id,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class Evaluation:
    candidate_id: str
    score: float
    metrics: Mapping[str, float] = field(default_factory=dict)


class CandidateBackend(Protocol):
    """The only surface a search method needs from model execution."""

    def evaluate(self, candidate: Candidate, *, split: str = "target") -> Evaluation:
        """Evaluate exactly one candidate and restore the backend afterwards."""

    def materialize(self, candidate: Candidate, output_dir: Path) -> None:
        """Write exactly one loadable model artifact."""
