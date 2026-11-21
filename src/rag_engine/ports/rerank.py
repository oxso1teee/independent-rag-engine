"""Reranker port."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_engine.domain import ChunkRef, require


@dataclass(frozen=True, slots=True)
class Candidate:
    chunk_ref: ChunkRef
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    candidate: Candidate
    score: float
    """Raw model logit; not a probability and not comparable across models."""

    def __post_init__(self) -> None:
        require(math.isfinite(self.score), "rerank scores must be finite")


@dataclass(frozen=True, slots=True)
class RerankBudget:
    pool_size: int = 30
    timeout_s: float = 10.0

    def __post_init__(self) -> None:
        require(self.pool_size >= 1 and self.timeout_s > 0, "invalid rerank budget")


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[Candidate], budget: RerankBudget
    ) -> Sequence[RerankedCandidate]:
        """Return at most ``budget.pool_size`` candidates ordered by descending score."""
        ...
