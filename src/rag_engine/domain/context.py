"""Evidence, the context handed to generation, and the report that accounts for every chunk.

Nothing here knows how a context is built. The models only state what a valid one looks like:
every piece of evidence carries the provenance a citation needs, the budget adds up, and the
report never claims to have used more tokens than it was given.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from rag_engine.domain.errors import require
from rag_engine.domain.ids import ChunkId, DocumentId, DocumentVersionId

_EVIDENCE_ID = re.compile(r"E[1-9][0-9]*")


class DropReason(StrEnum):
    DUPLICATE = "duplicate"
    BUDGET = "budget"
    DOC_CAP = "doc_cap"
    OVERLAP = "overlap"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class EvidenceLocator:
    """Where a piece of evidence comes from; enough to show and to verify a citation."""

    source_name: str
    source_key: str
    title: str | None
    heading_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        require(0 <= self.char_start < self.char_end, "invalid evidence span")
        require(
            (self.page_start is None) == (self.page_end is None)
            and (self.page_start is None or 0 < self.page_start <= (self.page_end or 0)),
            "invalid evidence page range",
        )


@dataclass(frozen=True, slots=True)
class Evidence:
    """One numbered passage of the context.

    ``chunk_id`` is the stored chunk whose text is shown (a section when small-to-big expanded a
    hit); ``matched`` lists the retrieved chunks it stands for.
    """

    evidence_id: str
    chunk_id: ChunkId
    matched: tuple[ChunkId, ...]
    document_id: DocumentId
    version_id: DocumentVersionId
    text: str
    truncated: bool
    locator: EvidenceLocator
    score: float
    tokens: int

    def __post_init__(self) -> None:
        require(_EVIDENCE_ID.fullmatch(self.evidence_id) is not None, "invalid evidence id")
        require(bool(self.text.strip()), "evidence text must not be empty")
        require(self.tokens > 0 and math.isfinite(self.score), "invalid evidence tokens or score")
        require(len(self.matched) > 0, "evidence must stand for at least one retrieved chunk")


@dataclass(frozen=True, slots=True)
class ContextBudget:
    window: int
    system: int
    question: int
    max_new: int
    margin: int
    evidence_budget: int

    def __post_init__(self) -> None:
        require(self.window > 0, "the context window must be positive")
        require(min(self.system, self.question, self.max_new, self.margin) >= 0, "negative budget")
        require(
            0
            < self.evidence_budget
            == self.window - self.system - self.question - self.max_new - self.margin,
            "the evidence budget does not add up",
        )


@dataclass(frozen=True, slots=True)
class IncludedChunk:
    chunk_id: ChunkId
    evidence_id: str
    tokens: int


@dataclass(frozen=True, slots=True)
class DroppedChunk:
    chunk_id: ChunkId
    reason: DropReason


@dataclass(frozen=True, slots=True)
class MergedChunk:
    """A retrieved chunk whose text was folded into the evidence of an overlapping neighbour."""

    chunk_id: ChunkId
    into: ChunkId


@dataclass(frozen=True, slots=True)
class ContextReport:
    included: tuple[IncludedChunk, ...]
    dropped: tuple[DroppedChunk, ...]
    merged: tuple[MergedChunk, ...]
    truncated: tuple[ChunkId, ...]
    tokens_used: int
    budget: int
    context_hash: str

    def __post_init__(self) -> None:
        require(0 <= self.tokens_used <= self.budget, "the context used more than its budget")
        require(len(self.context_hash) == 64, "the context hash is a SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class Context:
    evidence: tuple[Evidence, ...]
    budget: ContextBudget
    tokens_used: int
    report: ContextReport
    context_hash: str

    def __post_init__(self) -> None:
        require(self.tokens_used <= self.budget.evidence_budget, "the context exceeds its budget")
        require(
            [item.evidence_id for item in self.evidence]
            == [f"E{n}" for n in range(1, len(self.evidence) + 1)],
            "evidence ids must be E1..En in order",
        )
