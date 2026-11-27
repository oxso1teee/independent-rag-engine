"""What generation produces: the answer's citations and the verdict on them.

The models only state what a consistent result looks like. Whether an answer is grounded is decided
by ``generation``; whether a marker exists is decided against the ``Context`` it was generated from,
so a citation can never point at a source that was not in front of the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rag_engine.domain.context import EvidenceLocator
from rag_engine.domain.errors import require
from rag_engine.domain.ids import ChunkId, DocumentId, DocumentVersionId


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    REFUSED = "refused"


class CitationProblem(StrEnum):
    UNKNOWN_MARKER = "unknown_marker"
    NO_CITATION = "no_citation"
    UNGROUNDED = "ungrounded"


@dataclass(frozen=True, slots=True)
class Citation:
    """A marker of the answer resolved to the evidence it names."""

    evidence_id: str
    chunk_id: ChunkId
    document_id: DocumentId
    version_id: DocumentVersionId
    locator: EvidenceLocator
    uses: int

    def __post_init__(self) -> None:
        require(self.uses > 0, "a citation is used at least once")


@dataclass(frozen=True, slots=True)
class SentenceCheck:
    """Whether one cited sentence is supported by the evidence it cites.

    ``coverage`` is the share of the sentence's content terms found in the cited evidence;
    ``None`` means the sentence cannot be judged lexically (too few terms, or written in another
    script than its evidence), which is reported and never counted as a failure.
    """

    text: str
    evidence_ids: tuple[str, ...]
    coverage: float | None
    grounded: bool

    def __post_init__(self) -> None:
        require(len(self.evidence_ids) > 0, "a checked sentence cites at least one passage")
        require(
            self.coverage is None or 0.0 <= self.coverage <= 1.0,
            "coverage is a share between 0 and 1",
        )
        require(self.coverage is not None or self.grounded, "an unjudged sentence is not a failure")


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """Verdict of the citation validator for one answer."""

    cited: tuple[str, ...]
    unknown: tuple[str, ...]
    uncited_sentences: tuple[str, ...]
    sentences: tuple[SentenceCheck, ...]
    problems: tuple[CitationProblem, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def unjudged(self) -> int:
        return sum(item.coverage is None for item in self.sentences)
