"""Collapsing duplicates and capping how much one document can fill a result.

Candidates arrive best first and leave in the same order. An exact repeat of a text (same
``text_hash``) is dropped after its first occurrence, wherever it lives, and a document may hold
at most ``max_per_document`` places. Both rules are cheap, deterministic stand-ins for
heavier diversification; the counts are reported so a result can say what was removed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag_engine.domain import ChunkId, DocumentId, TextHash
from rag_engine.ports import ChunkInfo
from rag_engine.retrieval.fusion import FusedCandidate


@dataclass(frozen=True, slots=True)
class DiversityOutcome:
    kept: tuple[tuple[FusedCandidate, ChunkInfo], ...]
    collapsed_duplicates: int
    capped: int
    unknown: int

    @property
    def document_share(self) -> float:
        """Unique documents over kept chunks: 1.0 means no document appears twice."""
        if not self.kept:
            return 0.0
        return len({info.ref.document_id for _, info in self.kept}) / len(self.kept)


def diversify(
    candidates: Sequence[FusedCandidate],
    catalog: Mapping[ChunkId, ChunkInfo],
    *,
    top_k: int,
    max_per_document: int | None = None,
) -> DiversityOutcome:
    if top_k < 1 or (max_per_document is not None and max_per_document < 1):
        raise ValueError("top_k and max_per_document must be positive")
    kept: list[tuple[FusedCandidate, ChunkInfo]] = []
    seen_texts: set[TextHash] = set()
    per_document: dict[DocumentId, int] = {}
    collapsed = capped = unknown = 0
    for candidate in candidates:
        if len(kept) == top_k:
            break
        info = catalog.get(candidate.chunk_id)
        if info is None:
            unknown += 1
            continue
        if info.text_hash in seen_texts:
            collapsed += 1
            continue
        taken = per_document.get(info.ref.document_id, 0)
        if max_per_document is not None and taken >= max_per_document:
            capped += 1
            continue
        seen_texts.add(info.text_hash)
        per_document[info.ref.document_id] = taken + 1
        kept.append((candidate, info))
    return DiversityOutcome(tuple(kept), collapsed, capped, unknown)
