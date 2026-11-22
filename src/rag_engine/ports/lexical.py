"""Lexical (full-text) index port."""

from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_engine.domain import ChunkId, Language, ScoredChunk


@dataclass(frozen=True, slots=True)
class LexicalDocument:
    """One indexable chunk. ``language`` selects the analyser (stemmer) of the chunk.

    ``language`` is the last field with a default, so existing positional calls
    ``LexicalDocument(chunk_id, body, headings)`` keep working.
    """

    chunk_id: ChunkId
    body: str
    headings: str = ""
    language: Language = Language.UND


@runtime_checkable
class LexicalIndex(Protocol):
    def __len__(self) -> int: ...

    def upsert(self, documents: Sequence[LexicalDocument]) -> None: ...

    def remove(self, ids: Sequence[ChunkId]) -> None: ...

    def search(
        self, query: str, k: int, allowed: Set[ChunkId] | None = None
    ) -> Sequence[ScoredChunk]:
        """Top ``k`` by descending relevance; the raw query is never interpreted as syntax."""
        ...
