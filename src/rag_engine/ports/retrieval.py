"""Ports of the retrieval pipeline that read published chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag_engine.domain import ChunkId, ChunkRef, RetrievalFilters, TextHash


@dataclass(frozen=True, slots=True)
class ChunkInfo:
    """What the pipeline needs to know about a published chunk without loading its text."""

    ref: ChunkRef
    text_hash: TextHash


@runtime_checkable
class ChunkFilter(Protocol):
    def resolve(self, filters: RetrievalFilters) -> frozenset[ChunkId] | None:
        """Ids of the searchable chunks that satisfy every filter (all of them must hold).

        ``None`` means "no restriction" (no filter was given), which is not the same as an
        empty set, a filter that matched nothing. Only chunks of active documents' current
        versions that are indexed and not duplicates are ever returned.
        """
        ...


@runtime_checkable
class ChunkCatalog(Protocol):
    def describe(self, ids: Sequence[ChunkId]) -> Mapping[ChunkId, ChunkInfo]:
        """Identity of each requested chunk; unknown ids are absent from the result."""
        ...


@dataclass(frozen=True, slots=True)
class ChunkContent:
    """A published chunk with everything needed to show it as evidence and to cite it."""

    ref: ChunkRef
    text: str
    token_count: int
    level: int
    parent_id: ChunkId | None
    ordinal: int
    char_start: int
    char_end: int
    heading_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    source_name: str
    source_key: str
    title: str | None


@runtime_checkable
class ChunkContentReader(Protocol):
    def read(self, ids: Sequence[ChunkId]) -> Mapping[ChunkId, ChunkContent]:
        """Content of each requested chunk of any level; unknown ids are absent."""
        ...
