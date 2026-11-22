"""What the commands need from the outside world, as plain types.

The command line knows nothing about SQLite, the clock or the filesystem: it receives a
:class:`Container` and asks it for :class:`Services`. The composition root
(:mod:`rag_engine.container`) is the only place that builds the real ones.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_engine.domain import ChunkPolicy, IdFactory
from rag_engine.ingestion.identity import SourceRegistry
from rag_engine.ingestion.jobs import JobQueue
from rag_engine.ingestion.processing import DocumentProcessor
from rag_engine.ingestion.service import IngestService
from rag_engine.ports import ParseLimits, Store


@dataclass(frozen=True, slots=True)
class Services:
    store: Store
    db_path: Path
    registry: SourceRegistry
    ingest: IngestService
    queue: JobQueue


@dataclass(frozen=True, slots=True)
class Processing:
    """Everything needed to parse and chunk a document, with no database involved.

    ``build(cleaners, strategy)`` returns a processor for the given per-source cleaner patterns
    and, optionally, a chunking strategy that replaces the configured one.
    """

    ids: IdFactory
    policy: ChunkPolicy
    limits: ParseLimits
    build: Callable[[Sequence[str], str | None], DocumentProcessor]


class Container(Protocol):
    def open(
        self, config: Path | None, db: Path | None, *, read_only: bool
    ) -> AbstractContextManager[Services]:
        """Context-managed services; the store is closed when the block ends."""
        ...

    def processing(self, config: Path | None) -> Processing:
        """Parsing and chunking configured from ``config``; needs no database."""
        ...
