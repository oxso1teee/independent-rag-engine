"""What the commands need from the outside world, as plain types.

The command line knows nothing about SQLite, the clock or the filesystem: it receives a
:class:`Container` and asks it for :class:`Services`. The composition root
(:mod:`rag_engine.container`) is the only place that builds the real ones.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_engine.ingestion.identity import SourceRegistry
from rag_engine.ingestion.jobs import JobQueue
from rag_engine.ingestion.service import IngestService
from rag_engine.ports import Store


@dataclass(frozen=True, slots=True)
class Services:
    store: Store
    db_path: Path
    registry: SourceRegistry
    ingest: IngestService
    queue: JobQueue


class Container(Protocol):
    def open(
        self, config: Path | None, db: Path | None, *, read_only: bool
    ) -> AbstractContextManager[Services]:
        """Context-managed services; the store is closed when the block ends."""
        ...
