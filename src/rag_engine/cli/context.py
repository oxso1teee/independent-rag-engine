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

from rag_engine.context.builder import ContextBuilder, ContextPolicy
from rag_engine.domain import ChunkPolicy, EmbeddingModel, IdFactory, RetrievalMode
from rag_engine.embedding.service import EmbeddingService
from rag_engine.ingestion.identity import SourceRegistry
from rag_engine.ingestion.jobs import JobQueue, Worker
from rag_engine.ingestion.maintenance import IndexMaintenance
from rag_engine.ingestion.processing import DocumentProcessor
from rag_engine.ingestion.service import IngestService
from rag_engine.pipeline import RagPipeline
from rag_engine.ports import ParseLimits, Store, TokenCounter
from rag_engine.retrieval.service import RetrievalService
from rag_engine.retrieval.sufficiency import SufficiencyGate


@dataclass(frozen=True, slots=True)
class Services:
    store: Store
    db_path: Path
    registry: SourceRegistry
    ingest: IngestService
    queue: JobQueue
    index_worker: Worker | None = None
    maintenance: IndexMaintenance | None = None
    embedding_model: EmbeddingModel | None = None
    embedding_service: EmbeddingService | None = None
    keep_versions: int = 3


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


@dataclass(frozen=True, slots=True)
class SearchDefaults:
    mode: RetrievalMode
    top_k: int
    candidate_k: int


@dataclass(frozen=True, slots=True)
class SearchServices:
    """A read-only view of the index: the search pipeline, the store for display, defaults."""

    retrieval: RetrievalService
    store: Store
    ids: IdFactory
    defaults: SearchDefaults
    context: ContextBuilder
    counter: TokenCounter
    context_policy: ContextPolicy
    llm_window: int | None
    llm_max_new: int
    gate: Callable[[], SufficiencyGate] = lambda: SufficiencyGate(None, None)
    """The sufficiency gate for the index as it is now (its calibration is read each time)."""
    gate_environment: Callable[[], str] = lambda: ""
    """The setup besides the embedding model that a gate calibration is bound to."""
    release_models: Callable[[], None] = lambda: None
    """Drops the in-process embedder and reranker; the services are unusable for search after."""


@dataclass(frozen=True, slots=True)
class AskServices:
    """A configured query pipeline over the published index, for one command."""

    pipeline: RagPipeline
    ids: IdFactory
    defaults: SearchDefaults
    search: SearchServices


class Container(Protocol):
    def open(
        self, config: Path | None, db: Path | None, *, read_only: bool
    ) -> AbstractContextManager[Services]:
        """Context-managed services; the store is closed when the block ends."""
        ...

    def processing(self, config: Path | None) -> Processing:
        """Parsing and chunking configured from ``config``; needs no database."""
        ...

    def search(
        self, config: Path | None, db: Path | None
    ) -> AbstractContextManager[SearchServices]:
        """Search services over the published index; nothing is written."""
        ...

    def ask(
        self, config: Path | None, db: Path | None, *, release_models: bool = True
    ) -> AbstractContextManager[AskServices]:
        """The query pipeline over the published index; needs ``[llm]`` and an index.

        ``release_models`` (one-shot use) hands the in-process models' memory back before a local
        model is called; a service that stays up passes ``False``.
        """
        ...

    def serve(
        self, config: Path | None, db: Path | None, *, host: str | None, port: int | None
    ) -> int:
        """Run the HTTP API until it is stopped; returns the process exit code."""
        ...
