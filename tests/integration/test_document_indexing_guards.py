"""The indexing handler refuses stale, moved or half-observed work before it publishes."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.config import ParsingSettings, Settings
from rag_engine.container import build_processing, filesystem_adapter
from rag_engine.domain import (
    EmbeddingModel,
    ErrorCode,
    IdFactory,
    IngestionJob,
    JobKind,
    RagError,
    Source,
)
from rag_engine.embedding.service import EmbeddingService
from rag_engine.ingestion.identity import SourceRegistry
from rag_engine.ingestion.indexing import DocumentIndexingHandler
from rag_engine.ingestion.jobs import JobQueue
from rag_engine.ingestion.service import IngestService
from rag_engine.ingestion.sources import FilesystemSource
from rag_engine.ports import SourceAdapter
from rag_engine.storage.sqlite import SqliteStore
from rag_engine.testing import FakeClock, HashingEmbedder
from tests.factories import T0


@dataclass
class Setup:
    handler: DocumentIndexingHandler
    job: IngestionJob
    root: Path
    database: Path
    store: SqliteStore
    build: Callable[[Callable[[Source], SourceAdapter]], DocumentIndexingHandler]


@pytest.fixture
def setup(tmp_path: Path, store: SqliteStore, clock: FakeClock, ids: IdFactory) -> Setup:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("# Heading\n\nA paragraph about indexing and storage.\n")
    source = SourceRegistry(store, clock, ids, filesystem_adapter).add("notes", root)
    IngestService(store, clock, ids, filesystem_adapter).run(source.name)
    provider = HashingEmbedder(32)
    model = EmbeddingModel(
        id=provider.model_id,
        provider="fake",
        name="hashing",
        revision="1",
        dimension=32,
        normalized=True,
        max_input_tokens=provider.max_input_tokens,
        created_at=T0,
    )
    processing = build_processing(Settings(parsing=ParsingSettings(isolate=())), clock, ids)

    def build(factory: Callable[[Source], SourceAdapter]) -> DocumentIndexingHandler:
        return DocumentIndexingHandler(
            store,
            clock,
            ids,
            factory,
            lambda cleaners: processing.build(cleaners, None),
            EmbeddingService(provider, model, HeuristicTokenCounter(), clock),
            model,
            processing.limits,
            processing.policy,
        )

    job = JobQueue(store, clock, ids).claim()
    assert job is not None and job.kind is JobKind.INGEST_DOC
    return Setup(build(filesystem_adapter), job, root, tmp_path / "rag.db", store, build)


def sql(setup: Setup, statement: str) -> None:
    with closing(sqlite3.connect(setup.database)) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(statement)
        connection.commit()


def run(setup: Setup, heartbeat: Callable[[], bool] = lambda: True) -> RagError:
    with pytest.raises(RagError) as info:
        setup.handler(setup.job, heartbeat)
    return info.value


def test_a_document_that_is_no_longer_active_is_refused(setup: Setup) -> None:
    sql(setup, "UPDATE documents SET status = 'deleted'")
    assert run(setup).code is ErrorCode.E_INTEGRITY


def test_a_document_without_an_observed_revision_is_refused(setup: Setup) -> None:
    sql(setup, "UPDATE sync_state SET last_content_hash = NULL")
    assert run(setup).code is ErrorCode.E_INTEGRITY


def test_a_moved_source_is_refused(setup: Setup, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    setup.handler = setup.build(lambda source: FilesystemSource(elsewhere, source.policy))
    assert run(setup).code is ErrorCode.E_SOURCE_LOCATOR_CHANGED


def test_a_lost_lease_before_parsing_publishes_nothing(setup: Setup) -> None:
    error = run(setup, lambda: False)
    assert error.code is ErrorCode.E_TIMEOUT
    metadata = setup.store.reader().get_index_metadata()
    assert metadata is not None and metadata.index_generation == 0


def test_a_lease_lost_while_embedding_publishes_nothing(setup: Setup) -> None:
    answers = iter([True, False])
    error = run(setup, lambda: next(answers))
    assert error.code is ErrorCode.E_TIMEOUT
    metadata = setup.store.reader().get_index_metadata()
    assert metadata is not None and metadata.index_generation == 0
