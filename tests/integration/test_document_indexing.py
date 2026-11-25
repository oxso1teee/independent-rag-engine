"""The ingestion worker publishes a parsed document in one committed generation."""

from __future__ import annotations

from pathlib import Path

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.config import ParsingSettings, Settings
from rag_engine.container import build_processing, filesystem_adapter
from rag_engine.domain import EmbeddingModel, IdFactory, JobKind, JobStatus, SourceKey
from rag_engine.embedding.service import EmbeddingService
from rag_engine.ingestion.identity import SourceRegistry
from rag_engine.ingestion.indexing import DocumentIndexingHandler
from rag_engine.ingestion.jobs import JobQueue, Worker
from rag_engine.ingestion.service import IngestService
from rag_engine.storage.sqlite import SqliteStore
from rag_engine.testing import FakeClock, HashingEmbedder
from tests.factories import T0


def test_document_job_parses_embeds_and_publishes(
    tmp_path: Path, store: SqliteStore, clock: FakeClock, ids: IdFactory
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("# Heading\n\nA paragraph about indexing and storage.\n")
    source = SourceRegistry(store, clock, ids, filesystem_adapter).add("notes", root)
    assert IngestService(store, clock, ids, filesystem_adapter).run(source.name).jobs_enqueued == 1
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
    settings = Settings(parsing=ParsingSettings(isolate=()))
    processing = build_processing(settings, clock, ids)
    embed = EmbeddingService(provider, model, HeuristicTokenCounter(), clock)
    handler = DocumentIndexingHandler(
        store,
        clock,
        ids,
        filesystem_adapter,
        lambda cleaners: processing.build(cleaners, None),
        embed,
        model,
        processing.limits,
        processing.policy,
    )
    outcome = Worker(JobQueue(store, clock, ids), handler, JobKind.INGEST_DOC).run_once()
    assert outcome is not None and outcome.status is JobStatus.DONE
    document = store.reader().get_document_by_key(source.id, SourceKey("a.md"))
    assert document is not None and document.current_version_id is not None
    chunks = store.reader().list_chunks(document.current_version_id)
    assert chunks and any(chunk.chunk.indexed for chunk in chunks)
    assert store.reader().get_index_metadata().index_generation == 1  # type: ignore[union-attr]

    # A source that changes after its scan cannot replace the last working version.
    (root / "a.md").write_text("# Heading\n\nA second, much longer paragraph.\n")
    assert IngestService(store, clock, ids, filesystem_adapter).run(source.name).jobs_enqueued == 1
    (root / "a.md").write_text("# Heading\n\nChanged again after the scan.\n")
    failed = Worker(JobQueue(store, clock, ids), handler, JobKind.INGEST_DOC).run_once()
    assert failed is not None and failed.status is JobStatus.DEAD
    retained = store.reader().get_document(document.id)
    assert retained is not None and retained.current_version_id == document.current_version_id
    assert store.reader().get_index_metadata().index_generation == 1  # type: ignore[union-attr]
