"""Composition root: the one place that reads configuration and creates adapters.

``rag`` starts here (:func:`main`): settings are loaded, the SQLite store, the real clock,
identifier entropy and the filesystem adapter are built, and the command line is handed a
container that produces them. Everything below this module receives its collaborators
through constructors.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from rag_engine.chunking.registry import ChunkerRegistry, build_counter
from rag_engine.cli import main as cli_main
from rag_engine.cli.context import Container, Processing, Services
from rag_engine.config import IngestSettings, ParsingSettings, Settings, load_settings
from rag_engine.config.settings import EmbeddingSettings
from rag_engine.domain import (
    ChunkPolicy,
    EmbeddingModel,
    EmbeddingModelId,
    ErrorCode,
    IdFactory,
    JobKind,
    RagError,
    Source,
)
from rag_engine.embedding.ollama import OllamaEmbedder
from rag_engine.embedding.onnx import OnnxEmbedder
from rag_engine.embedding.service import EmbeddingService
from rag_engine.ingestion.identity import AdapterFactory, SourceRegistry
from rag_engine.ingestion.indexing import DocumentIndexingHandler
from rag_engine.ingestion.jobs import JobQueue, RetryPolicy, Worker
from rag_engine.ingestion.processing import DocumentProcessor
from rag_engine.ingestion.service import IngestService
from rag_engine.ingestion.sources import FilesystemSource
from rag_engine.observability import configure_logging
from rag_engine.parsing.cleaners import compile_cleaners
from rag_engine.parsing.isolation import ParserWorker
from rag_engine.parsing.registry import default_parsers
from rag_engine.ports import Clock, EmbeddingProvider, ParseLimits, SourceAdapter, Store
from rag_engine.storage.sqlite import open_store, open_store_read_only
from rag_engine.system import SystemClock, system_id_factory


def filesystem_adapter(source: Source) -> SourceAdapter:
    if source.locator is None:
        raise RagError(ErrorCode.E_CONFIG_INVALID, "only filesystem sources can be ingested")
    return FilesystemSource(Path(source.locator), source.policy)


@dataclass(slots=True)
class BuiltEmbedding:
    model: EmbeddingModel
    provider: EmbeddingProvider

    async def aclose(self) -> None:
        if isinstance(self.provider, OllamaEmbedder):
            await self.provider.aclose()
        elif isinstance(self.provider, OnnxEmbedder):
            self.provider.close()


def build_embedding(settings: EmbeddingSettings, clock: Clock) -> BuiltEmbedding:
    """Construct a pinned provider; ONNX verifies its local digest on load."""
    if not settings.enabled or settings.provider is None or settings.model is None:
        raise RagError(ErrorCode.E_CONFIG_INVALID, "embedding is not configured")
    if settings.revision is None or settings.dimension is None:
        raise RagError(ErrorCode.E_CONFIG_INVALID, "embedding identity is incomplete")
    model = EmbeddingModel(
        id=EmbeddingModelId.build(
            provider=settings.provider,
            name=settings.model,
            revision=settings.revision,
            dimension=settings.dimension,
            normalized=True,
            query_prefix=settings.query_prefix,
            document_prefix=settings.document_prefix,
        ),
        provider=settings.provider,
        name=settings.model,
        revision=settings.revision,
        dimension=settings.dimension,
        normalized=True,
        query_prefix=settings.query_prefix,
        document_prefix=settings.document_prefix,
        max_input_tokens=settings.max_input_tokens,
        created_at=clock.now(),
    )
    if settings.provider == "ollama":
        provider: EmbeddingProvider = OllamaEmbedder(
            model, base_url=settings.base_url, timeout_s=settings.timeout_s
        )
    elif settings.provider == "onnx":
        if settings.model_dir is None:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "embedding.model_dir is required")
        provider = OnnxEmbedder(model, settings.model_dir)
    else:
        raise RagError(ErrorCode.E_CONFIG_INVALID, "unsupported embedding provider")
    return BuiltEmbedding(model, provider)


def retry_policy(settings: IngestSettings) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=settings.max_attempts,
        base_delay_s=settings.retry_base_delay_s,
        factor=settings.retry_factor,
        max_delay_s=settings.retry_max_delay_s,
        jitter=settings.retry_jitter,
        lease_s=settings.lease_s,
    )


def parse_limits(settings: ParsingSettings) -> ParseLimits:
    return ParseLimits(
        max_text_chars=settings.max_text_chars,
        max_blocks=settings.max_blocks,
        max_pages=settings.max_pages,
        max_archive_bytes=settings.max_archive_mb * 1024 * 1024,
        max_archive_ratio=settings.max_archive_ratio,
        max_archive_entries=settings.max_archive_entries,
        max_nesting=settings.max_nesting,
        timeout_s=settings.timeout_s,
    )


def build_processing(settings: Settings, clock: Clock, ids: IdFactory) -> Processing:
    """Parsing and chunking as configured; the token counter is created once."""
    chunking = settings.chunking
    counter = build_counter(chunking.counter, chunking.tokenizer_json)
    policy = ChunkPolicy(
        target_tokens=chunking.target_tokens,
        max_tokens=chunking.max_tokens,
        min_tokens=chunking.min_tokens,
        overlap_ratio=chunking.overlap_ratio,
    )

    parsing = settings.parsing
    worker = ParserWorker(parsing.isolation_memory_mb) if parsing.isolate else None

    def build(cleaners: Sequence[str], strategy: str | None) -> DocumentProcessor:
        parsers = default_parsers(
            compile_cleaners(cleaners) if cleaners else None,
            isolate=parsing.isolate,
            worker=worker,
        )
        return DocumentProcessor(
            parsers, clock, ChunkerRegistry(counter, strategy or chunking.strategy)
        )

    return Processing(ids, policy, parse_limits(settings.parsing), build)


def build_services(
    store: Store,
    db_path: Path,
    clock: Clock,
    ids: IdFactory,
    adapter_factory: AdapterFactory,
    policy: RetryPolicy | None = None,
    rng: Callable[[], float] | None = None,
) -> Services:
    """Wire the services around one store; tests pass fakes for clock, ids and randomness."""
    return Services(
        store=store,
        db_path=db_path,
        registry=SourceRegistry(store, clock, ids, adapter_factory),
        ingest=IngestService(store, clock, ids, adapter_factory),
        queue=JobQueue(store, clock, ids, policy, rng)
        if rng
        else JobQueue(store, clock, ids, policy),
    )


class DefaultContainer:
    """Real clock, real entropy, SQLite at the configured (or overridden) path."""

    @contextmanager
    def open(self, config: Path | None, db: Path | None, *, read_only: bool) -> Iterator[Services]:
        settings = load_settings(config)
        configure_logging(settings.logging.level, json_output=settings.logging.json_output)
        path = db if db is not None else settings.storage.db_path
        clock = SystemClock()
        store = (
            open_store_read_only(path)
            if read_only
            else open_store(path, clock, writer_timeout_s=5.0)
        )
        built = None
        try:
            services = build_services(
                store,
                path,
                clock,
                system_id_factory(),
                filesystem_adapter,
                retry_policy(settings.ingest),
                random.SystemRandom().random,
            )
            if not read_only and settings.embedding.enabled:
                built = build_embedding(settings.embedding, clock)
                processing = build_processing(settings, clock, system_id_factory())
                limits = settings.effective_limits()
                service = EmbeddingService(
                    built.provider,
                    built.model,
                    build_counter(settings.chunking.counter, settings.chunking.tokenizer_json),
                    clock,
                    batch_size=limits.embed_batch,
                    batch_tokens=settings.embedding.batch_tokens,
                    concurrency=limits.embed_concurrency,
                    timeout_s=settings.embedding.timeout_s,
                )
                handler = DocumentIndexingHandler(
                    store,
                    clock,
                    processing.ids,
                    filesystem_adapter,
                    lambda cleaners: processing.build(cleaners, None),
                    service,
                    built.model,
                    processing.limits,
                    processing.policy,
                )
                services = Services(
                    store=services.store,
                    db_path=services.db_path,
                    registry=services.registry,
                    ingest=services.ingest,
                    queue=services.queue,
                    index_worker=Worker(services.queue, handler, JobKind.INGEST_DOC),
                )
            yield services
        finally:
            if built is not None:
                asyncio.run(built.aclose())
            store.close()

    def processing(self, config: Path | None) -> Processing:
        settings = load_settings(config)
        configure_logging(settings.logging.level, json_output=settings.logging.json_output)
        return build_processing(settings, SystemClock(), system_id_factory())


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``rag`` command; returns the process exit code."""
    container: Container = DefaultContainer()
    return cli_main(argv, container=container)
