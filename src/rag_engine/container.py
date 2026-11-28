"""Composition root: the one place that reads configuration and creates adapters.

``rag`` starts here (:func:`main`): settings are loaded, the SQLite store, the real clock,
identifier entropy and the filesystem adapter are built, and the command line is handed a
container that produces them. Everything below this module receives its collaborators
through constructors.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import random
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rag_engine.chunking.registry import ChunkerRegistry, build_counter
from rag_engine.cli import main as cli_main
from rag_engine.cli.context import (
    AskServices,
    Container,
    Processing,
    SearchDefaults,
    SearchServices,
    Services,
)
from rag_engine.config import IngestSettings, ParsingSettings, Settings, load_settings
from rag_engine.config.settings import EmbeddingSettings, LlmSettings
from rag_engine.context.builder import ContextBuilder, ContextPolicy
from rag_engine.domain import (
    ChunkPolicy,
    EmbeddingModel,
    EmbeddingModelId,
    ErrorCode,
    IdFactory,
    JobKind,
    RagError,
    RetrievalMode,
    Source,
)
from rag_engine.embedding.cross_encoder import OnnxCrossEncoder
from rag_engine.embedding.ollama import OllamaEmbedder
from rag_engine.embedding.onnx import OnnxEmbedder
from rag_engine.embedding.service import EmbeddingService
from rag_engine.generation.generator import AnswerGenerator, GenerationPolicy
from rag_engine.generation.resilience import ResiliencePolicy, ResilientLLM
from rag_engine.ingestion.identity import AdapterFactory, SourceRegistry
from rag_engine.ingestion.indexing import DocumentIndexingHandler
from rag_engine.ingestion.jobs import JobQueue, RetryPolicy, Worker
from rag_engine.ingestion.maintenance import IndexMaintenance
from rag_engine.ingestion.processing import DocumentProcessor
from rag_engine.ingestion.service import IngestService
from rag_engine.ingestion.sources import FilesystemSource
from rag_engine.llm import OllamaChat, OpenAICompatChat
from rag_engine.observability import configure_logging
from rag_engine.parsing.cleaners import compile_cleaners
from rag_engine.parsing.isolation import ParserWorker
from rag_engine.parsing.registry import default_parsers
from rag_engine.pipeline import PipelineLimits, RagPipeline, StageShares
from rag_engine.ports import (
    Clock,
    EmbeddingProvider,
    ParseLimits,
    RerankBudget,
    SourceAdapter,
    Store,
    TokenCounter,
)
from rag_engine.retrieval.query import QueryProcessor
from rag_engine.retrieval.rerank import RerankStage
from rag_engine.retrieval.service import FusionSettings, IndexSnapshot, RetrievalService
from rag_engine.retrieval.sufficiency import SufficiencyGate, thresholds_from_metadata
from rag_engine.storage.content import SqliteChunkContentReader
from rag_engine.storage.filters import SqliteChunkCatalog, SqliteChunkFilter
from rag_engine.storage.lexical import SqliteLexicalIndex
from rag_engine.storage.sqlite import open_store, open_store_read_only
from rag_engine.storage.vector_flat import FlatVectorIndex
from rag_engine.system import SystemClock, system_id_factory, trim_heap


def run_sync[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code, also inside a running event loop.

    Closing providers is asynchronous but happens in ``finally`` blocks of synchronous context
    managers, which a long-running service enters and leaves inside its own loop, where
    ``asyncio.run`` is not allowed; a short-lived helper thread with its own loop is used then.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


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


@dataclass(slots=True)
class BuiltChat:
    """The configured chat model behind the resilience wrapper, and what must be closed."""

    provider: ResilientLLM
    inner: OllamaChat | OpenAICompatChat

    async def aclose(self) -> None:
        await self.inner.aclose()


def build_chat(
    settings: LlmSettings,
    counter: TokenCounter,
    *,
    concurrency: int = 1,
    environ: Mapping[str, str] | None = None,
) -> BuiltChat:
    """Construct the chat provider; the API key is read from the environment, never from config."""
    if (
        not settings.enabled
        or settings.provider is None
        or settings.model is None
        or settings.context_window is None
    ):
        raise RagError(ErrorCode.E_CONFIG_INVALID, "llm is not configured")
    inner: OllamaChat | OpenAICompatChat
    if settings.provider == "ollama":
        inner = OllamaChat(
            settings.model,
            context_window=settings.context_window,
            max_new_tokens=settings.max_new_tokens,
            counter=counter,
            base_url=settings.base_url or "http://127.0.0.1:11434",
            timeout_s=settings.request_timeout_s,
            keep_alive=settings.keep_alive,
            think=settings.think,
        )
    elif settings.provider == "openai_compat":
        key = None
        if settings.api_key_env is not None:
            key = (os.environ if environ is None else environ).get(settings.api_key_env)
            if not key:
                raise RagError(
                    ErrorCode.E_CONFIG_INVALID,
                    f"the environment variable {settings.api_key_env} is not set",
                )
        inner = OpenAICompatChat(
            settings.model,
            base_url=settings.base_url or "",
            context_window=settings.context_window,
            max_new_tokens=settings.max_new_tokens,
            counter=counter,
            api_key=key,
            timeout_s=settings.request_timeout_s,
        )
    else:
        raise RagError(ErrorCode.E_CONFIG_INVALID, "unsupported llm provider")
    policy = ResiliencePolicy(
        timeout_s=settings.request_timeout_s,
        max_retries=settings.max_retries,
        concurrency=concurrency,
    )
    return BuiltChat(ResilientLLM(inner, policy), inner)


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
            if not read_only:
                services = replace(
                    services,
                    maintenance=IndexMaintenance(store, clock),
                    keep_versions=settings.ingest.keep_versions,
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
                    maintenance=services.maintenance,
                    embedding_model=built.model,
                    embedding_service=service,
                    keep_versions=services.keep_versions,
                )
            yield services
        finally:
            if built is not None:
                run_sync(built.aclose())
            store.close()

    @contextmanager
    def search(self, config: Path | None, db: Path | None) -> Iterator[SearchServices]:
        settings = load_settings(config)
        configure_logging(settings.logging.level, json_output=settings.logging.json_output)
        path = db if db is not None else settings.storage.db_path
        store = open_store_read_only(path)
        built = None
        stage: RerankStage | None = None
        try:
            retrieval = settings.retrieval
            embedder = None
            vectors: FlatVectorIndex | None = None
            prefix = ""
            if settings.embedding.enabled:
                built = build_embedding(settings.embedding, SystemClock())
                embedder = built.provider
                prefix = built.model.query_prefix
                vectors = FlatVectorIndex(
                    built.model.id,
                    built.model.dimension,
                    max_memory_mb=settings.effective_limits().max_loaded_vectors_mb,
                )

            def snapshot() -> IndexSnapshot:
                metadata = store.reader().get_index_metadata()
                if metadata is None:
                    return IndexSnapshot(0, None)
                model_id = metadata.active_embedding_model_id
                if vectors is not None and model_id == vectors.model_id:
                    vectors.refresh(path)  # a no-op while the published generation is unchanged
                    metadata = store.reader().get_index_metadata() or metadata
                return IndexSnapshot(metadata.index_generation, model_id)

            counter = build_counter(settings.chunking.counter, settings.chunking.tokenizer_json)
            rerank = settings.rerank
            if rerank.enabled and rerank.model and rerank.model_dir and rerank.revision:
                stage = RerankStage(
                    OnnxCrossEncoder(
                        rerank.model,
                        rerank.model_dir,
                        rerank.revision,
                        max_pair_tokens=rerank.max_pair_tokens,
                        batch_size=rerank.batch_size,
                        threads=rerank.threads,
                    ),
                    RerankBudget(pool_size=rerank.pool_size, timeout_s=rerank.timeout_s),
                )
            service = RetrievalService(
                embedder=embedder,
                query_prefix=prefix,
                vectors=vectors,
                lexical=SqliteLexicalIndex(path),
                filters=SqliteChunkFilter(path),
                catalog=SqliteChunkCatalog(path),
                content=SqliteChunkContentReader(path),
                rerank=stage,
                queries=QueryProcessor(counter, max_tokens=retrieval.query_max_tokens),
                snapshot=snapshot,
                fusion=FusionSettings(
                    method=retrieval.fusion,
                    rrf_k=retrieval.rrf_k,
                    dense_weight=retrieval.dense_weight,
                    lexical_weight=retrieval.lexical_weight,
                    alpha=retrieval.alpha,
                ),
                max_per_document=retrieval.max_per_document,
            )

            def gate() -> SufficiencyGate:
                metadata = store.reader().get_index_metadata()
                if metadata is None:
                    return SufficiencyGate(None, None)
                model_id = metadata.active_embedding_model_id
                return SufficiencyGate(
                    thresholds_from_metadata(metadata.calibration),
                    None if model_id is None else str(model_id),
                )

            def release_models() -> None:
                if stage is not None:
                    stage.release()
                if built is not None and isinstance(built.provider, OnnxEmbedder):
                    built.provider.release()
                trim_heap()  # freed pages go back to the system, not to the allocator

            yield SearchServices(
                retrieval=service,
                store=store,
                ids=system_id_factory(),
                defaults=SearchDefaults(
                    RetrievalMode(retrieval.mode), retrieval.top_k, retrieval.candidate_k
                ),
                context=ContextBuilder(counter, SqliteChunkContentReader(path)),
                counter=counter,
                context_policy=ContextPolicy(
                    max_chunks_per_document=settings.context.max_chunks_per_document,
                    order=settings.context.order,
                    small_to_big=settings.context.small_to_big,
                    min_evidence_items=settings.context.min_evidence_items,
                    item_overhead_tokens=settings.context.item_overhead_tokens,
                ),
                llm_window=settings.llm.context_window,
                llm_max_new=settings.llm.max_new_tokens,
                gate=gate,
                release_models=release_models,
            )
        finally:
            if stage is not None:
                stage.close()
            if built is not None:
                run_sync(built.aclose())
            store.close()

    @contextmanager
    def ask(
        self, config: Path | None, db: Path | None, *, release_models: bool = True
    ) -> Iterator[AskServices]:
        settings = load_settings(config)
        if not settings.llm.enabled:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "no language model is configured: set [llm]")
        answer = settings.answer
        with self.search(config, db) as searcher:
            chat = build_chat(
                settings.llm,
                searcher.counter,
                concurrency=settings.effective_limits().llm_concurrency,
            )
            try:
                generator = AnswerGenerator(
                    chat.provider,
                    policy=GenerationPolicy(
                        max_new_tokens=settings.llm.max_new_tokens,
                        strict_overlap=answer.strict_overlap,
                        min_answer_chars=answer.min_answer_chars,
                        unverified_on_invalid=True,
                    ),
                )
                pipeline = RagPipeline(
                    retrieval=searcher.retrieval,
                    builder=searcher.context,
                    context_policy=searcher.context_policy,
                    counter=searcher.counter,
                    gate=searcher.gate,
                    generator=generator,
                    clock=SystemClock(),
                    limits=PipelineLimits(
                        total_timeout_s=answer.total_timeout_s,
                        shares=StageShares(answer.retrieval_share, answer.context_share),
                        concurrency=settings.effective_limits().llm_concurrency,
                    ),
                    release_models=(
                        searcher.release_models
                        if release_models and answer.release_models
                        else None
                    ),
                )
                yield AskServices(pipeline, searcher.ids, searcher.defaults, searcher)
            finally:
                run_sync(chat.aclose())

    def serve(
        self, config: Path | None, db: Path | None, *, host: str | None, port: int | None
    ) -> int:
        """Run the HTTP API (needs the ``api`` extra) until it is stopped."""
        try:
            uvicorn = importlib.import_module("uvicorn")
            api = importlib.import_module("rag_engine.api.app")
        except ImportError as exc:
            raise RagError(
                ErrorCode.E_CONFIG_INVALID, "the api extra is required: install rag[api]"
            ) from exc
        settings = load_settings(config)
        configure_logging(settings.logging.level, json_output=settings.logging.json_output)
        bind = settings.api.model_copy(
            update={"host": host or settings.api.host, "port": port or settings.api.port}
        )
        token = api.resolve_token(bind)  # refuses a non-loopback bind without a token
        application = api.create_app(self, config, db, settings, token=token, bind=bind)
        uvicorn.run(application, host=bind.host, port=bind.port, log_config=None)
        return 0

    def processing(self, config: Path | None) -> Processing:
        settings = load_settings(config)
        configure_logging(settings.logging.level, json_output=settings.logging.json_output)
        return build_processing(settings, SystemClock(), system_id_factory())


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``rag`` command; returns the process exit code."""
    container: Container = DefaultContainer()
    return cli_main(argv, container=container)
