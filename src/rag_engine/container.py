"""Composition root: the one place that reads configuration and creates adapters.

``rag`` starts here (:func:`main`): settings are loaded, the SQLite store, the real clock,
identifier entropy and the filesystem adapter are built, and the command line is handed a
container that produces them. Everything below this module receives its collaborators
through constructors.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from rag_engine.cli import main as cli_main
from rag_engine.cli.context import Container, Services
from rag_engine.config import IngestSettings, load_settings
from rag_engine.domain import ErrorCode, IdFactory, RagError, Source
from rag_engine.ingestion.identity import AdapterFactory, SourceRegistry
from rag_engine.ingestion.jobs import JobQueue, RetryPolicy
from rag_engine.ingestion.service import IngestService
from rag_engine.ingestion.sources import FilesystemSource
from rag_engine.observability import configure_logging
from rag_engine.ports import Clock, SourceAdapter, Store
from rag_engine.storage.sqlite import open_store, open_store_read_only
from rag_engine.system import SystemClock, system_id_factory


def filesystem_adapter(source: Source) -> SourceAdapter:
    if source.locator is None:
        raise RagError(ErrorCode.E_CONFIG_INVALID, "only filesystem sources can be ingested")
    return FilesystemSource(Path(source.locator), source.policy)


def retry_policy(settings: IngestSettings) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=settings.max_attempts,
        base_delay_s=settings.retry_base_delay_s,
        factor=settings.retry_factor,
        max_delay_s=settings.retry_max_delay_s,
        jitter=settings.retry_jitter,
        lease_s=settings.lease_s,
    )


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
        try:
            yield build_services(
                store,
                path,
                clock,
                system_id_factory(),
                filesystem_adapter,
                retry_policy(settings.ingest),
                random.SystemRandom().random,
            )
        finally:
            store.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``rag`` command; returns the process exit code."""
    container: Container = DefaultContainer()
    return cli_main(argv, container=container)
