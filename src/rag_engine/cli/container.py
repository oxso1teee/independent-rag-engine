"""Composition root: builds the services the commands use.

The commands receive a :class:`Container` and never construct infrastructure themselves, so
tests substitute a container with a fake clock and a temporary database.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_engine.config import load_settings
from rag_engine.domain import ErrorCode, IdFactory, RagError, Source
from rag_engine.ingestion.identity import AdapterFactory, SourceRegistry
from rag_engine.ingestion.service import IngestService
from rag_engine.ingestion.sources import FilesystemSource
from rag_engine.ports import Clock, SourceAdapter, Store
from rag_engine.storage.sqlite import open_store, open_store_read_only
from rag_engine.system import SystemClock, system_id_factory


@dataclass(frozen=True, slots=True)
class Services:
    store: Store
    db_path: Path
    registry: SourceRegistry
    ingest: IngestService


class Container(Protocol):
    def open(
        self, config: Path | None, db: Path | None, *, read_only: bool
    ) -> AbstractContextManager[Services]:
        """Context-managed services; the store is closed when the block ends."""
        ...


def filesystem_adapter(source: Source) -> SourceAdapter:
    if source.locator is None:
        raise RagError(ErrorCode.E_CONFIG_INVALID, "only filesystem sources can be ingested")
    return FilesystemSource(Path(source.locator), source.policy)


def build_services(
    store: Store, db_path: Path, clock: Clock, ids: IdFactory, adapter_factory: AdapterFactory
) -> Services:
    return Services(
        store=store,
        db_path=db_path,
        registry=SourceRegistry(store, clock, ids, adapter_factory),
        ingest=IngestService(store, clock, ids, adapter_factory),
    )


class DefaultContainer:
    """Real clock, real entropy, SQLite at the configured (or overridden) path."""

    @contextmanager
    def open(self, config: Path | None, db: Path | None, *, read_only: bool) -> Iterator[Services]:
        settings = load_settings(config)
        path = db if db is not None else settings.storage.db_path
        clock = SystemClock()
        store = (
            open_store_read_only(path)
            if read_only
            else open_store(path, clock, writer_timeout_s=5.0)
        )
        try:
            yield build_services(store, path, clock, system_id_factory(), filesystem_adapter)
        finally:
            store.close()
