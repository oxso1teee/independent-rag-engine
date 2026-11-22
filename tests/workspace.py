"""Shared fixtures: a real store, a real directory tree and the ingestion services."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_engine.domain import IdFactory, Source
from rag_engine.ingestion.identity import SourceRegistry
from rag_engine.ingestion.jobs import JobQueue, RetryPolicy
from rag_engine.ingestion.service import IngestService
from rag_engine.ingestion.sources import FilesystemSource
from rag_engine.ports import SourceAdapter
from rag_engine.storage.sqlite import SqliteStore
from rag_engine.testing import FakeClock


def logical_dump(store: SqliteStore) -> str:
    """SHA-256 of every row of every table, independent of file layout and rowid order."""
    conn = sqlite3.connect(f"file:{store.path}?mode=ro", uri=True)
    try:
        digest = hashlib.sha256()
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        for table in tables:
            rows = sorted(repr(r) for r in conn.execute(f"SELECT * FROM {table}"))  # noqa: S608
            digest.update(table.encode())
            digest.update("\n".join(rows).encode())
        return digest.hexdigest()
    finally:
        conn.close()


@dataclass
class Workspace:
    root: Path
    store: SqliteStore
    clock: FakeClock
    ids: IdFactory
    registry: SourceRegistry
    ingest: IngestService
    queue: JobQueue
    source: Source
    adapters: list[SourceAdapter]

    def write(self, name: str, data: str | bytes = "content") -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_bytes(data)
        return path

    def run(self, **options: bool) -> object:
        return self.ingest.run(self.source.name, **options)


@pytest.fixture
def make_workspace(
    tmp_path: Path, store: SqliteStore, clock: FakeClock, ids: IdFactory
) -> Callable[..., Workspace]:
    def build(
        name: str = "docs", wrap: Callable[[SourceAdapter], SourceAdapter] | None = None
    ) -> Workspace:
        root = tmp_path / "root"
        root.mkdir(exist_ok=True)
        adapters: list[SourceAdapter] = []

        def factory(source: Source) -> SourceAdapter:
            assert source.locator is not None
            adapter: SourceAdapter = FilesystemSource(Path(source.locator), source.policy)
            if wrap is not None:
                adapter = wrap(adapter)
            adapters.append(adapter)
            return adapter

        registry = SourceRegistry(store, clock, ids, factory)
        queue = JobQueue(store, clock, ids, RetryPolicy())
        source = registry.add(name, root)
        return Workspace(
            root=root,
            store=store,
            clock=clock,
            ids=ids,
            registry=registry,
            ingest=IngestService(store, clock, ids, factory),
            queue=queue,
            source=source,
            adapters=adapters,
        )

    return build


@pytest.fixture
def workspace(make_workspace: Callable[..., Workspace]) -> Workspace:
    return make_workspace()
