"""The single-writer unit of work: lock, ``BEGIN IMMEDIATE``, commit or rollback."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

from rag_engine.domain import ErrorCode, RagError
from rag_engine.ports import MetadataWriter
from rag_engine.storage.sqlite.connection import WriterLock
from rag_engine.storage.sqlite.repository import SqliteWriter, translate


class UnitOfWorkHost(Protocol):
    """What a unit of work needs from the store that created it."""

    read_only: bool
    local: threading.local
    writer_lock: WriterLock

    def writer_connection(self) -> sqlite3.Connection: ...


class SqliteUnitOfWork:
    def __init__(self, store: UnitOfWorkHost) -> None:
        self._store = store

    @contextmanager
    def begin(self) -> Iterator[MetadataWriter]:
        store = self._store
        if store.read_only:
            raise RagError(ErrorCode.E_INTEGRITY, "the store was opened read-only")
        if getattr(store.local, "in_uow", False):
            raise RagError(ErrorCode.E_INTEGRITY, "nested units of work are not allowed")
        store.writer_lock.acquire()
        store.local.in_uow = True
        try:
            conn = store.writer_connection()
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise translate(exc) from exc
            try:
                yield SqliteWriter(conn)
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            try:
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise translate(exc) from exc
        finally:
            store.local.in_uow = False
            store.writer_lock.release()
