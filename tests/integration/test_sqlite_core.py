"""P2.1: connection lifecycle, WAL, single writer, unit of work, crash behaviour."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from rag_engine.domain import ErrorCode, IdFactory, RagError, Sha256Hex, SourcePolicy
from rag_engine.ports import MetadataReader, MetadataWriter, Store, UnitOfWork
from rag_engine.storage.sqlite import (
    SqliteStore,
    check_capabilities,
    open_store,
    open_store_read_only,
)
from rag_engine.testing import FakeClock
from tests.factories import make_source


def test_store_satisfies_the_ports(store: SqliteStore) -> None:
    assert isinstance(store, Store)
    assert isinstance(store.reader(), MetadataReader)
    assert isinstance(store.unit_of_work(), UnitOfWork)
    with store.unit_of_work().begin() as writer:
        assert isinstance(writer, MetadataWriter)


def test_database_uses_wal_and_enforces_foreign_keys(store: SqliteStore) -> None:
    conn = sqlite3.connect(store.path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()
    with pytest.raises(RagError) as info, store.unit_of_work().begin() as writer:
        # a document of a source that does not exist violates a foreign key
        writer._run(  # type: ignore[attr-defined]
            "INSERT INTO documents (id, source_id, source_key, status, created_at, updated_at)"
            " VALUES ('d', 'missing', 'k', 'active', 't', 't')"
        )
    assert info.value.code is ErrorCode.E_INTEGRITY


def test_commit_makes_changes_visible_and_durable(
    store: SqliteStore, ids: IdFactory, tmp_path: Path, clock: FakeClock
) -> None:
    source = make_source(ids)
    with store.unit_of_work().begin() as writer:
        writer.put_source(source)
    assert store.reader().get_source(source.id) == source
    store.close()
    with open_store(tmp_path / "rag.db", clock) as reopened:
        assert reopened.reader().get_source_by_name("notes") == source
        assert reopened.reader().list_sources() == [source]


def test_exception_inside_the_unit_of_work_rolls_everything_back(
    store: SqliteStore, ids: IdFactory
) -> None:
    first = make_source(ids, name="first", locator="/a")
    second = make_source(ids, name="second", locator="/b")

    class Boom(Exception):
        pass

    def failing_unit() -> None:
        with store.unit_of_work().begin() as writer:
            writer.put_source(first)
            writer.put_source(second)
            raise Boom

    with pytest.raises(Boom):
        failing_unit()

    assert store.reader().list_sources() == []
    # the store is still usable after a rollback
    with store.unit_of_work().begin() as writer:
        writer.put_source(first)
    assert [s.name for s in store.reader().list_sources()] == ["first"]


def test_a_constraint_violation_leaves_no_partial_state(store: SqliteStore, ids: IdFactory) -> None:
    with store.unit_of_work().begin() as writer:
        writer.put_source(make_source(ids, name="same", locator="/a"))

    def violating_unit() -> None:
        with store.unit_of_work().begin() as writer:
            writer.put_source(make_source(ids, name="other", locator="/b"))
            writer.put_source(make_source(ids, name="same", locator="/c"))  # duplicate name

    with pytest.raises(RagError) as info:
        violating_unit()
    assert info.value.code is ErrorCode.E_INTEGRITY
    assert [s.name for s in store.reader().list_sources()] == ["same"]


def test_second_concurrent_writer_gets_store_locked(store: SqliteStore, tmp_path: Path) -> None:
    other = SqliteStore(tmp_path / "rag.db")
    try:
        with store.unit_of_work().begin():
            with pytest.raises(RagError) as info, other.unit_of_work().begin():
                pytest.fail("a second writer must not get in")
            assert info.value.code is ErrorCode.E_STORE_LOCKED
            assert info.value.retryable
        # released after the first writer is done
        with other.unit_of_work().begin():
            pass
    finally:
        other.close()


def test_a_waiting_writer_is_serialised_behind_the_first(
    store: SqliteStore, tmp_path: Path, ids: IdFactory
) -> None:
    waiting = SqliteStore(tmp_path / "rag.db", writer_timeout_s=5.0)
    started = threading.Event()
    results: list[str] = []

    def second_writer() -> None:
        started.set()
        with waiting.unit_of_work().begin() as writer:
            writer.put_source(make_source(ids, name="late", locator="/late"))
        results.append("done")

    thread = threading.Thread(target=second_writer)
    try:
        with store.unit_of_work().begin() as writer:
            writer.put_source(make_source(ids, name="early", locator="/early"))
            thread.start()
            started.wait(2)
            assert results == []  # blocked while the first writer is open
        thread.join(5)
    finally:
        waiting.close()
    assert results == ["done"]
    assert [s.name for s in store.reader().list_sources()] == ["early", "late"]


def test_nested_units_of_work_are_rejected(store: SqliteStore) -> None:
    with (
        store.unit_of_work().begin(),
        pytest.raises(RagError) as info,
        store.unit_of_work().begin(),
    ):
        pytest.fail("a nested unit of work must not start")
    assert info.value.code is ErrorCode.E_INTEGRITY


def test_readers_are_not_blocked_by_an_open_writer(store: SqliteStore, ids: IdFactory) -> None:
    with store.unit_of_work().begin() as writer:
        writer.put_source(make_source(ids))
    seen: list[int] = []

    def read() -> None:
        seen.append(len(store.reader().list_sources()))

    with store.unit_of_work().begin() as writer:
        writer.put_source(make_source(ids, name="uncommitted", locator="/u"))
        threads = [threading.Thread(target=read) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert not any(t.is_alive() for t in threads)
        assert len(writer.list_sources()) == 2  # the writer sees its own change
    assert seen == [1, 1, 1, 1]  # readers saw only committed data and never waited


def test_closing_releases_connections_and_the_lock(
    tmp_path: Path, clock: FakeClock, ids: IdFactory
) -> None:
    store = open_store(tmp_path / "rag.db", clock)
    store.reader().list_sources()
    store.close()
    store.close()  # idempotent
    with pytest.raises(RagError) as info:
        store.reader()
    assert info.value.code is ErrorCode.E_INTEGRITY
    with pytest.raises(RagError):
        store.unit_of_work()
    # the file is free again for another handle
    with open_store(tmp_path / "rag.db", clock) as again, again.unit_of_work().begin() as writer:
        writer.put_source(make_source(ids))


def test_open_store_failure_does_not_leak_the_lock(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "rag.db"
    path.write_bytes(b"this is not a database" * 100)
    with pytest.raises(RagError):
        open_store(path, clock)
    other = SqliteStore(path)
    try:
        other._lock.acquire()
        other._lock.release()
    finally:
        other.close()


def test_state_survives_an_interrupted_transaction(
    tmp_path: Path, clock: FakeClock, ids: IdFactory
) -> None:
    db = tmp_path / "rag.db"
    with open_store(db, clock) as first, first.unit_of_work().begin() as writer:
        writer.put_source(make_source(ids, name="committed", locator="/committed"))

    child = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from rag_engine.storage.sqlite import open_store
        from rag_engine.testing import FakeClock
        from tests.factories import make_ids, make_source

        store = open_store(Path({str(db)!r}), FakeClock())
        ids = make_ids(1_800_000_000_000)
        with store.unit_of_work().begin() as writer:
            writer.put_source(make_source(ids, name="torn", locator="/torn"))
            os._exit(3)  # the process dies in the middle of the transaction
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", child],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
    )
    assert result.returncode == 3, result.stderr.decode()

    with open_store(db, clock) as recovered:
        # nothing of the torn transaction survived and the lock was released by the OS
        assert [s.name for s in recovered.reader().list_sources()] == ["committed"]
        with recovered.unit_of_work().begin() as writer:
            writer.put_source(make_source(ids, name="after", locator="/after"))
        assert len(recovered.reader().list_sources()) == 2
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_old_sqlite_versions_are_refused() -> None:
    check_capabilities()
    with pytest.raises(RagError) as info:
        check_capabilities((3, 34, 0))
    assert info.value.code is ErrorCode.E_RESOURCE_GATE


def test_read_only_store_cannot_write_and_leaves_the_file_untouched(
    store: SqliteStore, ids: IdFactory
) -> None:
    with store.unit_of_work().begin() as writer:
        writer.put_source(make_source(ids))
    before = store.path.read_bytes()
    with open_store_read_only(store.path) as ro:
        assert len(ro.reader().list_sources()) == 1
        with pytest.raises(RagError), ro.unit_of_work().begin():
            pass
    assert store.path.read_bytes() == before


def test_read_only_open_of_a_missing_database_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(RagError) as info:
        open_store_read_only(tmp_path / "absent.db")
    assert info.value.code is ErrorCode.E_SOURCE_NOT_FOUND
    assert not (tmp_path / "absent.db").exists()


def test_changing_a_locator_without_rebase_is_refused(store: SqliteStore, ids: IdFactory) -> None:
    source = make_source(ids, policy=SourcePolicy())
    with store.unit_of_work().begin() as writer:
        writer.put_source(source)
    moved = replace(
        source, locator="/elsewhere", locator_fingerprint=Sha256Hex.of_text("/elsewhere")
    )
    with pytest.raises(RagError) as info, store.unit_of_work().begin() as writer:
        writer.put_source(moved)
    assert info.value.code is ErrorCode.E_SOURCE_LOCATOR_CHANGED
    with store.unit_of_work().begin() as writer:
        rebased = writer.rebase_source(
            source.id, "/elsewhere", Sha256Hex.of_text("/elsewhere"), source.updated_at
        )
    assert rebased.locator == "/elsewhere"
    assert store.reader().get_source(source.id) == rebased


def test_new_database_is_private_and_paths_with_spaces_and_unicode_work(
    tmp_path: Path, clock: FakeClock, ids: IdFactory
) -> None:
    path = tmp_path / "Мои документы" / "базa знаний.db"
    with open_store(path, clock) as store, store.unit_of_work().begin() as writer:
        writer.put_source(make_source(ids))
    assert path.stat().st_mode & 0o777 == 0o600
    with open_store_read_only(path) as ro:
        assert [s.name for s in ro.reader().list_sources()] == ["notes"]
