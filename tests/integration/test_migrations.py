"""P2.2: numbered migrations, checksums, refusal of newer schemas, backup."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag_engine.domain import ErrorCode, IdFactory, RagError
from rag_engine.storage.migrations import (
    Migration,
    Migrator,
    load_packaged_migrations,
    split_statements,
)
from rag_engine.storage.sqlite import backup_database, open_store
from rag_engine.testing import FakeClock
from tests.factories import T0, make_source

V1 = Migration(1, "one", "CREATE TABLE a (x INTEGER);")
V2 = Migration(2, "two", "CREATE TABLE b (y INTEGER);")


def _conn(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path, isolation_level=None)


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r[0] for r in rows}


def test_new_database_gets_schema_v1(tmp_path: Path, clock: FakeClock) -> None:
    with open_store(tmp_path / "rag.db", clock) as store:
        conn = _conn(store.path)
        try:
            assert {
                "sources",
                "documents",
                "versions",
                "sync_state",
                "jobs",
                "index_state",
                "schema_migrations",
            } <= _tables(conn)
            row = conn.execute("SELECT version, name FROM schema_migrations").fetchall()
            assert row == [(1, "core")]
        finally:
            conn.close()
        metadata = store.reader().get_index_metadata()
        assert metadata is not None
        assert metadata.schema_version == 1
        assert metadata.index_generation == 0


def test_reopening_does_not_reapply_migrations(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "rag.db"
    open_store(path, clock).close()
    clock.advance(__import__("datetime").timedelta(days=1))
    open_store(path, clock).close()
    conn = _conn(path)
    try:
        rows = conn.execute("SELECT version, applied_at FROM schema_migrations").fetchall()
    finally:
        conn.close()
    assert [r[0] for r in rows] == [1]
    assert rows[0][1].startswith("2026-01-01")  # applied once, on the first open


def test_migrator_applies_pending_versions_in_order(tmp_path: Path) -> None:
    conn = _conn(tmp_path / "m.db")
    try:
        assert Migrator([V1]).apply(conn, T0) == [1]
        assert Migrator([V1, V2]).apply(conn, T0) == [2]
        assert Migrator([V1, V2]).apply(conn, T0) == []
        assert {"a", "b"} <= _tables(conn)
        assert Migrator([V1, V2]).current_version(conn) == 2
    finally:
        conn.close()


def test_failed_migration_is_rolled_back_atomically(tmp_path: Path) -> None:
    broken = Migration(2, "broken", "CREATE TABLE c (z INTEGER);\nTHIS IS NOT SQL;")
    conn = _conn(tmp_path / "m.db")
    try:
        Migrator([V1]).apply(conn, T0)
        with pytest.raises(RagError) as info:
            Migrator([V1, broken]).apply(conn, T0)
        assert info.value.code is ErrorCode.E_INTEGRITY
        assert "0002_broken" in info.value.message
        assert "c" not in _tables(conn)  # DDL of the failed migration was rolled back
        assert Migrator([V1]).current_version(conn) == 1
        assert not conn.in_transaction
    finally:
        conn.close()


def test_checksum_mismatch_is_detected(tmp_path: Path) -> None:
    conn = _conn(tmp_path / "m.db")
    try:
        Migrator([V1]).apply(conn, T0)
        tampered = Migration(1, "one", "CREATE TABLE a (x INTEGER, extra TEXT);")
        with pytest.raises(RagError) as info:
            Migrator([tampered]).apply(conn, T0)
        assert info.value.code is ErrorCode.E_INTEGRITY
        assert "modified" in info.value.message
    finally:
        conn.close()


def test_line_endings_do_not_change_the_checksum() -> None:
    assert Migration(1, "x", "A\r\nB\r\n").checksum == Migration(1, "x", "A\nB\n").checksum


def test_database_from_a_newer_version_is_refused(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "rag.db"
    open_store(path, clock).close()
    conn = _conn(path)
    conn.execute(
        "INSERT INTO schema_migrations (version, name, checksum, applied_at)"
        " VALUES (99, 'future', 'x', 't')"
    )
    conn.close()
    with pytest.raises(RagError) as info:
        open_store(path, clock)
    assert info.value.code is ErrorCode.E_MIGRATION_REQUIRED
    assert info.value.context["database_version"] == 99


def test_history_with_gaps_is_rejected(tmp_path: Path) -> None:
    conn = _conn(tmp_path / "m.db")
    try:
        Migrator([V1, V2]).apply(conn, T0)
        conn.execute("DELETE FROM schema_migrations WHERE version = 1")
        with pytest.raises(RagError) as info:
            Migrator([V1, V2]).apply(conn, T0)
        assert info.value.code is ErrorCode.E_INTEGRITY
    finally:
        conn.close()


def test_migration_numbering_must_be_contiguous() -> None:
    with pytest.raises(RagError):
        Migrator([V2])
    with pytest.raises(RagError):
        Migrator([V1, V1])


def test_packaged_migrations_are_valid() -> None:
    migrations = load_packaged_migrations()
    assert [m.version for m in migrations] == list(range(1, len(migrations) + 1))
    assert migrations[0].name == "core"


def test_statement_splitter_keeps_triggers_whole() -> None:
    script = (
        "-- comment\nCREATE TABLE t (x);\n"
        "CREATE TRIGGER g AFTER INSERT ON t BEGIN\n  UPDATE t SET x = 1;\nEND;\n"
    )
    assert len(split_statements(script)) == 2
    with pytest.raises(RagError):
        split_statements("CREATE TABLE broken (")


def test_backup_is_a_verified_consistent_copy(
    tmp_path: Path, clock: FakeClock, ids: IdFactory
) -> None:
    with open_store(tmp_path / "rag.db", clock) as store:
        with store.unit_of_work().begin() as writer:
            writer.put_source(make_source(ids))
        target = backup_database(store.path, tmp_path / "backups" / "copy.db")
    assert target.is_file()
    assert not list((tmp_path / "backups").glob("*.partial"))
    with open_store(target, clock) as copy:
        assert [s.name for s in copy.reader().list_sources()] == ["notes"]


def test_backup_refuses_to_overwrite_and_reports_missing_source(
    tmp_path: Path, clock: FakeClock
) -> None:
    with open_store(tmp_path / "rag.db", clock) as store:
        target = backup_database(store.path, tmp_path / "copy.db")
        with pytest.raises(RagError) as info:
            backup_database(store.path, target)
        assert info.value.code is ErrorCode.E_INTEGRITY
    with pytest.raises(RagError) as missing:
        backup_database(tmp_path / "absent.db", tmp_path / "other.db")
    assert missing.value.code is ErrorCode.E_SOURCE_NOT_FOUND
    assert not (tmp_path / "other.db").exists()


def test_backup_with_a_corrupt_copy_is_not_left_behind(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rag_engine.storage.sqlite.backup as backup_module

    def fail(*_: object) -> None:
        raise RagError(ErrorCode.E_INTEGRITY, "the backup failed its integrity check")

    monkeypatch.setattr(backup_module, "_verify", fail)
    with open_store(tmp_path / "rag.db", clock) as store, pytest.raises(RagError):
        backup_database(store.path, tmp_path / "bad.db")
    assert not (tmp_path / "bad.db").exists()
    assert not (tmp_path / "bad.db.partial").exists()
