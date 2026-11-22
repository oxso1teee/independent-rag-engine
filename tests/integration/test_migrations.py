"""P2.2: numbered migrations, checksums, refusal of newer schemas, backup."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag_engine.domain import ErrorCode, IdFactory, RagError, Sha256Hex
from rag_engine.storage.migrations import (
    Migration,
    Migrator,
    load_packaged_migrations,
    split_statements,
)
from rag_engine.storage.sqlite import backup_database, open_store
from rag_engine.testing import FakeClock
from tests.factories import T0, make_ids, make_source

V1 = Migration(1, "one", "CREATE TABLE a (x INTEGER);")
V2 = Migration(2, "two", "CREATE TABLE b (y INTEGER);")


def _conn(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path, isolation_level=None)


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r[0] for r in rows}


def test_new_database_gets_the_latest_schema(tmp_path: Path, clock: FakeClock) -> None:
    with open_store(tmp_path / "rag.db", clock) as store:
        conn = _conn(store.path)
        try:
            assert {
                "sources",
                "documents",
                "document_versions",
                "sync_state",
                "jobs",
                "index_state",
                "schema_migrations",
            } <= _tables(conn)
            row = conn.execute("SELECT version, name FROM schema_migrations").fetchall()
            assert row == [(1, "core"), (2, "align_data_model")]
            assert "versions" not in _tables(conn)
        finally:
            conn.close()
        metadata = store.reader().get_index_metadata()
        assert metadata is not None
        assert metadata.schema_version == 2
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
    assert [r[0] for r in rows] == [1, 2]
    assert {r[1][:10] for r in rows} == {"2026-01-01"}  # applied once, on the first open
    assert not list(tmp_path.glob("*.bak"))  # a fresh database needs no backup


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


PUBLISHED_0001_SHA256 = "b04ed778d1ae9e802297a1235ff3bf2cca89b1ae22074ea08064053142ab748b"


def test_the_published_migration_0001_is_unchanged() -> None:
    first = load_packaged_migrations()[0]
    assert first.checksum == PUBLISHED_0001_SHA256


def test_migration_scripts_never_edit_the_schema_version_themselves() -> None:
    # the runner records the version in schema_migrations; index_state mirrors it in code
    for migration in load_packaged_migrations():
        assert (
            "index_state"
            not in migration.sql.replace("-- ", "").split("CREATE TABLE index_state")[0]
            or migration.version == 1
        )
        if migration.version > 1:
            assert "index_state" not in migration.sql


class V1Data:
    """Identifiers of the rows written into a schema-v1 database."""

    def __init__(self) -> None:
        ids = make_ids()
        self.source = str(ids.source_id())
        self.docs = {key: str(ids.document_id()) for key in "abcde"}
        self.version = str(ids.document_version_id())
        self.jobs = [str(ids.job_id()), str(ids.job_id())]
        self.hash = "a" * 64
        self.statuses = {
            "a": "pending",
            "b": "active",
            "c": "missing",
            "d": "failed",
            "e": "deleted",
        }


def _build_v1_database(path: Path) -> tuple[V1Data, dict[str, int]]:
    """A database exactly as released with schema v1, filled through plain SQL."""
    data = V1Data()
    conn = _conn(path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        Migrator(load_packaged_migrations()[:1]).apply(conn, T0)
        stamp = "2026-01-01T00:00:00.000000+00:00"
        conn.execute(
            "INSERT INTO sources VALUES (?, 'notes', 'fs', '/n', ?, ?, ?, ?)",
            (
                data.source,
                Sha256Hex.of_text("/n").hex,
                '{"include":["**/*"],"exclude":[],"max_file_bytes":1000,"cleaners":[],'
                '"follow_symlinks":false,"delete_grace_days":30}',
                stamp,
                stamp,
            ),
        )
        for key, status in data.statuses.items():
            conn.execute(
                "INSERT INTO documents (id, source_id, source_key, current_version_id, status,"
                " missing_since, tags_json, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?)",
                (
                    data.docs[key],
                    data.source,
                    f"{key}.md",
                    data.version if key == "b" else None,
                    status,
                    stamp if status == "missing" else None,
                    stamp,
                    stamp,
                ),
            )
        conn.execute(
            "INSERT INTO versions VALUES (?, ?, 1, ?, 5, 6, 'text/plain', 'content',"
            " 'p', '1', '1', 'fp', 0, '[]', ?)",
            (data.version, data.docs["b"], data.hash, stamp),
        )
        conn.execute(
            "INSERT INTO sync_state VALUES (?, 'a.md', 1, 2, ?, ?, 'changed', 0, NULL, NULL)",
            (data.source, data.hash, stamp),
        )
        for job_id, key in zip(data.jobs, "ab", strict=True):
            conn.execute(
                "INSERT INTO jobs VALUES (?, 'ingest_doc', ?, ?, 'discovered', 'queued', NULL,"
                ' 0, NULL, \'{"mime":"text/plain"}\', ?, ?, NULL)',
                (job_id, data.source, data.docs[key], stamp, stamp),
            )
        conn.execute(
            "INSERT INTO index_state VALUES (1, 1, 7, NULL, 'none', 'none', 'none', '{}', ?, NULL)",
            (stamp,),
        )
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
            for t in ("sources", "documents", "versions", "sync_state", "jobs", "index_state")
        }
        return data, counts
    finally:
        conn.close()


def test_upgrade_of_a_real_v1_database_keeps_every_row_and_link(
    tmp_path: Path, clock: FakeClock
) -> None:
    path = tmp_path / "rag.db"
    data, before = _build_v1_database(path)
    with open_store(path, clock) as store:
        conn = _conn(store.path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            after = {
                "sources": conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
                "documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "versions": conn.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0],
                "sync_state": conn.execute("SELECT COUNT(*) FROM sync_state").fetchone()[0],
                "jobs": conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
                "index_state": conn.execute("SELECT COUNT(*) FROM index_state").fetchone()[0],
            }
            assert after == before  # nothing lost, in particular no cascade through documents
            statuses = dict(conn.execute("SELECT id, status FROM documents").fetchall())
            expected = {"a": "active", "b": "active", "c": "missing", "d": "failed", "e": "deleted"}
            assert statuses == {data.docs[k]: v for k, v in expected.items()}
            assert conn.execute(
                "SELECT current_version_id FROM documents WHERE id = ?", (data.docs["b"],)
            ).fetchone() == (data.version,)
            assert conn.execute("SELECT document_id FROM jobs ORDER BY id").fetchall() == [
                (data.docs["a"],),
                (data.docs["b"],),
            ]
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
            assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            history = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
            assert [v for v, _ in history] == [1, 2]
            assert history[0][1] == PUBLISHED_0001_SHA256  # 0001 was neither changed nor re-run
        finally:
            conn.close()
        metadata = store.reader().get_index_metadata()
        assert metadata is not None
        assert (metadata.schema_version, metadata.index_generation) == (2, 7)  # mirrored, kept
        source = store.reader().get_source_by_name("notes")
        assert source is not None
        docs = store.reader().list_documents(source.id)  # every row maps back to the domain
        assert [d.status.value for d in docs] == [
            "active",
            "active",
            "missing",
            "failed",
            "deleted",
        ]
        assert [d.is_searchable for d in docs] == [False, True, False, False, False]
    # the automatic backup carries the real versions and opens as a v1 database
    (backup,) = tmp_path.glob("*.bak")
    assert backup.name == "rag.db.pre-migration-v1-to-v2.bak"
    assert backup.stat().st_mode & 0o777 == 0o600
    old = _conn(backup)
    try:
        assert old.execute("SELECT MAX(version) FROM schema_migrations").fetchone() == (1,)
        assert old.execute(
            "SELECT status FROM documents WHERE id = ?", (data.docs["a"],)
        ).fetchone() == ("pending",)
    finally:
        old.close()


def test_content_hash_is_unique_per_document_after_the_upgrade(
    tmp_path: Path, clock: FakeClock
) -> None:
    path = tmp_path / "rag.db"
    data, _ = _build_v1_database(path)
    open_store(path, clock).close()
    conn = _conn(path)
    try:
        stamp = "2026-01-01T00:00:00.000000+00:00"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO document_versions (id, document_id, seq, content_hash, size_bytes,"
                " mtime_ns, mime, detected_by, parser_id, parser_version, normalizer_version,"
                " chunker_fingerprint, created_at)"
                " VALUES ('v9', ?, 2, ?, 1, 1, 't', 'c', 'p', '1', '1', 'fp', ?)",
                (data.docs["b"], data.hash, stamp),
            )
    finally:
        conn.close()


def test_backup_names_follow_the_real_versions_and_never_collide(tmp_path: Path) -> None:
    from rag_engine.storage.sqlite.store import backup_path

    db = tmp_path / "rag.db"
    first = backup_path(db, 3, 5)
    assert first.name == "rag.db.pre-migration-v3-to-v5.bak"
    first.write_bytes(b"x")
    assert backup_path(db, 3, 5).name == "rag.db.pre-migration-v3-to-v5.2.bak"
    assert backup_path(db, 1, 2).name == "rag.db.pre-migration-v1-to-v2.bak"


def test_a_table_rebuild_that_breaks_foreign_keys_is_rolled_back(tmp_path: Path) -> None:
    parent = Migration(
        1,
        "tables",
        "CREATE TABLE p (id INTEGER PRIMARY KEY);\n"
        "CREATE TABLE c (id INTEGER PRIMARY KEY, p_id INTEGER REFERENCES p (id));\n"
        "INSERT INTO p VALUES (1);\n"
        "INSERT INTO c VALUES (1, 1);\n",
    )
    broken = Migration(2, "orphan", "-- migrate: foreign_keys=off\nDELETE FROM p;")
    conn = _conn(tmp_path / "m.db")
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        Migrator([parent]).apply(conn, T0)
        with pytest.raises(RagError) as info:
            Migrator([parent, broken]).apply(conn, T0)
        assert info.value.code is ErrorCode.E_INTEGRITY
        assert "foreign keys" in info.value.message
        assert conn.execute("SELECT COUNT(*) FROM p").fetchone() == (1,)  # rolled back
        assert conn.execute("PRAGMA foreign_keys").fetchone() == (1,)  # restored
        assert Migrator([parent]).current_version(conn) == 1
    finally:
        conn.close()


def test_backup_hook_runs_only_for_existing_databases(tmp_path: Path) -> None:
    calls: list[tuple[int, int]] = []
    conn = _conn(tmp_path / "m.db")
    try:
        Migrator([V1, V2]).apply(conn, T0, lambda a, b: calls.append((a, b)))
        assert calls == []  # fresh database
        v3 = Migration(3, "three", "CREATE TABLE d (z INTEGER);")
        Migrator([V1, V2, v3]).apply(conn, T0, lambda a, b: calls.append((a, b)))
        assert calls == [(2, 3)]
        Migrator([V1, V2, v3]).apply(conn, T0, lambda a, b: calls.append((a, b)))
        assert calls == [(2, 3)]  # nothing pending: no backup
    finally:
        conn.close()
