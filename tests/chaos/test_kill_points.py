"""F3: ``kill -9`` at every stage of ingestion, a job and a migration (NFR-06).

After each kill the database must open and pass ``integrity_check`` and ``foreign_key_check``,
the interrupted work must be either entirely absent or entirely present (never half done), and
running the command again must end in exactly the state an uninterrupted run produces.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from rag_engine.container import build_services, filesystem_adapter
from rag_engine.domain import IngestionJob, JobStatus
from rag_engine.ingestion.jobs import Worker
from rag_engine.ingestion.service import IngestReport
from rag_engine.storage.backup import backup_database
from rag_engine.storage.sqlite import open_store
from rag_engine.testing import FakeClock
from tests.factories import make_ids
from tests.v1_database import build_v1_database

REPO = Path(__file__).resolve().parents[2]
SIGKILL = -9
INGEST_POINTS = ["after_scan", "apply_start", "apply_mid", "before_commit", "after_commit"]


def clock_after(days: int = 0) -> FakeClock:
    clock = FakeClock()
    clock.advance(timedelta(days=days))
    return clock


def ingest_again(database: Path, days: int = 0) -> IngestReport:
    clock = clock_after(days)
    store = open_store(database, clock)
    try:
        services = build_services(
            store, database, clock, make_ids(2_100_000_000_000), filesystem_adapter
        )
        return services.ingest.run("docs")
    finally:
        store.close()


def semantic_state(database: Path) -> dict[str, Any]:
    """What the database means, without identifiers and timestamps."""
    with open_store(database, FakeClock()) as store:
        reader = store.reader()
        state: dict[str, Any] = {}
        key_of = {}
        for source in reader.list_sources():
            documents = reader.list_documents(source.id)
            key_of.update({d.id: d.source_key.value for d in documents})
            state["documents"] = {
                d.source_key.value: (d.status.value, d.missing_since is not None) for d in documents
            }
            state["sync"] = {
                s.source_key.value: (
                    s.status.value,
                    None if s.last_content_hash is None else s.last_content_hash.hex,
                )
                for s in reader.list_sync_states(source.id)
            }
        state["jobs"] = sorted(
            (key_of.get(j.document_id, "-"), j.kind.value, j.status.value)  # type: ignore[arg-type]
            for j in reader.list_jobs()
        )
        return state


def assert_database_is_sound(database: Path) -> None:
    conn = sqlite3.connect(database)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def run_child(point: str, database: Path, days: int = 0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "tests.chaos.child", point, str(database), str(days)],
        cwd=REPO,
        capture_output=True,
        check=False,
    )


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A source with 30 ingested documents, then edited on disk (3 changed, 2 gone, 3 new)."""
    root = tmp_path / "root"
    root.mkdir()
    for i in range(30):
        (root / f"doc{i:02d}.md").write_text(f"# document {i}\n", encoding="utf-8")
    database = tmp_path / "rag.db"
    clock = FakeClock()
    store = open_store(database, clock)
    try:
        services = build_services(store, database, clock, make_ids(), filesystem_adapter)
        services.registry.add("docs", root)
        assert services.ingest.run("docs").new == 30
    finally:
        store.close()
    for i in range(3):
        (root / f"doc{i:02d}.md").write_text(f"# document {i} changed\n", encoding="utf-8")
    for i in (28, 29):
        (root / f"doc{i:02d}.md").unlink()
    for i in range(3):
        (root / f"new{i}.md").write_text(f"# brand new {i}\n", encoding="utf-8")
    return database, root


@pytest.mark.chaos
@pytest.mark.parametrize("point", INGEST_POINTS)
def test_kill_during_ingest_never_leaves_half_done_work(
    corpus: tuple[Path, Path], tmp_path: Path, point: str
) -> None:
    database, _ = corpus
    reference_db = backup_database(database, tmp_path / "reference.db")
    before = semantic_state(database)
    ingest_again(reference_db)
    expected = semantic_state(reference_db)
    assert before != expected  # the scenario really has something to do

    result = run_child(point, database)
    assert result.returncode == SIGKILL, result.stderr.decode()

    assert_database_is_sound(database)
    after_kill = semantic_state(database)
    assert after_kill in (before, expected)  # all or nothing
    if point == "after_commit":
        assert after_kill == expected
    else:
        assert after_kill == before  # the transaction never committed

    report = ingest_again(database)  # the next run finishes the work
    assert semantic_state(database) == expected
    assert_database_is_sound(database)
    assert report.new + report.modified + report.deleted > 0 or point == "after_commit"
    settled = semantic_state(database)
    ingest_again(database)
    assert semantic_state(database) == settled  # and a further run changes nothing


@pytest.mark.chaos
def test_kill_during_garbage_collection_keeps_the_tombstones(
    corpus: tuple[Path, Path], tmp_path: Path
) -> None:
    database, _ = corpus
    ingest_again(database)  # 2 files are gone: tombstones
    reference_db = backup_database(database, tmp_path / "reference.db")
    before = semantic_state(database)
    assert sum(1 for s, _ in before["documents"].values() if s == "missing") == 2
    ingest_again(reference_db, days=31)
    expected = semantic_state(reference_db)
    assert len(expected["documents"]) == len(before["documents"]) - 2  # collected

    result = run_child("gc_mid", database, days=31)
    assert result.returncode == SIGKILL, result.stderr.decode()
    assert_database_is_sound(database)
    assert semantic_state(database) == before  # the half-done collection was rolled back

    assert ingest_again(database, days=31).gc_removed == 2
    assert semantic_state(database) == expected
    assert_database_is_sound(database)


@pytest.mark.chaos
def test_kill_in_the_middle_of_a_job_is_recovered_without_duplicates(
    corpus: tuple[Path, Path],
) -> None:
    database, _ = corpus
    ingest_again(database)  # queues one job per document that needs processing
    with open_store(database, FakeClock()) as store:
        total = len(store.reader().list_jobs())
        assert total > 2

    result = run_child("job_mid", database)
    assert result.returncode == SIGKILL, result.stderr.decode()
    assert_database_is_sound(database)

    clock = FakeClock()
    with open_store(database, clock) as store:
        reader = store.reader()
        counts = reader.count_jobs()
        assert counts[JobStatus.DONE] == 1 and counts[JobStatus.LEASED] == 1
        (stuck,) = reader.list_jobs(JobStatus.LEASED)
        assert stuck.attempts == 1
        services = build_services(
            store, database, clock, make_ids(2_200_000_000_000), filesystem_adapter
        )
        # while the dead worker's lease is valid nobody can take that job over
        claimed = services.queue.claim()
        assert claimed is not None and claimed.id != stuck.id
        assert services.queue.complete(claimed)
        clock.advance(timedelta(seconds=301))
        seen: list[IngestionJob] = []

        def handler(job: IngestionJob, heartbeat: Callable[[], bool]) -> None:
            del heartbeat
            seen.append(job)

        Worker(services.queue, handler).run_until_idle(max_jobs=1000)
        final = reader.list_jobs()
        assert len(final) == total  # no job was duplicated
        assert {j.status for j in final} == {JobStatus.DONE}
        recovered = reader.get_job(stuck.id)
        assert recovered is not None and recovered.attempts == 2
        assert len({j.id for j in seen}) == len(seen)  # nothing processed twice in the rerun


@pytest.mark.chaos
def test_kill_in_the_middle_of_a_migration_leaves_the_old_schema_and_retries(
    tmp_path: Path,
) -> None:
    database = tmp_path / "rag.db"
    _, counts = build_v1_database(database)

    result = run_child("migration_mid", database)
    assert result.returncode == SIGKILL, result.stderr.decode()

    conn = sqlite3.connect(database)
    try:
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone() == (1,)
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone() == (counts["documents"],)
    finally:
        conn.close()
    assert len(list(tmp_path.glob("*.bak"))) == 1  # the backup was taken before the attempt

    with open_store(database, FakeClock()) as store:  # the next start migrates cleanly
        source = store.reader().get_source_by_name("notes")
        assert source is not None
        assert store.reader().count_documents(source.id) == counts["documents"]
        metadata = store.reader().get_index_metadata()
        assert metadata is not None and metadata.schema_version == 2
    assert_database_is_sound(database)
    names = sorted(p.name for p in tmp_path.glob("*.bak"))
    assert names == [
        "rag.db.pre-migration-v1-to-v2.2.bak",
        "rag.db.pre-migration-v1-to-v2.bak",
    ]
