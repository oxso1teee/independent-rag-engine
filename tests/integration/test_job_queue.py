"""P2.7: job queue, leases, retries, dead letters, crash recovery, concurrent workers."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest

from rag_engine.domain import (
    DocumentId,
    ErrorCode,
    IdFactory,
    IngestionJob,
    JobKind,
    JobStatus,
    RagError,
    SourceId,
)
from rag_engine.ingestion.jobs import JobQueue, RetryPolicy, Worker
from rag_engine.storage.sqlite import SqliteStore, open_store
from rag_engine.testing import FakeClock
from tests.factories import make_ids, make_source


@pytest.fixture
def queue(store: SqliteStore, clock: FakeClock, ids: IdFactory) -> JobQueue:
    return JobQueue(store, clock, ids, RetryPolicy(max_attempts=3, base_delay_s=30, lease_s=300))


@pytest.fixture
def source_id(store: SqliteStore, ids: IdFactory) -> SourceId:
    source = make_source(ids)
    with store.unit_of_work().begin() as writer:
        writer.put_source(source)
    return source.id


def add_document(store: SqliteStore, ids: IdFactory, source_id: SourceId, key: str) -> DocumentId:
    from rag_engine.domain import Document, DocumentStatus, SourceKey
    from tests.factories import T0

    document = Document(
        id=ids.document_id(),
        source_id=source_id,
        source_key=SourceKey(key),
        current_version_id=None,
        status=DocumentStatus.PENDING,
        created_at=T0,
        updated_at=T0,
    )
    with store.unit_of_work().begin() as writer:
        writer.put_document(document)
    return document.id


def ok(job: IngestionJob, heartbeat: Callable[[], bool]) -> None:
    del job, heartbeat


def test_enqueue_is_idempotent_while_a_job_is_active(
    queue: JobQueue, store: SqliteStore, ids: IdFactory, source_id: SourceId
) -> None:
    doc = add_document(store, ids, source_id, "a.md")
    first = queue.enqueue(JobKind.INGEST_DOC, source_id, doc, {"n": 1})
    second = queue.enqueue(JobKind.INGEST_DOC, source_id, doc, {"n": 2})
    assert second.id == first.id
    assert len(store.reader().list_jobs()) == 1
    # a different document, and a document-less job of another kind, are separate jobs
    other = add_document(store, ids, source_id, "b.md")
    assert queue.enqueue(JobKind.INGEST_DOC, source_id, other).id != first.id
    assert queue.enqueue(JobKind.GC, source_id).id != first.id
    assert queue.enqueue(JobKind.GC, source_id).kind is JobKind.GC
    assert len(store.reader().list_jobs()) == 3


def test_a_finished_job_can_be_enqueued_again(
    queue: JobQueue, store: SqliteStore, ids: IdFactory, source_id: SourceId
) -> None:
    doc = add_document(store, ids, source_id, "a.md")
    first = queue.enqueue(JobKind.INGEST_DOC, source_id, doc)
    outcome = Worker(queue, ok).run_once()
    assert outcome is not None and outcome.status is JobStatus.DONE
    again = queue.enqueue(JobKind.INGEST_DOC, source_id, doc)
    assert again.id != first.id


def test_claim_lease_heartbeat_and_complete(
    queue: JobQueue, store: SqliteStore, clock: FakeClock, ids: IdFactory, source_id: SourceId
) -> None:
    doc = add_document(store, ids, source_id, "a.md")
    queue.enqueue(JobKind.INGEST_DOC, source_id, doc)
    job = queue.claim()
    assert job is not None and job.status is JobStatus.LEASED and job.attempts == 1
    assert job.lease_until == clock.now() + timedelta(seconds=300)
    assert queue.claim() is None  # nobody else can take it
    clock.advance(timedelta(seconds=200))
    assert queue.heartbeat(job)
    stored = store.reader().get_job(job.id)
    assert stored is not None and stored.lease_until == clock.now() + timedelta(seconds=300)
    assert queue.complete(job)
    done = store.reader().get_job(job.id)
    assert done is not None and done.status is JobStatus.DONE and done.finished_at is not None
    assert not queue.complete(job)  # completing twice is harmless and reported


def test_failures_back_off_exponentially_and_end_dead(
    queue: JobQueue, store: SqliteStore, clock: FakeClock, ids: IdFactory, source_id: SourceId
) -> None:
    doc = add_document(store, ids, source_id, "a.md")
    queued = queue.enqueue(JobKind.INGEST_DOC, source_id, doc)
    flaky = RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "down")  # retryable
    delays: list[float] = []
    for attempt in (1, 2):
        job = queue.claim()
        assert job is not None and job.attempts == attempt
        assert queue.fail(job, flaky) is JobStatus.FAILED
        failed = store.reader().get_job(queued.id)
        assert failed is not None and failed.error_code is ErrorCode.E_PROVIDER_UNAVAILABLE
        delays.append((failed.available_at - clock.now()).total_seconds())
        assert queue.claim() is None  # not before the retry time
        clock.advance(timedelta(seconds=failed.available_at.timestamp() - clock.now().timestamp()))
    assert delays == [30.0, 60.0]
    last = queue.claim()
    assert last is not None and last.attempts == 3
    assert queue.fail(last, flaky) is JobStatus.DEAD  # attempt limit reached
    dead = store.reader().get_job(queued.id)
    assert dead is not None and dead.status is JobStatus.DEAD and dead.finished_at is not None
    assert queue.claim() is None


def test_the_backoff_is_capped() -> None:
    policy = RetryPolicy(base_delay_s=30, factor=2, max_delay_s=100)
    assert [policy.delay_for(n).total_seconds() for n in (1, 2, 3, 4, 10)] == [
        30,
        60,
        100,
        100,
        100,
    ]
    with pytest.raises(RagError):
        RetryPolicy(max_attempts=0)


def test_a_poison_file_goes_dead_without_blocking_the_queue(
    queue: JobQueue, store: SqliteStore, ids: IdFactory, source_id: SourceId
) -> None:
    poison = add_document(store, ids, source_id, "poison.pdf")
    good = [add_document(store, ids, source_id, f"ok{i}.md") for i in range(3)]
    queue.enqueue(JobKind.INGEST_DOC, source_id, poison)
    for doc in good:
        queue.enqueue(JobKind.INGEST_DOC, source_id, doc)
    handled: list[str] = []

    def handler(job: IngestionJob, heartbeat: Callable[[], bool]) -> None:
        del heartbeat
        if job.document_id == poison:
            raise RagError(ErrorCode.E_PARSE_FAILED, "cannot parse")  # not retryable
        handled.append(str(job.document_id))

    outcomes = Worker(queue, handler).run_until_idle()
    assert [o.status for o in outcomes].count(JobStatus.DONE) == 3
    assert [o.status for o in outcomes].count(JobStatus.DEAD) == 1
    assert len(handled) == 3
    counts = queue.snapshot()
    assert counts == {JobStatus.DONE: 3, JobStatus.DEAD: 1}


def test_unexpected_handler_bugs_do_not_kill_the_worker(
    queue: JobQueue, store: SqliteStore, ids: IdFactory, source_id: SourceId
) -> None:
    doc = add_document(store, ids, source_id, "a.md")
    queued = queue.enqueue(JobKind.INGEST_DOC, source_id, doc)

    def buggy(job: IngestionJob, heartbeat: Callable[[], bool]) -> None:
        del job, heartbeat
        raise ZeroDivisionError("secret document text must not be stored")

    outcome = Worker(queue, buggy).run_once()
    assert outcome is not None and outcome.status is JobStatus.FAILED
    stored = store.reader().get_job(queued.id)
    assert stored is not None
    assert stored.error_code is ErrorCode.E_INTEGRITY
    assert "secret" not in repr(stored)


def test_an_expired_lease_is_recovered_and_the_stale_worker_is_fenced_off(
    queue: JobQueue, store: SqliteStore, clock: FakeClock, ids: IdFactory, source_id: SourceId
) -> None:
    doc = add_document(store, ids, source_id, "a.md")
    queue.enqueue(JobKind.INGEST_DOC, source_id, doc)
    stale = queue.claim()
    assert stale is not None
    clock.advance(timedelta(seconds=301))
    assert queue.recover() == 1
    fresh = queue.claim()
    assert fresh is not None and fresh.id == stale.id and fresh.attempts == 2
    # the first worker wakes up late: none of its calls may affect the job
    assert not queue.heartbeat(stale)
    assert not queue.complete(stale)
    assert queue.fail(stale, RagError(ErrorCode.E_TIMEOUT, "late")) is JobStatus.FAILED
    current = store.reader().get_job(fresh.id)
    assert current is not None and current.status is JobStatus.LEASED
    assert queue.complete(fresh)


def test_a_job_that_keeps_killing_its_worker_ends_dead(
    queue: JobQueue, store: SqliteStore, clock: FakeClock, ids: IdFactory, source_id: SourceId
) -> None:
    doc = add_document(store, ids, source_id, "crasher.md")
    queued = queue.enqueue(JobKind.INGEST_DOC, source_id, doc)
    for _ in range(3):  # max_attempts: the worker "dies" after every claim
        assert queue.claim() is not None
        clock.advance(timedelta(seconds=301))
    assert queue.claim() is None
    dead = store.reader().get_job(queued.id)
    assert dead is not None and dead.status is JobStatus.DEAD
    assert dead.error_code is ErrorCode.E_TIMEOUT


def test_a_crash_in_the_middle_of_processing_is_recovered_after_restart(
    tmp_path: Path, clock: FakeClock, ids: IdFactory
) -> None:
    db = tmp_path / "rag.db"
    with open_store(db, clock) as store:
        source = make_source(ids)
        with store.unit_of_work().begin() as writer:
            writer.put_source(source)
        doc = add_document(store, ids, source.id, "a.md")
        JobQueue(store, clock, ids).enqueue(JobKind.INGEST_DOC, source.id, doc)

    child = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from rag_engine.ingestion.jobs import JobQueue, Worker
        from rag_engine.storage.sqlite import open_store
        from rag_engine.testing import FakeClock
        from tests.factories import make_ids

        clock = FakeClock()
        store = open_store(Path({str(db)!r}), clock)
        queue = JobQueue(store, clock, make_ids(1_800_000_000_000))

        def handler(job, heartbeat):
            os._exit(9)  # killed halfway through the work

        Worker(queue, handler).run_once()
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", child],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
    )
    assert result.returncode == 9, result.stderr.decode()

    with open_store(db, clock) as store:
        queue = JobQueue(store, clock, ids)
        (stuck,) = store.reader().list_jobs()
        assert stuck.status is JobStatus.LEASED and stuck.attempts == 1  # found after restart
        assert queue.claim() is None  # the lease is still valid: nobody steals it early
        clock.advance(timedelta(seconds=301))
        seen: list[int] = []

        def handler(job: IngestionJob, heartbeat: Callable[[], bool]) -> None:
            del heartbeat
            seen.append(job.attempts)

        outcome = Worker(queue, handler).run_once()
        assert outcome is not None and outcome.status is JobStatus.DONE
        assert seen == [2]  # processed again, exactly once more
        (job,) = store.reader().list_jobs()  # no duplicate was created
        assert job.id == stuck.id and job.status is JobStatus.DONE


def test_two_workers_never_process_the_same_job(
    tmp_path: Path, clock: FakeClock, ids: IdFactory
) -> None:
    db = tmp_path / "rag.db"
    with open_store(db, clock) as seed:
        source = make_source(ids)
        with seed.unit_of_work().begin() as writer:
            writer.put_source(source)
        documents = [add_document(seed, ids, source.id, f"{i}.md") for i in range(30)]
        seeding = JobQueue(seed, clock, ids)
        for doc in documents:
            seeding.enqueue(JobKind.INGEST_DOC, source.id, doc)

    processed: list[str] = []
    guard = threading.Lock()
    errors: list[BaseException] = []

    def work(seed_value: int) -> None:
        local = SqliteStore(db, writer_timeout_s=10.0)
        try:
            local_queue = JobQueue(local, clock, make_ids(1_800_000_000_000 + seed_value))

            def handler(job: IngestionJob, heartbeat: Callable[[], bool]) -> None:
                del heartbeat
                with guard:
                    processed.append(str(job.id))

            Worker(local_queue, handler).run_until_idle()
        except BaseException as exc:
            errors.append(exc)
        finally:
            local.close()

    threads = [threading.Thread(target=work, args=(n,)) for n in (1, 2, 3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors
    assert len(processed) == 30 and len(set(processed)) == 30
    with open_store(db, clock) as store:
        assert store.reader().count_jobs() == {JobStatus.DONE: 30}


def test_workers_can_filter_by_kind(
    queue: JobQueue, store: SqliteStore, ids: IdFactory, source_id: SourceId
) -> None:
    doc = add_document(store, ids, source_id, "a.md")
    queue.enqueue(JobKind.INGEST_DOC, source_id, doc)
    queue.enqueue(JobKind.GC, source_id)
    outcomes = Worker(queue, ok, kind=JobKind.GC).run_until_idle()
    assert [o.job.kind for o in outcomes] == [JobKind.GC]
    assert queue.snapshot() == {JobStatus.DONE: 1, JobStatus.QUEUED: 1}


def test_losing_the_lease_while_working_is_reported_not_completed(
    queue: JobQueue, store: SqliteStore, clock: FakeClock, ids: IdFactory, source_id: SourceId
) -> None:
    doc = add_document(store, ids, source_id, "a.md")
    queue.enqueue(JobKind.INGEST_DOC, source_id, doc)

    def slow(job: IngestionJob, heartbeat: Callable[[], bool]) -> None:
        del job, heartbeat
        clock.advance(timedelta(seconds=400))  # the work outlives the lease...
        queue.recover()  # ...and another process recovers the job
        assert queue.claim() is not None  # ...and takes it over

    outcome = Worker(queue, slow).run_once()
    assert outcome is not None and outcome.status is JobStatus.LEASED  # lease lost
    (job,) = store.reader().list_jobs()
    assert job.status is JobStatus.LEASED and job.attempts == 2  # the new owner is untouched
