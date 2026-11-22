"""Persistent job queue: leases, heartbeats, retries with backoff, dead letters.

Every state change is one unit of work, so a crash at any point leaves the queue
consistent: a job is either still queued, leased (and recovered once its lease expires),
retried later, done or dead. A job that keeps crashing its worker is declared dead after
``max_attempts`` instead of blocking the queue.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import timedelta

from rag_engine.domain import (
    ContextValue,
    DocumentId,
    ErrorCode,
    IdFactory,
    IngestionJob,
    JobKind,
    JobStatus,
    RagError,
    SourceId,
)
from rag_engine.ports import Clock, Store


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff without jitter (deterministic): ``base * factor ** (attempt - 1)``."""

    max_attempts: int = 5
    base_delay_s: float = 30.0
    factor: float = 2.0
    max_delay_s: float = 3600.0
    lease_s: float = 300.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.base_delay_s < 0 or self.factor < 1 or self.lease_s <= 0:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "invalid retry policy")

    def delay_for(self, attempt: int) -> timedelta:
        seconds = min(self.max_delay_s, self.base_delay_s * self.factor ** max(attempt - 1, 0))
        return timedelta(seconds=seconds)


class JobQueue:
    """Operations on the persistent queue (the unit of work is opened per operation)."""

    def __init__(
        self, store: Store, clock: Clock, ids: IdFactory, policy: RetryPolicy | None = None
    ) -> None:
        self._store = store
        self._clock = clock
        self._ids = ids
        self.policy = policy or RetryPolicy()

    def enqueue(
        self,
        kind: JobKind,
        source_id: SourceId,
        document_id: DocumentId | None = None,
        payload: Mapping[str, ContextValue] | None = None,
    ) -> IngestionJob:
        """Idempotent: an active job for the same (kind, source, document) is returned as is."""
        now = self._clock.now()
        job = IngestionJob(
            id=self._ids.job_id(),
            kind=kind,
            source_id=source_id,
            document_id=document_id,
            payload=payload or {},
            available_at=now,
            created_at=now,
        )
        with self._store.unit_of_work().begin() as writer:
            return writer.enqueue_job(job)

    def recover(self) -> int:
        """Re-queue (or kill, after too many attempts) jobs whose lease expired."""
        with self._store.unit_of_work().begin() as writer:
            return writer.release_expired_leases(self._clock.now(), self.policy.max_attempts)

    def claim(self, kind: JobKind | None = None) -> IngestionJob | None:
        """Lease the next claimable job; expired leases are recovered first."""
        now = self._clock.now()
        with self._store.unit_of_work().begin() as writer:
            writer.release_expired_leases(now, self.policy.max_attempts)
            return writer.lease_next_job(kind, now, self.policy.lease_s)

    def heartbeat(self, job: IngestionJob) -> bool:
        """Extend the lease; ``False`` means the lease was lost and the work must stop."""
        with self._store.unit_of_work().begin() as writer:
            return writer.heartbeat_job(
                job.id, job.attempts, self._clock.now(), self.policy.lease_s
            )

    def complete(self, job: IngestionJob) -> bool:
        with self._store.unit_of_work().begin() as writer:
            return writer.complete_job(job.id, job.attempts, self._clock.now())

    def fail(self, job: IngestionJob, error: RagError) -> JobStatus:
        """Record a failed attempt; returns the resulting status (``failed`` or ``dead``)."""
        now = self._clock.now()
        retry_at = None
        if error.retryable and job.attempts < self.policy.max_attempts:
            retry_at = now + self.policy.delay_for(job.attempts)
        with self._store.unit_of_work().begin() as writer:
            writer.fail_job(job.id, job.attempts, error.code, now, retry_at)
        return JobStatus.DEAD if retry_at is None else JobStatus.FAILED

    def snapshot(self) -> dict[JobStatus, int]:
        return self._store.reader().count_jobs()


JobHandler = Callable[[IngestionJob, Callable[[], bool]], None]
"""``handler(job, heartbeat)``; ``heartbeat()`` extends the lease and returns whether it is held."""


@dataclass(frozen=True, slots=True)
class JobOutcome:
    job: IngestionJob
    status: JobStatus  # done, failed (will retry) or dead
    error_code: ErrorCode | None = None


class Worker:
    """Pulls jobs and runs the handler; failures never escape and never block the queue."""

    def __init__(self, queue: JobQueue, handler: JobHandler, kind: JobKind | None = None) -> None:
        self._queue = queue
        self._handler = handler
        self._kind = kind

    def run_once(self) -> JobOutcome | None:
        """Process at most one job; ``None`` when nothing is claimable."""
        job = self._queue.claim(self._kind)
        if job is None:
            return None
        try:
            self._handler(job, lambda: self._queue.heartbeat(job))
        except RagError as error:
            return JobOutcome(job, self._queue.fail(job, error), error.code)
        except Exception:
            # An unexpected bug must not kill the worker or the queue. The message is not
            # kept (it could contain document text); the job is retried, bounded by
            # ``max_attempts``.
            unexpected = RagError(
                ErrorCode.E_INTEGRITY, "unexpected error while processing a job", retryable=True
            )
            return JobOutcome(job, self._queue.fail(job, unexpected), unexpected.code)
        if not self._queue.complete(job):
            # The lease was lost while working: someone else owns the job now.
            return JobOutcome(replace(job), JobStatus.LEASED, ErrorCode.E_TIMEOUT)
        return JobOutcome(job, JobStatus.DONE)

    def run_until_idle(self, max_jobs: int = 1000) -> list[JobOutcome]:
        outcomes: list[JobOutcome] = []
        for _ in range(max_jobs):
            outcome = self.run_once()
            if outcome is None:
                break
            outcomes.append(outcome)
        return outcomes
