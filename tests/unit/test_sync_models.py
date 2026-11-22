"""Domain models and invariants added for Phase 2: sync state, jobs, index metadata."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from rag_engine.domain import (
    ContentHash,
    Document,
    DocumentStatus,
    ErrorCode,
    IdFactory,
    IndexMetadata,
    JobId,
    JobKind,
    JobStage,
    JobStatus,
    ModelStatus,
    RagError,
    SourceRevision,
    SyncStatus,
    thaw_context,
)
from rag_engine.domain.invariants import (
    check_generation_monotonic,
    check_single_active_model,
    check_sync_state_matches_document,
    check_version_transition,
)
from rag_engine.domain.models import EmbeddingModel
from tests.factories import T0, amend, make_document, make_job, make_sync_state

NAIVE = datetime(2026, 1, 1)  # noqa: DTZ001 - deliberately timezone-naive
PLUS_ONE = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=1)))


def raises_integrity(factory: object) -> None:
    with pytest.raises(RagError) as info:
        factory()  # type: ignore[operator]
    assert info.value.code is ErrorCode.E_INTEGRITY


def test_job_ids_are_typed_uuid7_and_increase(ids: IdFactory) -> None:
    first, second = ids.job_id(), ids.job_id()
    assert isinstance(first, JobId) and first < second
    assert JobId.parse(str(first)) == first


class TestSyncState:
    def test_valid_states(self, ids: IdFactory) -> None:
        source_id = ids.source_id()
        ok = make_sync_state(
            source_id,
            last_revision=SourceRevision(3, 4),
            last_content_hash=ContentHash.of_bytes(b"x"),
        )
        assert ok.status is SyncStatus.OK
        failed = make_sync_state(
            source_id,
            status=SyncStatus.FAILED,
            last_error_code=ErrorCode.E_UNSUPPORTED_FORMAT,
            next_retry_at=T0 + timedelta(hours=1),
            attempts=2,
        )
        assert failed.attempts == 2

    @pytest.mark.parametrize(
        "changes",
        [
            {"attempts": -1},
            {"status": SyncStatus.FAILED},  # a failure needs a code
            {"next_retry_at": T0},  # only for failed
            {"last_seen_at": NAIVE},
            {"last_seen_at": PLUS_ONE},
            {
                "status": SyncStatus.FAILED,
                "last_error_code": ErrorCode.E_TIMEOUT,
                "next_retry_at": NAIVE,
            },
        ],
    )
    def test_violations(self, ids: IdFactory, changes: dict[str, object]) -> None:
        raises_integrity(lambda: make_sync_state(ids.source_id(), "a.md", **changes))


class TestIngestionJob:
    def test_defaults_describe_a_fresh_job(self, ids: IdFactory) -> None:
        job = make_job(ids, ids.source_id())
        assert (job.status, job.stage, job.attempts) == (
            JobStatus.QUEUED,
            JobStage.DISCOVERED,
            0,
        )
        assert JobStatus.QUEUED.is_active and not JobStatus.DONE.is_active

    def test_payload_is_deeply_immutable_and_json_friendly(self, ids: IdFactory) -> None:
        job = make_job(ids, ids.source_id(), payload={"mime": "text/plain", "n": [1, {"a": 2}]})
        with pytest.raises(TypeError):
            job.payload["x"] = 1  # type: ignore[index]
        assert thaw_context(job.payload) == {"mime": "text/plain", "n": [1, {"a": 2}]}

    def test_state_machine_invariants(self, ids: IdFactory) -> None:
        sid = ids.source_id()
        leased = make_job(ids, sid, status=JobStatus.LEASED, lease_until=T0 + timedelta(seconds=5))
        assert leased.lease_until is not None
        done = make_job(ids, sid, status=JobStatus.DONE, finished_at=T0)
        assert done.finished_at == T0
        failed = make_job(ids, sid, status=JobStatus.FAILED, error_code=ErrorCode.E_TIMEOUT)
        assert failed.error_code is ErrorCode.E_TIMEOUT
        for changes in (
            {"status": JobStatus.LEASED},  # lease without a deadline
            {"lease_until": T0},  # deadline without a lease
            {"status": JobStatus.DONE},  # done without finished_at
            {"status": JobStatus.DEAD},
            {"finished_at": T0},  # finished but still queued
            {"error_code": ErrorCode.E_TIMEOUT},  # error on a queued job
            {"attempts": -1},
            {"available_at": NAIVE},
        ):
            raises_integrity(lambda c=changes: make_job(ids, sid, **c))

    def test_ingest_doc_needs_a_document(self, ids: IdFactory) -> None:
        job = make_job(ids, ids.source_id(), kind=JobKind.GC)
        assert job.document_id is None
        raises_integrity(lambda: amend(make_job(ids, ids.source_id()), document_id=None))


class TestIndexMetadata:
    def make(self, **changes: object) -> IndexMetadata:
        base = IndexMetadata(
            schema_version=1,
            index_generation=0,
            active_embedding_model_id=None,
            lexical_analyzer_id="none",
            chunker_default_fingerprint="none",
            vector_store_kind="none",
            created_at=T0,
        )
        return replace(base, **changes)  # type: ignore[arg-type]

    def test_valid_and_frozen(self) -> None:
        meta = self.make(calibration={"k": [1, 2]})
        assert thaw_context(meta.calibration) == {"k": [1, 2]}
        with pytest.raises(TypeError):
            meta.calibration["x"] = 1  # type: ignore[index]

    @pytest.mark.parametrize(
        "changes",
        [
            {"schema_version": 0},
            {"index_generation": -1},
            {"created_at": NAIVE},
            {"last_gc_at": NAIVE},
        ],
    )
    def test_violations(self, changes: dict[str, object]) -> None:
        raises_integrity(lambda: self.make(**changes))

    def test_generation_is_monotonic(self) -> None:
        old = self.make(index_generation=3)
        check_generation_monotonic(old, self.make(index_generation=3))
        check_generation_monotonic(old, self.make(index_generation=4))
        raises_integrity(lambda: check_generation_monotonic(old, self.make(index_generation=2)))


class TestDocumentWithoutVersion:
    def awaiting(self, ids: IdFactory) -> Document:
        return make_document(
            document_id=ids.document_id(),
            source_id=ids.source_id(),
            version_id=None,
            status=DocumentStatus.ACTIVE,
        )

    def test_active_without_a_version_is_valid_but_not_searchable(self, ids: IdFactory) -> None:
        document = self.awaiting(ids)
        assert document.current_version_id is None
        assert not document.is_searchable
        with_version = amend(document, current_version_id=ids.document_version_id())
        assert with_version.is_searchable
        for status in (DocumentStatus.FAILED, DocumentStatus.DELETED):
            assert not amend(with_version, status=status).is_searchable

    def test_first_commit_moves_null_to_a_version(self, ids: IdFactory) -> None:
        before = self.awaiting(ids)
        after = amend(before, current_version_id=ids.document_version_id())
        check_version_transition(None, before)  # first discovery
        check_version_transition(before, before)  # still waiting
        check_version_transition(before, after)  # first successful commit
        newer = amend(after, current_version_id=ids.document_version_id())
        check_version_transition(after, newer)  # an update

    def test_a_document_never_loses_its_version(self, ids: IdFactory) -> None:
        before = self.awaiting(ids)
        after = amend(before, current_version_id=ids.document_version_id())
        raises_integrity(lambda: check_version_transition(after, before))
        raises_integrity(lambda: check_version_transition(after, self.awaiting(ids)))


def test_sync_state_must_match_its_document(ids: IdFactory) -> None:
    source_id = ids.source_id()
    document = make_document(
        document_id=ids.document_id(),
        source_id=source_id,
        version_id=None,
        status=DocumentStatus.ACTIVE,
        key="a.md",
    )
    check_sync_state_matches_document(document, make_sync_state(source_id, "a.md"))
    raises_integrity(
        lambda: check_sync_state_matches_document(document, make_sync_state(source_id, "other.md"))
    )
    raises_integrity(
        lambda: check_sync_state_matches_document(
            document, make_sync_state(source_id, "a.md", status=SyncStatus.MISSING)
        )
    )


def test_at_most_one_active_embedding_model() -> None:
    def model(name: str, status: ModelStatus) -> EmbeddingModel:
        from rag_engine.domain import EmbeddingModelId

        mid = EmbeddingModelId.build(
            provider="p",
            name=name,
            revision="1",
            dimension=8,
            normalized=True,
            query_prefix="",
            document_prefix="",
        )
        return EmbeddingModel(
            id=mid,
            provider="p",
            name=name,
            revision="1",
            dimension=8,
            normalized=True,
            max_input_tokens=128,
            status=status,
            created_at=T0,
        )

    active, retired = model("a", ModelStatus.ACTIVE), model("b", ModelStatus.RETIRED)
    check_single_active_model([], indexing_started=False)
    check_single_active_model([active, retired], indexing_started=True)
    raises_integrity(
        lambda: check_single_active_model(
            [active, model("c", ModelStatus.ACTIVE)], indexing_started=False
        )
    )
    raises_integrity(lambda: check_single_active_model([retired], indexing_started=True))
