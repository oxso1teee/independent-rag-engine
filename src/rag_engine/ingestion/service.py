"""Ingestion use cases: scan a source, apply the detected changes, report on the state.

``IngestService.run`` is one scan: discover, detect changes, check the mass-delete guard,
then apply everything in a single unit of work (documents, sync states, tombstones and the
jobs that plan the processing). Nothing is written when nothing changed. ``plan`` does the
same without writing anything (``--dry-run``). Parsing, chunking and embedding are later
phases: the jobs created here wait in the queue for them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from rag_engine.domain import (
    Document,
    DocumentStatus,
    ErrorCode,
    IdFactory,
    IngestionJob,
    JobKind,
    RagError,
    Source,
    SourceKey,
    SourceRevision,
    SyncState,
    SyncStatus,
)
from rag_engine.ingestion.change import (
    ChangeDetector,
    ChangeKind,
    ChangePlan,
    DetectedChange,
    Move,
    documents_by_key,
    sync_states_by_key,
)
from rag_engine.ingestion.identity import AdapterFactory, SourceRegistry, match_moves
from rag_engine.ingestion.lifecycle import (
    Lifecycle,
    check_mass_delete,
    restore,
    tombstone,
)
from rag_engine.observability import get_logger, request_scope, stage_scope
from rag_engine.ports import Clock, MetadataReader, MetadataWriter, Store

_log = get_logger("rag_engine.ingestion")


def _counts(report: IngestReport) -> dict[str, int]:
    """Numbers only: events never carry paths or document text."""
    return {
        k: v for k, v in report.to_dict().items() if isinstance(v, int) and not isinstance(v, bool)
    }


def _revision(change: DetectedChange) -> SourceRevision:
    if change.revision is None:  # pragma: no cover - new entries always carry a revision
        raise RagError(ErrorCode.E_INTEGRITY, "a new entry without a revision")
    return change.revision


@dataclass(frozen=True, slots=True)
class IngestReport:
    source: str
    dry_run: bool
    new: int
    modified: int
    unchanged: int
    deleted: int
    inaccessible: int
    unsupported: int
    moved: int
    restored: int
    jobs_enqueued: int
    skipped: int
    gc_removed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "dry_run": self.dry_run,
            "new": self.new,
            "modified": self.modified,
            "unchanged": self.unchanged,
            "deleted": self.deleted,
            "inaccessible": self.inaccessible,
            "unsupported": self.unsupported,
            "moved": self.moved,
            "restored": self.restored,
            "jobs_enqueued": self.jobs_enqueued,
            "skipped": self.skipped,
            "gc_removed": self.gc_removed,
        }


class IngestService:
    def __init__(
        self,
        store: Store,
        clock: Clock,
        ids: IdFactory,
        adapter_factory: AdapterFactory,
    ) -> None:
        self._store = store
        self._clock = clock
        self._ids = ids
        self._adapter_factory = adapter_factory
        self._registry = SourceRegistry(store, clock, ids, adapter_factory)
        self._lifecycle = Lifecycle(store, clock)

    # -- public ----------------------------------------------------------------------

    def plan(self, name: str, *, accept_mass_delete: bool = False) -> IngestReport:
        """What a run would do; reads the store and the files, writes nothing."""
        with request_scope(self._ids.request_id()):
            with stage_scope(_log, "ingest.scan", clock=self._clock, dry_run=True):
                source, plan = self._detect(name, self._store.reader(), accept_mass_delete)
            report = self._report(source, plan, dry_run=True, jobs=0, gc=0)
            _log.info("ingest.planned", **_counts(report))
            return report

    def run(self, name: str, *, accept_mass_delete: bool = False) -> IngestReport:
        with request_scope(self._ids.request_id()):
            with stage_scope(_log, "ingest.scan", clock=self._clock, dry_run=False):
                source, plan = self._detect(name, self._store.reader(), accept_mass_delete)
            jobs = 0
            if plan.mutates:
                with (
                    stage_scope(_log, "ingest.apply", clock=self._clock),
                    self._store.unit_of_work().begin() as writer,
                ):
                    jobs = self._apply(writer, source, plan)
            with stage_scope(_log, "ingest.gc", clock=self._clock):
                gc = self._lifecycle.collect_garbage(source).removed
            report = self._report(source, plan, dry_run=False, jobs=jobs, gc=gc)
            _log.info("ingest.done", **_counts(report))
            return report

    # -- detection --------------------------------------------------------------------

    def _detect(
        self, name: str, reader: MetadataReader, accept_mass_delete: bool
    ) -> tuple[Source, ChangePlan]:
        source = self._registry.get(name)
        adapter = self._adapter_factory(source)
        self._registry.verify(source, adapter)
        states = sync_states_by_key(reader.list_sync_states(source.id))
        documents = documents_by_key(reader.list_documents(source.id))
        report = adapter.scan({k: s.last_revision for k, s in states.items() if s.last_revision})
        plan = ChangeDetector(adapter).detect(source.id, report, states, documents)
        plan = self._with_moves(plan, states, documents)
        tracked = sum(1 for s in states.values() if s.status is not SyncStatus.MISSING)
        check_mass_delete(
            vanished=len(plan.of_kind(ChangeKind.DELETED)),
            tracked=tracked,
            entries_seen=plan.entries_seen,
            accept=accept_mass_delete,
        )
        return source, plan

    @staticmethod
    def _with_moves(
        plan: ChangePlan, states: dict[SourceKey, SyncState], documents: dict[SourceKey, Document]
    ) -> ChangePlan:
        vanished = [
            (c.key, documents[c.key].id, states[c.key].last_content_hash)
            for c in plan.of_kind(ChangeKind.DELETED)
            if c.key in documents and states[c.key].last_content_hash is not None
        ]
        appeared = [
            (c.key, c.content_hash)
            for c in plan.of_kind(ChangeKind.NEW)
            if c.content_hash is not None and c.error is None
        ]
        held_elsewhere = [
            state.last_content_hash
            for key, state in states.items()
            if state.last_content_hash is not None
            and state.status is not SyncStatus.MISSING
            and key not in {v[0] for v in vanished}
        ]
        found, _ = match_moves(
            [(k, d, h) for k, d, h in vanished if h is not None], appeared, held_elsewhere
        )
        if not found:
            return plan
        by_new = {c.key: c for c in plan.of_kind(ChangeKind.NEW)}
        moves = tuple(
            Move(document_id, old, new, _revision(by_new[new]), digest)
            for old, document_id, new, digest in found
        )
        gone = {m.old_key for m in moves} | {m.new_key for m in moves}
        remaining = tuple(c for c in plan.changes if c.key not in gone)
        return replace(plan, changes=remaining, moves=moves)

    # -- application ------------------------------------------------------------------

    def _apply(self, writer: MetadataWriter, source: Source, plan: ChangePlan) -> int:
        now = self._clock.now()
        jobs = 0
        for move in plan.moves:
            document = writer.get_document(move.document_id)
            state = writer.get_sync_state(source.id, move.old_key)
            if document is None or state is None:
                continue
            writer.delete_sync_state(source.id, move.old_key)
            writer.put_document(replace(document, source_key=move.new_key, updated_at=now))
            writer.put_sync_state(
                replace(
                    state, source_key=move.new_key, last_revision=move.revision, last_seen_at=now
                )
            )
        for change in plan.changes:
            document = writer.get_document_by_key(source.id, change.key)
            state = writer.get_sync_state(source.id, change.key)
            match change.kind:
                case ChangeKind.NEW | ChangeKind.MODIFIED:
                    jobs += self._apply_content(writer, source, change, document, state, now)
                case ChangeKind.UNCHANGED:
                    jobs += self._apply_unchanged(writer, source, change, document, state, now)
                case ChangeKind.DELETED:
                    if document is not None:
                        tombstone(writer, document, now)
                case ChangeKind.INACCESSIBLE:
                    self._apply_inaccessible(writer, source, change, state, now)
        return jobs

    def _apply_content(
        self,
        writer: MetadataWriter,
        source: Source,
        change: DetectedChange,
        document: Document | None,
        state: SyncState | None,
        now: datetime,
    ) -> int:
        if change.revision is None or change.content_hash is None:  # pragma: no cover
            raise RagError(ErrorCode.E_INTEGRITY, "a content change without revision or hash")
        unsupported = change.error is ErrorCode.E_UNSUPPORTED_FORMAT
        if document is None:
            document = Document(
                id=self._ids.document_id(),
                source_id=source.id,
                source_key=change.key,
                current_version_id=None,
                status=DocumentStatus.FAILED if unsupported else DocumentStatus.ACTIVE,
                created_at=now,
                updated_at=now,
            )
        else:
            status = DocumentStatus.FAILED if unsupported else DocumentStatus.ACTIVE
            document = replace(document, status=status, missing_since=None, updated_at=now)
        writer.put_document(document)
        writer.put_sync_state(
            SyncState(
                source_id=source.id,
                source_key=change.key,
                last_revision=change.revision,
                last_content_hash=change.content_hash,
                last_seen_at=now,
                status=SyncStatus.FAILED if unsupported else SyncStatus.CHANGED,
                attempts=0,
                last_error_code=ErrorCode.E_UNSUPPORTED_FORMAT if unsupported else None,
            )
        )
        if unsupported:
            return 0
        return self._enqueue(writer, source, document, change, now)

    def _apply_unchanged(
        self,
        writer: MetadataWriter,
        source: Source,
        change: DetectedChange,
        document: Document | None,
        state: SyncState | None,
        now: datetime,
    ) -> int:
        if state is None or document is None:
            return 0
        jobs = 0
        if change.restored:
            document = restore(document, now)
            writer.put_document(document)
            status = (
                SyncStatus.OK if document.current_version_id is not None else SyncStatus.CHANGED
            )
            state = replace(state, status=status, last_seen_at=now)
            writer.put_sync_state(state)
        elif change.touch and change.revision is not None:
            writer.put_sync_state(replace(state, last_revision=change.revision, last_seen_at=now))
        return jobs

    def _apply_inaccessible(
        self,
        writer: MetadataWriter,
        source: Source,
        change: DetectedChange,
        state: SyncState | None,
        now: datetime,
    ) -> None:
        if state is None:
            return  # never seen and unreadable: nothing to remember yet
        if (
            state.status is SyncStatus.FAILED
            and state.last_error_code is ErrorCode.E_SOURCE_NOT_FOUND
        ):
            return  # already recorded; rescans stay write-free
        if state.status is SyncStatus.MISSING:
            return
        writer.put_sync_state(
            replace(
                state,
                status=SyncStatus.FAILED,
                attempts=state.attempts + 1,
                last_error_code=ErrorCode.E_SOURCE_NOT_FOUND,
                last_seen_at=now,
            )
        )

    def _enqueue(
        self,
        writer: MetadataWriter,
        source: Source,
        document: Document,
        change: DetectedChange,
        now: datetime,
    ) -> int:
        mime = change.mime
        payload: dict[str, str | int] = {
            "content_hash": change.content_hash.hex if change.content_hash else ""
        }
        if mime is not None:
            payload["mime"] = mime.mime
            payload["detected_by"] = mime.detected_by
            if mime.encoding is not None:
                payload["encoding"] = mime.encoding
        if change.revision is not None:
            payload["size_bytes"] = change.revision.size_bytes
        job = IngestionJob(
            id=self._ids.job_id(),
            kind=JobKind.INGEST_DOC,
            source_id=source.id,
            document_id=document.id,
            payload=payload,
            available_at=now,
            created_at=now,
        )
        stored = writer.enqueue_job(job)
        return 1 if stored.id == job.id else 0

    # -- reporting --------------------------------------------------------------------

    @staticmethod
    def _report(
        source: Source, plan: ChangePlan, *, dry_run: bool, jobs: int, gc: int
    ) -> IngestReport:
        counts = plan.counts()
        unsupported = sum(1 for c in plan.changes if c.error is ErrorCode.E_UNSUPPORTED_FORMAT)
        inaccessible = counts["inaccessible"]
        return IngestReport(
            source=source.name,
            dry_run=dry_run,
            new=counts["new"] - sum(1 for c in plan.of_kind(ChangeKind.NEW) if c.error),
            modified=counts["modified"]
            - sum(1 for c in plan.of_kind(ChangeKind.MODIFIED) if c.error),
            unchanged=counts["unchanged"],
            deleted=counts["deleted"],
            inaccessible=inaccessible,
            unsupported=unsupported,
            moved=counts["moved"],
            restored=sum(1 for c in plan.changes if c.restored),
            jobs_enqueued=jobs,
            skipped=plan.skipped,
            gc_removed=gc,
        )


@dataclass(frozen=True, slots=True)
class SourceStatus:
    name: str
    locator: str | None
    documents: dict[str, int]
    awaiting_first_ingest: int  # active documents without a committed version yet
    sync: dict[str, int]


@dataclass(frozen=True, slots=True)
class StoreStatus:
    sources: tuple[SourceStatus, ...]
    jobs: dict[str, int]
    schema_version: int | None
    index_generation: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "index_generation": self.index_generation,
            "jobs": self.jobs,
            "sources": [
                {
                    "name": s.name,
                    "locator": s.locator,
                    "documents": s.documents,
                    "awaiting_first_ingest": s.awaiting_first_ingest,
                    "sync": s.sync,
                }
                for s in self.sources
            ],
        }


def collect_status(reader: MetadataReader) -> StoreStatus:
    """Counts of documents, sync states and jobs; read-only."""
    sources: list[SourceStatus] = []
    for source in reader.list_sources():
        documents = {
            status.value: count
            for status in DocumentStatus
            if (count := reader.count_documents(source.id, status))
        }
        sync: dict[str, int] = {}
        for state in reader.list_sync_states(source.id):
            sync[state.status.value] = sync.get(state.status.value, 0) + 1
        sources.append(
            SourceStatus(
                source.name,
                source.locator,
                documents,
                reader.count_documents(source.id, DocumentStatus.ACTIVE, has_version=False),
                dict(sorted(sync.items())),
            )
        )
    metadata = reader.get_index_metadata()
    jobs = {status.value: n for status, n in sorted(reader.count_jobs().items())}
    return StoreStatus(
        sources=tuple(sources),
        jobs=jobs,
        schema_version=None if metadata is None else metadata.schema_version,
        index_generation=None if metadata is None else metadata.index_generation,
    )


__all__ = [
    "IngestReport",
    "IngestService",
    "SourceStatus",
    "StoreStatus",
    "collect_status",
]
