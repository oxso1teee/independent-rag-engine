"""Health, readiness, statistics, sources, documents and jobs."""

from __future__ import annotations

import base64
import binascii
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import APIRouter, BackgroundTasks, Header, Query, Request, Response
from fastapi.responses import JSONResponse

from rag_engine import __version__
from rag_engine.api import schemas
from rag_engine.api.limits import IdempotencyStore
from rag_engine.api.problems import problem
from rag_engine.api.state import ApiState
from rag_engine.domain import (
    Document,
    DocumentId,
    DocumentStatus,
    ErrorCode,
    IngestionJob,
    JobStatus,
    RagError,
    Source,
    SourceKind,
    normalize_source_key,
)
from rag_engine.ingestion.service import collect_status
from rag_engine.storage.sqlite import StoreNotInitialised

ALLOWED_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"})
MAX_PAGE = 200
IdempotencyHeader = Annotated[
    str | None, Header(alias="Idempotency-Key", min_length=1, max_length=128)
]


def state_of(request: Request) -> ApiState:
    state = request.app.state.api
    assert isinstance(state, ApiState)  # noqa: S101 - set by create_app
    return state


def _iso(value: datetime) -> str:
    return value.isoformat()


def _managed(state: ApiState, source: Source) -> bool:
    if source.kind is not SourceKind.FS or source.locator is None:
        return False
    try:
        return Path(source.locator).resolve().is_relative_to(state.managed_root.resolve())
    except OSError:
        return False


def _source_out(state: ApiState, source: Source) -> schemas.SourceOut:
    return schemas.SourceOut(
        id=str(source.id),
        name=source.name,
        kind=source.kind.value,
        managed=_managed(state, source),
        created_at=_iso(source.created_at),
    )


def _document_out(source_name: str, document: Document) -> schemas.DocumentOut:
    return schemas.DocumentOut(
        id=str(document.id),
        source=source_name,
        key=str(document.source_key),
        status=document.status.value,
        title=document.title,
        tags=list(document.tags),
        current_version=None
        if document.current_version_id is None
        else str(document.current_version_id),
        indexed=document.current_version_id is not None,
        updated_at=_iso(document.updated_at),
    )


def _job_out(job: IngestionJob, names: dict[str, str]) -> schemas.JobOut:
    return schemas.JobOut(
        id=str(job.id),
        kind=job.kind.value,
        source=names.get(str(job.source_id)),
        document_id=None if job.document_id is None else str(job.document_id),
        stage=job.stage.value,
        status=job.status.value,
        attempts=job.attempts,
        error_code=None if job.error_code is None else job.error_code.value,
        created_at=_iso(job.created_at),
        finished_at=None if job.finished_at is None else _iso(job.finished_at),
    )


def _encode_cursor(**fields: str) -> str:
    return base64.urlsafe_b64encode(json.dumps(fields).encode()).decode().rstrip("=")


def _decode_cursor(cursor: str | None) -> dict[str, str]:
    if cursor is None:
        return {}
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, binascii.Error):
        raise RagError(ErrorCode.E_QUERY_INVALID, "the cursor is not valid") from None
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise RagError(ErrorCode.E_QUERY_INVALID, "the cursor is not valid")
    return data


def build_router() -> APIRouter:
    router = APIRouter()

    # ------------------------------------------------------------------- health

    @router.get("/health", response_model=schemas.Health, tags=["health"], summary="Liveness")
    def health() -> schemas.Health:
        return schemas.Health(status="ok", version=__version__)

    @router.get(
        "/ready",
        response_model=schemas.Readiness,
        tags=["health"],
        summary="Readiness for queries",
        responses={503: {"model": schemas.Readiness}},
    )
    def ready(request: Request, response: Response) -> schemas.Readiness:
        state = state_of(request)
        checks = {
            "store": False,
            "index_published": False,
            "embedding_configured": state.settings.embedding.enabled,
            "llm_configured": state.settings.llm.enabled,
        }
        detail: dict[str, str] = {}
        try:
            with state.container.open(state.config, state.db, read_only=True) as services:
                checks["store"] = True
                metadata = services.store.reader().get_index_metadata()
                checks["index_published"] = metadata is not None and metadata.index_generation > 0
        except StoreNotInitialised:
            detail["store"] = "no database yet"
        except RagError as error:
            detail["store"] = error.code.value
        if not checks["index_published"]:
            detail.setdefault("index_published", "nothing has been indexed yet")
        ok = checks["store"] and checks["index_published"] and checks["embedding_configured"]
        response.status_code = 200 if ok else 503
        return schemas.Readiness(ready=ok, checks=checks, detail=detail)

    @router.get("/stats", response_model=schemas.Stats, tags=["health"], summary="Index statistics")
    def stats(request: Request) -> schemas.Stats:
        state = state_of(request)
        try:
            with state.container.open(state.config, state.db, read_only=True) as services:
                data = collect_status(services.store.reader()).to_dict()
        except StoreNotInitialised:
            return schemas.Stats(schema_version=None, index_generation=None, sources=[], jobs={})
        return schemas.Stats(
            schema_version=cast("int | None", data["schema_version"]),
            index_generation=cast("int | None", data["index_generation"]),
            sources=[
                {k: v for k, v in item.items() if k != "locator"}  # no server paths in responses
                for item in cast("list[dict[str, Any]]", data["sources"])
            ],
            jobs=cast("dict[str, int]", data["jobs"]),
        )

    # ------------------------------------------------------------------ sources

    @router.get("/sources", response_model=list[schemas.SourceOut], tags=["sources"])
    def list_sources(request: Request) -> list[schemas.SourceOut]:
        state = state_of(request)
        try:
            with state.container.open(state.config, state.db, read_only=True) as services:
                return [_source_out(state, s) for s in services.registry.list()]
        except StoreNotInitialised:
            return []

    @router.post(
        "/sources", response_model=schemas.SourceOut, status_code=201, tags=["sources"],
        summary="Create a managed source for uploaded documents",
    )  # fmt: skip
    def create_source(body: schemas.SourceCreate, request: Request) -> schemas.SourceOut:
        state = state_of(request)
        root = state.managed_root / body.name
        root.mkdir(parents=True, exist_ok=True)
        with state.container.open(state.config, state.db, read_only=False) as services:
            source = services.registry.add(body.name, root)
        return _source_out(state, source)

    @router.get("/sources/{name}", response_model=schemas.SourceOut, tags=["sources"])
    def get_source(name: str, request: Request) -> schemas.SourceOut:
        state = state_of(request)
        with state.container.open(state.config, state.db, read_only=True) as services:
            return _source_out(state, services.registry.get(name))

    @router.post(
        "/sources/{name}/ingest", status_code=202, response_model=schemas.Accepted,
        tags=["sources"], summary="Scan a source and queue what changed",
    )  # fmt: skip
    def ingest_source(
        name: str, request: Request, background: BackgroundTasks, accept_mass_delete: bool = False
    ) -> schemas.Accepted:
        state = state_of(request)
        with state.container.open(state.config, state.db, read_only=False) as services:
            report = services.ingest.run(name, accept_mass_delete=accept_mass_delete)
        background.add_task(state.drain_queue)
        return schemas.Accepted(ingest=report.to_dict(), indexing="scheduled")

    # ---------------------------------------------------------------- documents

    @router.get("/documents", response_model=schemas.Page[schemas.DocumentOut], tags=["documents"])
    def list_documents(
        request: Request,
        source: str | None = None,
        status: DocumentStatus | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50,
        cursor: str | None = None,
    ) -> schemas.Page[schemas.DocumentOut]:
        state = state_of(request)
        position = _decode_cursor(cursor)
        items: list[schemas.DocumentOut] = []
        next_cursor: str | None = None
        try:
            with state.container.open(state.config, state.db, read_only=True) as services:
                reader = services.store.reader()
                for src in reader.list_sources():
                    if source is not None and src.name != source:
                        continue
                    if "source" in position and src.name < position["source"]:
                        continue
                    after = (
                        normalize_source_key(position["key"])
                        if position.get("source") == src.name and "key" in position
                        else None
                    )
                    room = limit + 1 - len(items)
                    for document in reader.list_documents(src.id, status, after, room):
                        items.append(_document_out(src.name, document))
                    if len(items) > limit:
                        break
        except StoreNotInitialised:
            return schemas.Page[schemas.DocumentOut](items=[])
        if len(items) > limit:
            items = items[:limit]
            next_cursor = _encode_cursor(source=items[-1].source, key=items[-1].key)
        return schemas.Page[schemas.DocumentOut](items=items, next_cursor=next_cursor)

    @router.get(
        "/documents/{document_id}", response_model=schemas.DocumentDetail, tags=["documents"]
    )
    def get_document(document_id: str, request: Request) -> schemas.DocumentDetail:
        state = state_of(request)
        try:
            wanted = DocumentId.parse(document_id)
        except (ValueError, RagError):
            raise RagError(ErrorCode.E_QUERY_INVALID, "the document id is not valid") from None
        with state.container.open(state.config, state.db, read_only=True) as services:
            reader = services.store.reader()
            document = reader.get_document(wanted)
            source = None if document is None else reader.get_source(document.source_id)
            if document is None or source is None:
                raise RagError(ErrorCode.E_SOURCE_NOT_FOUND, "no such document")
            versions = [
                schemas.VersionOut(
                    id=str(v.id),
                    seq=v.seq,
                    content_hash=str(v.content_hash),
                    size_bytes=v.size_bytes,
                    mime=v.mime,
                    created_at=_iso(v.created_at),
                    parse_warnings=list(v.parse_warnings),
                )
                for v in reader.list_versions(document.id)
            ]
            chunks = (
                None
                if document.current_version_id is None
                else len(reader.list_chunks(document.current_version_id))
            )
            base = _document_out(source.name, document)
        return schemas.DocumentDetail(**base.model_dump(), versions=versions, chunks=chunks)

    @router.post(
        "/documents", status_code=202, response_model=schemas.Accepted, tags=["documents"],
        summary="Add or replace a document in a managed source",
        responses={409: {"description": "Idempotency-Key reused with a different body"}},
    )  # fmt: skip
    async def create_document(
        body: schemas.DocumentCreate,
        request: Request,
        background: BackgroundTasks,
        idempotency_key: IdempotencyHeader = None,
    ) -> Response:
        state = state_of(request)
        raw = await request.body()
        fingerprint = IdempotencyStore.fingerprint(request.method.encode(), raw)
        scope = f"{state.token or 'open'}:{idempotency_key}"
        if idempotency_key is not None:
            verdict, stored = state.idempotency.lookup(scope, fingerprint)
            if verdict == "conflict":
                return problem(
                    request, 409, "E_IDEMPOTENCY_CONFLICT",
                    "this Idempotency-Key was used with a different request body",
                )  # fmt: skip
            if verdict == "replay" and stored is not None:
                return JSONResponse(
                    stored.body, status_code=stored.status, headers={"Idempotent-Replay": "true"}
                )
        accepted = _store_document(state, body)
        background.add_task(state.drain_queue)
        payload = accepted.model_dump(mode="json")
        if idempotency_key is not None:
            state.idempotency.remember(scope, fingerprint, 202, payload)
        return JSONResponse(payload, status_code=202)

    @router.delete(
        "/documents/{document_id}", status_code=202, response_model=schemas.Accepted,
        tags=["documents"], summary="Remove a document of a managed source",
    )  # fmt: skip
    def delete_document(
        document_id: str, request: Request, background: BackgroundTasks
    ) -> schemas.Accepted:
        state = state_of(request)
        try:
            wanted = DocumentId.parse(document_id)
        except (ValueError, RagError):
            raise RagError(ErrorCode.E_QUERY_INVALID, "the document id is not valid") from None
        with state.container.open(state.config, state.db, read_only=False) as services:
            reader = services.store.reader()
            document = reader.get_document(wanted)
            source = None if document is None else reader.get_source(document.source_id)
            if document is None or source is None:
                raise RagError(ErrorCode.E_SOURCE_NOT_FOUND, "no such document")
            _require_managed(state, source)
            _path_for(state, source, str(document.source_key)).unlink(missing_ok=True)
            report = services.ingest.run(source.name, accept_mass_delete=True)
        background.add_task(state.drain_queue)
        return schemas.Accepted(ingest=report.to_dict(), indexing="scheduled")

    # --------------------------------------------------------------------- jobs

    @router.get("/jobs", response_model=schemas.Page[schemas.JobOut], tags=["jobs"])
    def list_jobs(
        request: Request,
        status: JobStatus | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50,
        cursor: str | None = None,
    ) -> schemas.Page[schemas.JobOut]:
        state = state_of(request)
        after = _decode_cursor(cursor).get("after")
        try:
            with state.container.open(state.config, state.db, read_only=True) as services:
                reader = services.store.reader()
                names = {str(s.id): s.name for s in reader.list_sources()}
                jobs = [j for j in reader.list_jobs(status) if after is None or str(j.id) > after][
                    : limit + 1
                ]
        except StoreNotInitialised:
            return schemas.Page[schemas.JobOut](items=[])
        more = len(jobs) > limit
        shown = jobs[:limit]
        return schemas.Page[schemas.JobOut](
            items=[_job_out(j, names) for j in shown],
            next_cursor=_encode_cursor(after=str(shown[-1].id)) if more else None,
        )

    @router.get("/jobs/{job_id}", response_model=schemas.JobOut, tags=["jobs"])
    def get_job(job_id: str, request: Request) -> schemas.JobOut:
        from rag_engine.domain import JobId

        state = state_of(request)
        try:
            wanted = JobId.parse(job_id)
        except (ValueError, RagError):
            raise RagError(ErrorCode.E_QUERY_INVALID, "the job id is not valid") from None
        with state.container.open(state.config, state.db, read_only=True) as services:
            reader = services.store.reader()
            job = reader.get_job(wanted)
            if job is None:
                raise RagError(ErrorCode.E_SOURCE_NOT_FOUND, "no such job")
            return _job_out(job, {str(s.id): s.name for s in reader.list_sources()})

    return router


# ----------------------------------------------------------------------- managed files


def _require_managed(state: ApiState, source: Source) -> None:
    if not _managed(state, source):
        raise RagError(
            ErrorCode.E_SOURCE_LOCATOR_CHANGED,
            "this source is managed outside the API; change its files, not the API",
            context={"source": source.name},
        )


def _path_for(state: ApiState, source: Source, key: str) -> Path:
    """The file of ``key`` inside a managed source; refuses anything that leaves its root."""
    parts = key.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise RagError(ErrorCode.E_QUERY_INVALID, "the document key is not a clean relative path")
    assert source.locator is not None  # noqa: S101 - a managed source has a root
    root = Path(source.locator).resolve()
    target = (root / key).resolve()
    if not target.is_relative_to(root):
        raise RagError(ErrorCode.E_QUERY_INVALID, "the document key leaves its source")
    return target


def _store_document(state: ApiState, body: schemas.DocumentCreate) -> schemas.Accepted:
    try:
        skey = normalize_source_key(body.key)
    except RagError:  # a domain invariant: empty, dot or dot-dot segment
        raise RagError(
            ErrorCode.E_QUERY_INVALID, "the document key is not a clean relative path"
        ) from None
    key = str(skey)
    if Path(key).suffix.lower() not in ALLOWED_SUFFIXES:
        raise RagError(
            ErrorCode.E_UNSUPPORTED_FORMAT,
            "the document key needs a supported extension",
            context={"allowed": sorted(ALLOWED_SUFFIXES)},
        )
    if body.text is not None:
        data = body.text.encode("utf-8")
    else:
        try:
            data = base64.b64decode(body.content_base64 or "", validate=True)
        except (ValueError, binascii.Error):
            raise RagError(
                ErrorCode.E_QUERY_INVALID, "content_base64 is not valid base64"
            ) from None
    if not data:
        raise RagError(ErrorCode.E_QUERY_INVALID, "the document is empty")
    if len(data) > state.bind.max_body_bytes:
        raise RagError(ErrorCode.E_QUERY_INVALID, "the document exceeds the size limit")
    with state.container.open(state.config, state.db, read_only=False) as services:
        source = services.registry.get(body.source)
        _require_managed(state, source)
        target = _path_for(state, source, key)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".upload-")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        report = services.ingest.run(source.name)
        document = services.store.reader().get_document_by_key(source.id, skey)
        out = None if document is None else _document_out(source.name, document)
    return schemas.Accepted(document=out, ingest=report.to_dict(), indexing="scheduled")
