"""Search and answers (JSON or server-sent events) over the same services as the command line."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from rag_engine.api import schemas
from rag_engine.api.problems import problem, status_for
from rag_engine.api.routes_manage import state_of
from rag_engine.api.state import ApiState, QueryBackend
from rag_engine.domain import (
    AskStatus,
    ErrorCode,
    RagError,
    RequestId,
    RerankMode,
    RetrievalFilters,
    RetrievalMode,
    RetrievalQuery,
)
from rag_engine.pipeline import FinalEvent
from rag_engine.pipeline.presentation import answer_to_dict, event_to_dict
from rag_engine.retrieval.presentation import load_details, outcome_to_dict, parse_filters

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _filters(body: schemas.FilterIn) -> RetrievalFilters:
    return parse_filters(
        sources=body.sources,
        path_prefix=body.path_prefix,
        path_glob=body.path_glob,
        tags_any=body.tags_any,
        tags_all=body.tags_all,
        language=body.language,
        kinds=body.kinds,
        modified_after=body.modified_after,
        modified_before=body.modified_before,
        doc_ids=body.doc_ids,
        exclude_doc_ids=body.exclude_doc_ids,
        cli=False,
    )


def _query(
    backend: QueryBackend,
    request_id: RequestId,
    text: str,
    mode: str | None,
    top_k: int | None,
    filters: schemas.FilterIn,
    rerank: bool | None,
    *,
    debug: bool = False,
) -> RetrievalQuery:
    defaults = backend.search.defaults
    depth = top_k or defaults.top_k
    return RetrievalQuery(
        request_id=request_id,
        text=text,
        mode=RetrievalMode(mode) if mode else defaults.mode,
        top_k=depth,
        candidate_k=max(defaults.candidate_k, depth),
        filters=_filters(filters),
        rerank=RerankMode.AUTO if rerank is None else RerankMode.ON if rerank else RerankMode.OFF,
        debug=debug,
    )


def _check_length(state: ApiState, text: str) -> None:
    if len(text) > state.bind.max_question_chars:
        raise RagError(
            ErrorCode.E_QUERY_INVALID,
            f"the question exceeds {state.bind.max_question_chars} characters",
        )


def _sse(name: str, data: dict[str, Any]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/search", response_model=schemas.SearchOut, tags=["query"], summary="Search")
    async def search(body: schemas.SearchIn, request: Request) -> dict[str, Any]:
        state = state_of(request)
        _check_length(state, body.query)
        backend = await run_in_threadpool(state.query_backend)
        query = _query(
            backend, request.state.request_id, body.query, body.mode, body.top_k, body.filters,
            body.rerank, debug=body.debug,
        )  # fmt: skip
        outcome = await backend.search.retrieval.search(query)
        details = await run_in_threadpool(load_details, backend.search.store, outcome.result)
        return outcome_to_dict(outcome, details, debug=body.debug)

    @router.post(
        "/answers",
        response_model=schemas.AnswerOut,
        tags=["query"],
        summary="Answer a question with citations, or refuse",
        responses={
            200: {
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
                "description": "JSON, or server-sent events when `stream` is true: "
                "evidence, delta*, retract?, citations, final",
            }
        },
    )
    async def answers(body: schemas.AnswerIn, request: Request) -> Any:
        state = state_of(request)
        _check_length(state, body.question)
        backend = await run_in_threadpool(state.query_backend)
        if backend.pipeline is None:
            raise RagError(ErrorCode.E_CONFIG_INVALID, "no language model is configured: set [llm]")
        query = _query(
            backend, request.state.request_id, body.question, body.mode, body.top_k, body.filters,
            body.rerank,
        )  # fmt: skip
        pipeline = backend.pipeline
        if body.stream:
            return StreamingResponse(
                _events(pipeline.stream(query, verified_only=not body.live)),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        answer = await pipeline.ask(query)
        if answer.status is AskStatus.ERROR and answer.error is not None:
            error = answer.error
            code = error.code
            return problem(
                request,
                status_for(code),
                code.value,
                error.message,
                category="provider",
                retryable=error.retryable,
                context={"stage": error.stage},
            )
        return answer_to_dict(
            answer, sources=body.include_sources, show_unverified=body.include_unverified_text
        )

    return router


async def _events(stream: AsyncIterator[Any]) -> AsyncIterator[bytes]:
    async for event in stream:
        data = event_to_dict(event)
        name = str(data.pop("event"))
        yield _sse(name, data)
        if isinstance(event, FinalEvent):
            return
