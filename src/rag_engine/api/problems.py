"""Errors as ``application/problem+json`` (RFC 9457) with the engine's stable codes.

Every failure the API reports has the same shape: ``type``, ``title``, ``status``, ``detail`` and
the extensions ``code`` (the stable error code), ``category``, ``retryable`` and ``request_id``.
The detail is the error's own message, which never carries document text, secrets or file paths;
unexpected exceptions become a generic 500 whose detail says nothing about the cause.
"""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from types import MappingProxyType
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_engine.domain import ErrorCode, RagError
from rag_engine.observability import get_logger

PROBLEM_JSON = "application/problem+json"
_log = get_logger("rag_engine.api")

HTTP_STATUS: Mapping[ErrorCode, int] = MappingProxyType(
    {
        ErrorCode.E_CONFIG_INVALID: 503,
        ErrorCode.E_SOURCE_NOT_FOUND: 404,
        ErrorCode.E_SOURCE_LOCATOR_CHANGED: 409,
        ErrorCode.E_MASS_DELETE_GUARD: 409,
        ErrorCode.E_PARSE_FAILED: 422,
        ErrorCode.E_UNSUPPORTED_FORMAT: 415,
        ErrorCode.E_EMBED_DIM_MISMATCH: 500,
        ErrorCode.E_EMBED_MODEL_MISMATCH: 409,
        ErrorCode.E_PROVIDER_TIMEOUT: 504,
        ErrorCode.E_PROVIDER_UNAVAILABLE: 502,
        ErrorCode.E_RATE_LIMITED: 429,
        ErrorCode.E_CONTEXT_BUDGET: 422,
        ErrorCode.E_CONTEXT_TRUNCATED: 502,
        ErrorCode.E_DEGENERATE_ANSWER: 502,
        ErrorCode.E_CITATION_INVALID: 502,
        ErrorCode.E_FILTER_INVALID: 400,
        ErrorCode.E_QUERY_INVALID: 400,
        ErrorCode.E_RESOURCE_GATE: 503,
        ErrorCode.E_STORE_LOCKED: 503,
        ErrorCode.E_MIGRATION_REQUIRED: 503,
        ErrorCode.E_INTEGRITY: 500,
        ErrorCode.E_TIMEOUT: 504,
    }
)


def status_for(code: ErrorCode) -> int:
    return HTTP_STATUS[code]


def _request_id(request: Request) -> str | None:
    value = getattr(request.state, "request_id", None)
    return None if value is None else str(value)


def _body(
    status: int,
    code: str,
    detail: str,
    *,
    category: str,
    retryable: bool,
    request: Request,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": f"urn:rag:problem:{code.lower().replace('_', '-')}",
        "title": HTTPStatus(status).phrase,
        "status": status,
        "detail": detail,
        "code": code,
        "category": category,
        "retryable": retryable,
        "instance": request.url.path,
        "request_id": _request_id(request),
    }
    if context:
        body["context"] = dict(context)
    return body


def problem(
    request: Request,
    status: int,
    code: str,
    detail: str,
    *,
    category: str = "user",
    retryable: bool = False,
    headers: Mapping[str, str] | None = None,
    context: Mapping[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        _body(
            status,
            code,
            detail,
            category=category,
            retryable=retryable,
            request=request,
            context=context,
        ),
        status_code=status,
        media_type=PROBLEM_JSON,
        headers=dict(headers or {}),
    )


def rag_problem(request: Request, error: RagError) -> JSONResponse:
    status = status_for(error.code)
    headers = {"Retry-After": "1"} if error.retryable and status == 503 else None
    context = None if error.category.value == "internal" else error.to_dict()["context"]
    return problem(
        request,
        status,
        error.code.value,
        error.message,
        category=error.category.value,
        retryable=error.retryable,
        headers=headers,
        context=context if isinstance(context, dict) else None,
    )


def install(app: FastAPI) -> None:
    @app.exception_handler(RagError)
    async def on_rag_error(request: Request, error: RagError) -> JSONResponse:
        return rag_problem(request, error)

    @app.exception_handler(RequestValidationError)
    async def on_invalid(request: Request, error: RequestValidationError) -> JSONResponse:
        fields = [
            {"field": ".".join(str(p) for p in item["loc"]), "message": item["msg"]}
            for item in error.errors()
        ]
        return problem(
            request,
            422,
            "E_QUERY_INVALID",
            "the request does not match the schema",
            context={"errors": fields},
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, error: StarletteHTTPException) -> JSONResponse:
        code = {401: "E_UNAUTHORIZED", 404: "E_NOT_FOUND", 405: "E_METHOD_NOT_ALLOWED"}.get(
            error.status_code, "E_HTTP"
        )
        return problem(
            request,
            error.status_code,
            code,
            str(error.detail),
            headers=error.headers,
        )

    @app.exception_handler(Exception)
    async def on_unexpected(request: Request, error: Exception) -> JSONResponse:
        _log.error("api.unhandled", error_type=type(error).__name__)
        return problem(
            request,
            500,
            "E_INTERNAL",
            "an internal error occurred",
            category="internal",
            retryable=False,
        )
