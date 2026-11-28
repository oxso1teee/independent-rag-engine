"""The HTTP API application: wiring, authentication, limits and the OpenAPI document."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPBearer
from starlette.middleware.cors import CORSMiddleware

from rag_engine import __version__
from rag_engine.api import problems, routes_manage, routes_query, schemas
from rag_engine.api.auth import authorised, resolve_token
from rag_engine.api.limits import BodyLimitMiddleware, IdempotencyStore
from rag_engine.api.state import ApiState
from rag_engine.cli.context import Container
from rag_engine.config import Settings
from rag_engine.config.settings import ApiSettings
from rag_engine.system import system_id_factory

__all__ = ["create_app", "resolve_token"]

_bearer = HTTPBearer(auto_error=False, description="Bearer token from the api.token_env variable")

# Fixed wording: the standard library's phrases differ between Python versions (422 changed in
# 3.13), which would make the committed OpenAPI snapshot depend on the interpreter.
_ERROR_DESCRIPTIONS = {
    400: "The request is invalid (for example a bad filter or cursor)",
    401: "A valid bearer token is required",
    404: "Nothing matches the request",
    409: "The request conflicts with the current state",
    413: "The request body exceeds the size limit",
    422: "The request does not match the schema",
    429: "A model server is rate limiting",
    500: "An internal error occurred",
    502: "A model server failed or returned an unusable answer",
    503: "The service is not ready, not configured, locked or out of resources",
    504: "A stage ran out of time",
}
COMMON_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {
        "model": schemas.Problem,
        "description": text,
        "content": {"application/problem+json": {}},
    }
    for code, text in _ERROR_DESCRIPTIONS.items()
}
DESCRIPTION = """\
Local-first retrieval-augmented generation over your own documents.

Errors are `application/problem+json` (RFC 9457) with the engine's stable `code`. An answer has
one of four statuses: `answered`, `insufficient_evidence`, `unverified` or `error`; only an
`answered` result carries text, and every citation in it points at evidence of the same request.
"""


def create_app(
    container: Container,
    config: Path | None,
    db: Path | None,
    settings: Settings,
    *,
    token: str | None,
    bind: ApiSettings | None = None,
) -> FastAPI:
    api_settings = bind or settings.api
    path = db if db is not None else settings.storage.db_path
    state = ApiState(
        container=container,
        config=config,
        db=path,
        settings=settings,
        bind=api_settings,
        token=token,
        managed_root=path.resolve().parent / "inline",
        idempotency=IdempotencyStore(api_settings.idempotency_ttl_s),
    )
    ids = system_id_factory()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await asyncio.to_thread(state.close)

    app = FastAPI(
        title="Independent RAG Engine API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/v1/openapi.json",
    )
    app.state.api = state
    problems.install(app)

    async def require_auth(request: Request, _: Annotated[Any, Depends(_bearer)]) -> None:
        if not authorised(request, state.token):
            raise HTTPException(
                status_code=401,
                detail="a valid bearer token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.middleware("http")
    async def request_id(request: Request, call_next: Any) -> Any:
        request.state.request_id = ids.request_id()
        response = await call_next(request)
        response.headers["X-Request-ID"] = str(request.state.request_id)
        return response

    app.add_middleware(BodyLimitMiddleware, max_bytes=api_settings.max_body_bytes)
    if api_settings.cors_origins:  # CORS is off unless origins are listed explicitly
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(api_settings.cors_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
        )

    open_router = routes_manage.build_router()
    protected = Depends(require_auth)
    for route in list(open_router.routes):
        # Health stays open for supervisors; everything else needs the token when one is set.
        if getattr(route, "path", "") != "/health":
            route.dependencies.append(protected)  # type: ignore[attr-defined]
    app.include_router(open_router, prefix="/v1", responses=COMMON_ERRORS)
    app.include_router(
        routes_query.build_router(), prefix="/v1", dependencies=[protected], responses=COMMON_ERRORS
    )
    return app
