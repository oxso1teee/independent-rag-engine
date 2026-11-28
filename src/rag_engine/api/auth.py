"""Bearer-token authentication and the rule that a public bind needs one.

With no token set the API is open, which is acceptable only on the loopback interface; the
settings and ``resolve_token`` refuse any other bind without a token. Tokens are compared in
constant time and never logged or echoed. Health is open so that a supervisor can probe it.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Mapping

from fastapi import Request

from rag_engine.config.settings import ApiSettings, is_loopback_host
from rag_engine.domain import ErrorCode, RagError

MIN_TOKEN_CHARS = 16


def resolve_token(settings: ApiSettings, environ: Mapping[str, str] | None = None) -> str | None:
    """The bearer token from the environment variable the settings name, or ``None``.

    Refuses a non-loopback bind without a token and a configured-but-missing or weak token.
    """
    env = os.environ if environ is None else environ
    loopback = is_loopback_host(settings.host)
    token = "" if settings.token_env is None else env.get(settings.token_env, "")
    if not token:
        if not loopback:
            raise RagError(
                ErrorCode.E_CONFIG_INVALID,
                "a token is required to listen on a non-loopback address: set the environment "
                f"variable {settings.token_env or 'named by api.token_env'}",
            )
        return None
    if len(token) < MIN_TOKEN_CHARS:
        raise RagError(
            ErrorCode.E_CONFIG_INVALID,
            f"the environment variable {settings.token_env} must hold a token of at least "
            f"{MIN_TOKEN_CHARS} characters",
        )
    return token


def presented(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    return value.strip() if scheme.lower() == "bearer" and value.strip() else None


def authorised(request: Request, token: str | None) -> bool:
    if token is None:
        return True
    given = presented(request)
    return given is not None and secrets.compare_digest(given.encode(), token.encode())
