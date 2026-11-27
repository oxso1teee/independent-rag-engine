"""Error mapping shared by the chat adapters.

Every failure becomes a ``RagError`` with a stable code, and the split matters to callers: what is
worth retrying (the server is unreachable, restarting or overloaded) is marked retryable, what is
not (a model that is not installed, a rejected request, a runner that ran out of memory) is not,
so a retry never repeats an out-of-memory crash. Response bodies are reduced to a short excerpt and
no header ever reaches an error, so an API key cannot leak through a message.
"""

from __future__ import annotations

import httpx

from rag_engine.domain import ContextValue, ErrorCode, RagError

_EXCERPT = 200
_MEMORY_HINTS = ("out of memory", "unable to allocate", "not enough memory", "requires more")


def excerpt(text: str, secret: str | None = None) -> str:
    flat = " ".join(text.split())
    if secret:
        flat = flat.replace(secret, "***")
    return flat[:_EXCERPT] + ("…" if len(flat) > _EXCERPT else "")


def transport_error(provider: str, exc: httpx.HTTPError) -> RagError:
    if isinstance(exc, httpx.TimeoutException):
        return RagError(ErrorCode.E_PROVIDER_TIMEOUT, f"{provider} request timed out")
    return RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, f"{provider} is unreachable")


def status_error(provider: str, status: int, body: str, secret: str | None = None) -> RagError:
    """The error for an unsuccessful HTTP status, given a bounded body excerpt (key removed)."""
    detail = excerpt(body, secret)
    context: dict[str, ContextValue] = {"status": status, "detail": detail}
    if status == 429:
        return RagError(ErrorCode.E_RATE_LIMITED, f"{provider} rate limit", context=context)
    if _looks_like_memory_failure(detail):
        return RagError(
            ErrorCode.E_RESOURCE_GATE,
            f"{provider} could not load or run the model for lack of memory",
            context=context,
        )
    if status >= 500:
        return RagError(
            ErrorCode.E_PROVIDER_UNAVAILABLE, f"{provider} failed", retryable=True, context=context
        )
    if status in {401, 403}:
        return RagError(
            ErrorCode.E_CONFIG_INVALID, f"{provider} refused the credentials", context=context
        )
    if status == 404:
        return RagError(
            ErrorCode.E_PROVIDER_UNAVAILABLE,
            f"{provider} does not know the model or endpoint",
            retryable=False,
            context=context,
        )
    return RagError(
        ErrorCode.E_PROVIDER_UNAVAILABLE,
        f"{provider} rejected the request",
        retryable=False,
        context=context,
    )


def stream_error(provider: str, message: str, secret: str | None = None) -> RagError:
    """A failure the server reported inside an already started stream: never retried."""
    detail = excerpt(message, secret)
    if _looks_like_memory_failure(detail):
        return RagError(
            ErrorCode.E_RESOURCE_GATE,
            f"{provider} ran out of memory while generating",
            context={"detail": detail},
        )
    return RagError(
        ErrorCode.E_PROVIDER_UNAVAILABLE,
        f"{provider} failed while streaming",
        retryable=False,
        context={"detail": detail},
    )


def invalid_response(provider: str, what: str) -> RagError:
    return RagError(
        ErrorCode.E_PROVIDER_UNAVAILABLE,
        f"{provider} returned an invalid response: {what}",
        retryable=False,
    )


def _looks_like_memory_failure(detail: str) -> bool:
    lowered = detail.lower()
    return any(hint in lowered for hint in _MEMORY_HINTS)
