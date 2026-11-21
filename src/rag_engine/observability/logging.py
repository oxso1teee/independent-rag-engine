"""Structured, safe logging.

Every event is a JSON-serialisable mapping. A request scope binds ``request_id``
for all events emitted inside it (correlation across stages), and a stage scope
binds ``stage``. By default the processor chain redacts values stored under
sensitive keys (document text, prompts, answers, secrets) and replaces any
over-long string, so user content cannot reach the logs by accident.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any, TextIO

import structlog
from structlog.contextvars import bound_contextvars, merge_contextvars
from structlog.typing import EventDict, FilteringBoundLogger, Processor

from rag_engine.domain import RagError, RequestId
from rag_engine.observability.timing import Timer
from rag_engine.ports import Clock

REDACTED = "[REDACTED]"
MAX_VALUE_CHARS = 200

_SENSITIVE_EXACT = frozenset(
    {
        "text",
        "content",
        "body",
        "prompt",
        "question",
        "query",
        "answer",
        "completion",
        "document",
        "normalized_text",
        "token",
        "access_token",
        "refresh_token",
        "bearer",
        "password",
        "authorization",
    }
)
_SENSITIVE_SUFFIXES = ("_text", "_prompt", "_content", "_body")
_SENSITIVE_PARTS = ("password", "secret", "api_key", "apikey", "credential")
_UNTOUCHED = frozenset({"event", "exception", "stack", "timestamp", "level"})


def is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in _SENSITIVE_EXACT
        or lowered.endswith(_SENSITIVE_SUFFIXES)
        or any(part in lowered for part in _SENSITIVE_PARTS)
    )


def _scrub(value: object, depth: int = 0) -> object:
    if isinstance(value, str):
        if len(value) > MAX_VALUE_CHARS:
            return f"[TRUNCATED {len(value)} chars]"
        return value
    if depth >= 4:
        return "[NESTED]"
    if isinstance(value, dict):
        return {
            str(k): REDACTED if is_sensitive_key(str(k)) else _scrub(v, depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        return [_scrub(v, depth + 1) for v in value]
    return value


def redact_sensitive(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Structlog processor: redact sensitive keys and cap string sizes."""
    for key in list(event_dict):
        if key in _UNTOUCHED:
            continue
        event_dict[key] = REDACTED if is_sensitive_key(key) else _scrub(event_dict[key])
    return event_dict


def text_digest(text: str) -> str:
    """Short content fingerprint, safe to log in place of the text itself."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _timestamper(clock: Clock | None) -> Processor:
    if clock is None:
        return structlog.processors.TimeStamper(fmt="iso", utc=True)

    def add_timestamp(
        _: Any, __: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        event_dict["timestamp"] = clock.now().isoformat()
        return event_dict

    return add_timestamp


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = True,
    stream: TextIO | None = None,
    clock: Clock | None = None,
) -> None:
    """Configure structlog globally. Safe to call repeatedly."""
    levels = logging.getLevelNamesMapping()
    if level.upper() not in levels:
        raise ValueError(f"unknown log level: {level}")
    renderer: Processor = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    processors: list[Processor] = [
        merge_contextvars,
        structlog.processors.add_log_level,
        _timestamper(clock),
        redact_sensitive,
        structlog.processors.format_exc_info,
        renderer,
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(levels[level.upper()]),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    logger: FilteringBoundLogger = structlog.get_logger(name) if name else structlog.get_logger()
    return logger


@contextmanager
def request_scope(request_id: RequestId) -> Iterator[None]:
    """Attach ``request_id`` to every event emitted inside the block."""
    with bound_contextvars(request_id=str(request_id)):
        yield


@contextmanager
def stage_scope(
    logger: FilteringBoundLogger,
    name: str,
    *,
    clock: Clock | None = None,
    **fields: object,
) -> Iterator[None]:
    """Log ``stage.start`` / ``stage.done`` (or ``stage.failed``) around a pipeline stage."""
    timer = Timer(clock) if clock is not None else None

    def on_done() -> dict[str, object]:
        return {"duration_ms": timer.elapsed_ms()} if timer is not None else {}

    with bound_contextvars(stage=name):
        logger.info("stage.start", **fields)
        try:
            yield
        except RagError as exc:
            logger.error(
                "stage.failed",
                error_code=exc.code.value,
                error_category=exc.category.value,
                retryable=exc.retryable,
                **on_done(),
            )
            raise
        except Exception as exc:
            logger.error("stage.failed", error_type=type(exc).__name__, **on_done())
            raise
        logger.info("stage.done", **on_done())
