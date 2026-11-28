"""Structured, safe logging.

Every event is a JSON-serialisable mapping. A request scope binds ``request_id``
for all events emitted inside it (correlation across stages), and a stage scope
binds ``stage``. By default the processor chain

- accepts only constant-style event names (``stage.done``); anything else is
  replaced by ``log.event_redacted`` plus a digest and length of the original,
- replaces ``exc_info`` by diagnostics that cannot carry content: exception type,
  cause chain types, ``RagError`` code/category/retryable and traceback frames
  (file name, line, function) but never the exception message,
- redacts values stored under sensitive keys (document text, prompts, answers,
  secrets) and replaces any over-long string,

so user content cannot reach the logs by accident.
"""

from __future__ import annotations

import hashlib
import logging
import logging.handlers
import re
import sys
import traceback
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path, PurePath
from typing import Any, TextIO, cast

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
        "exception",
        "stack",
        "stack_info",
    }
)
_SENSITIVE_SUFFIXES = ("_text", "_prompt", "_content", "_body")
_SENSITIVE_PARTS = ("password", "secret", "api_key", "apikey", "credential")
_UNTOUCHED = frozenset({"event", "timestamp", "level"})
_EVENT_NAME = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,79}")
_MAX_TRACEBACK_FRAMES = 30
_MAX_CHAIN = 5


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


def sanitize_event(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Structlog processor: event names are constants, never interpolated content."""
    event = event_dict.get("event")
    if isinstance(event, str) and _EVENT_NAME.fullmatch(event):
        return event_dict
    original = str(event)
    event_dict["event"] = "log.event_redacted"
    event_dict["event_digest"] = text_digest(original)
    event_dict["event_chars"] = len(original)
    return event_dict


def _exception_from(exc_info: object) -> BaseException | None:
    if isinstance(exc_info, BaseException):
        return exc_info
    if exc_info is True:
        return sys.exc_info()[1]
    if isinstance(exc_info, tuple) and len(exc_info) == 3:
        candidate = exc_info[1]
        return candidate if isinstance(candidate, BaseException) else None
    return None


def _type_name(exc: BaseException) -> str:
    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def summarize_exception(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Structlog processor: replace ``exc_info`` by content-free diagnostics."""
    exc = _exception_from(event_dict.pop("exc_info", None))
    if exc is None:
        return event_dict
    event_dict["exception_type"] = _type_name(exc)
    if isinstance(exc, RagError):
        event_dict.setdefault("error_code", exc.code.value)
        event_dict.setdefault("error_category", exc.category.value)
        event_dict.setdefault("retryable", exc.retryable)
    chain: list[str] = []
    cause = exc.__cause__ or exc.__context__
    while cause is not None and len(chain) < _MAX_CHAIN:
        chain.append(_type_name(cause))
        cause = cause.__cause__ or cause.__context__
    if chain:
        event_dict["exception_chain"] = chain
    frames = traceback.extract_tb(exc.__traceback__)[-_MAX_TRACEBACK_FRAMES:]
    if frames:
        event_dict["traceback"] = [
            f"{PurePath(f.filename).name}:{f.lineno} in {f.name}" for f in frames
        ]
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


class RotatingStream:
    """A text stream that writes to a file and rotates it by size (``max_bytes``, ``backups``).

    The file is created readable by its owner only: logs can mention identifiers and error codes.
    """

    def __init__(self, path: Path, max_bytes: int, backups: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
        path.chmod(0o600)

    def write(self, text: str) -> int:
        if not text.strip("\n"):
            return len(text)  # ``print`` writes the line and its newline separately
        record = logging.LogRecord("rag", logging.INFO, "", 0, text.rstrip("\n"), None, None)
        self._handler.handle(record)
        return len(text)

    def flush(self) -> None:
        self._handler.flush()

    def close(self) -> None:
        self._handler.close()


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = True,
    stream: TextIO | RotatingStream | None = None,
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
        sanitize_event,
        summarize_exception,
        redact_sensitive,
        renderer,
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(levels[level.upper()]),
        logger_factory=structlog.PrintLoggerFactory(file=cast(TextIO, stream or sys.stderr)),
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
