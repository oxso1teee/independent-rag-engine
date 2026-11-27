"""Central error taxonomy.

Every failure surfaced by the engine is a :class:`RagError` carrying a stable
machine-readable code, a category, a retryable flag, a human-readable message
and a small scalar context. Codes are part of the public contract: they are
never renamed or reused.

Error ``context`` must never contain document text, prompts or secrets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

type ContextValue = (
    str | int | float | bool | Sequence[ContextValue] | Mapping[str, ContextValue] | None
)


def _freeze(value: object) -> ContextValue:
    """Deeply immutable copy: mappings become read-only views, sequences become tuples."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, ContextValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("error context keys must be strings")
            frozen[key] = _freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return tuple(_freeze(item) for item in value)
    raise TypeError(f"unsupported error context value: {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, ContextValue]) -> Mapping[str, ContextValue]:
    frozen = _freeze(value)
    return frozen if isinstance(frozen, Mapping) else MappingProxyType({})


def _thaw(value: ContextValue) -> object:
    """Plain, mutable, JSON-friendly copy of a frozen context value."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_thaw(item) for item in value]
    return value


def freeze_context(value: Mapping[str, ContextValue]) -> Mapping[str, ContextValue]:
    """Deeply immutable copy of a context mapping (shared by models that carry payloads)."""
    return _freeze_mapping(value)


def thaw_context(value: Mapping[str, ContextValue]) -> dict[str, object]:
    """Plain, JSON-friendly copy of a frozen context mapping."""
    thawed = _thaw(value)
    return thawed if isinstance(thawed, dict) else {}


class ParseReason(StrEnum):
    """Stable tokens that say *why* a document could not be parsed (never document text)."""

    CORRUPT = "corrupt"
    ENCRYPTED = "encrypted"
    NO_TEXT_LAYER = "no_text_layer"
    TOO_LARGE = "too_large"
    TOO_MANY_PAGES = "too_many_pages"
    TOO_MANY_BLOCKS = "too_many_blocks"
    TOO_DEEP = "too_deep"
    ZIP_BOMB = "zip_bomb"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    WORKER_FAILED = "worker_failed"
    ENCODING = "encoding"


class ErrorCategory(StrEnum):
    USER = "user"
    CONFIG = "config"
    SOURCE = "source"
    PARSE = "parse"
    PROVIDER = "provider"
    RESOURCE = "resource"
    STORAGE = "storage"
    INTEGRITY = "integrity"
    INTERNAL = "internal"


class ErrorCode(StrEnum):
    E_CONFIG_INVALID = "E_CONFIG_INVALID"
    E_SOURCE_NOT_FOUND = "E_SOURCE_NOT_FOUND"
    E_SOURCE_LOCATOR_CHANGED = "E_SOURCE_LOCATOR_CHANGED"
    E_MASS_DELETE_GUARD = "E_MASS_DELETE_GUARD"
    E_PARSE_FAILED = "E_PARSE_FAILED"
    E_UNSUPPORTED_FORMAT = "E_UNSUPPORTED_FORMAT"
    E_EMBED_DIM_MISMATCH = "E_EMBED_DIM_MISMATCH"
    E_EMBED_MODEL_MISMATCH = "E_EMBED_MODEL_MISMATCH"
    E_PROVIDER_TIMEOUT = "E_PROVIDER_TIMEOUT"
    E_PROVIDER_UNAVAILABLE = "E_PROVIDER_UNAVAILABLE"
    E_RATE_LIMITED = "E_RATE_LIMITED"
    E_CONTEXT_BUDGET = "E_CONTEXT_BUDGET"
    E_CONTEXT_TRUNCATED = "E_CONTEXT_TRUNCATED"
    E_DEGENERATE_ANSWER = "E_DEGENERATE_ANSWER"
    E_CITATION_INVALID = "E_CITATION_INVALID"
    E_FILTER_INVALID = "E_FILTER_INVALID"
    E_QUERY_INVALID = "E_QUERY_INVALID"
    E_RESOURCE_GATE = "E_RESOURCE_GATE"
    E_STORE_LOCKED = "E_STORE_LOCKED"
    E_MIGRATION_REQUIRED = "E_MIGRATION_REQUIRED"
    E_INTEGRITY = "E_INTEGRITY"
    E_TIMEOUT = "E_TIMEOUT"


@dataclass(frozen=True, slots=True)
class ErrorSpec:
    """Static properties of an error code."""

    category: ErrorCategory
    retryable: bool


_C = ErrorCategory
_ERROR_SPECS: Mapping[ErrorCode, ErrorSpec] = MappingProxyType(
    {
        ErrorCode.E_CONFIG_INVALID: ErrorSpec(_C.CONFIG, False),
        ErrorCode.E_SOURCE_NOT_FOUND: ErrorSpec(_C.SOURCE, False),
        ErrorCode.E_SOURCE_LOCATOR_CHANGED: ErrorSpec(_C.SOURCE, False),
        ErrorCode.E_MASS_DELETE_GUARD: ErrorSpec(_C.SOURCE, False),
        ErrorCode.E_PARSE_FAILED: ErrorSpec(_C.PARSE, False),
        ErrorCode.E_UNSUPPORTED_FORMAT: ErrorSpec(_C.PARSE, False),
        ErrorCode.E_EMBED_DIM_MISMATCH: ErrorSpec(_C.PROVIDER, False),
        ErrorCode.E_EMBED_MODEL_MISMATCH: ErrorSpec(_C.INTEGRITY, False),
        ErrorCode.E_PROVIDER_TIMEOUT: ErrorSpec(_C.PROVIDER, True),
        ErrorCode.E_PROVIDER_UNAVAILABLE: ErrorSpec(_C.PROVIDER, True),
        ErrorCode.E_RATE_LIMITED: ErrorSpec(_C.PROVIDER, True),
        ErrorCode.E_CONTEXT_BUDGET: ErrorSpec(_C.RESOURCE, False),
        ErrorCode.E_CONTEXT_TRUNCATED: ErrorSpec(_C.PROVIDER, False),
        ErrorCode.E_DEGENERATE_ANSWER: ErrorSpec(_C.PROVIDER, False),
        ErrorCode.E_CITATION_INVALID: ErrorSpec(_C.PROVIDER, False),
        ErrorCode.E_FILTER_INVALID: ErrorSpec(_C.USER, False),
        ErrorCode.E_QUERY_INVALID: ErrorSpec(_C.USER, False),
        ErrorCode.E_RESOURCE_GATE: ErrorSpec(_C.RESOURCE, False),
        ErrorCode.E_STORE_LOCKED: ErrorSpec(_C.STORAGE, True),
        ErrorCode.E_MIGRATION_REQUIRED: ErrorSpec(_C.STORAGE, False),
        ErrorCode.E_INTEGRITY: ErrorSpec(_C.INTEGRITY, False),
        ErrorCode.E_TIMEOUT: ErrorSpec(_C.RESOURCE, True),
    }
)


def error_spec(code: ErrorCode) -> ErrorSpec:
    """Return the static category/retryable properties of ``code``."""
    return _ERROR_SPECS[code]


class RagError(Exception):
    """Base class of every engine error."""

    code: ErrorCode
    category: ErrorCategory
    retryable: bool
    message: str
    context: Mapping[str, ContextValue]

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool | None = None,
        context: Mapping[str, ContextValue] | None = None,
    ) -> None:
        super().__init__(f"{code.value}: {message}")
        spec = _ERROR_SPECS[code]
        self.code = code
        self.category = spec.category
        self.retryable = spec.retryable if retryable is None else retryable
        self.message = message
        self.context = _freeze_mapping(context or {})

    def __reduce__(self) -> tuple[object, ...]:
        """Support ``copy``, ``deepcopy`` and ``pickle`` (the context is rebuilt frozen)."""
        return (
            _restore_error,
            (type(self), self.code.value, self.message, self.retryable, _thaw(self.context)),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain structure suitable for logs and API problems."""
        return {
            "code": self.code.value,
            "category": self.category.value,
            "retryable": self.retryable,
            "message": self.message,
            "context": _thaw(self.context),
        }


def _restore_error(
    cls: type[RagError],
    code: str,
    message: str,
    retryable: bool,
    context: Mapping[str, ContextValue],
) -> RagError:
    return cls(ErrorCode(code), message, retryable=retryable, context=context)


def violation(message: str, **context: ContextValue) -> RagError:
    """Build the error used for violated domain invariants."""
    return RagError(ErrorCode.E_INTEGRITY, message, context=context)


def require(condition: bool, message: str, **context: ContextValue) -> None:
    """Raise :class:`RagError` (``E_INTEGRITY``) unless ``condition`` holds."""
    if not condition:
        raise violation(message, **context)


def parse_failed(reason: ParseReason, message: str, **context: ContextValue) -> RagError:
    """``E_PARSE_FAILED`` with a machine-readable reason in its context."""
    return RagError(ErrorCode.E_PARSE_FAILED, message, context={"reason": reason.value, **context})
