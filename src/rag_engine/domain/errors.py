"""Central error taxonomy.

Every failure surfaced by the engine is a :class:`RagError` carrying a stable
machine-readable code, a category, a retryable flag, a human-readable message
and a small scalar context. Codes are part of the public contract: they are
never renamed or reused.

Error ``context`` must never contain document text, prompts or secrets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

type ContextValue = str | int | float | bool | None


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
        self.context = MappingProxyType(dict(context or {}))

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain structure suitable for logs and API problems."""
        return {
            "code": self.code.value,
            "category": self.category.value,
            "retryable": self.retryable,
            "message": self.message,
            "context": dict(self.context),
        }


def violation(message: str, **context: ContextValue) -> RagError:
    """Build the error used for violated domain invariants."""
    return RagError(ErrorCode.E_INTEGRITY, message, context=context)


def require(condition: bool, message: str, **context: ContextValue) -> None:
    """Raise :class:`RagError` (``E_INTEGRITY``) unless ``condition`` holds."""
    if not condition:
        raise violation(message, **context)
