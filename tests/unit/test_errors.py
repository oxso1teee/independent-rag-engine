from __future__ import annotations

import pytest

from rag_engine.domain import ErrorCategory, ErrorCode, RagError, error_spec, require, violation

# Snapshot of the public contract: codes are never renamed, reused or re-categorised silently.
EXPECTED: dict[str, tuple[str, bool]] = {
    "E_CONFIG_INVALID": ("config", False),
    "E_SOURCE_NOT_FOUND": ("source", False),
    "E_SOURCE_LOCATOR_CHANGED": ("source", False),
    "E_MASS_DELETE_GUARD": ("source", False),
    "E_PARSE_FAILED": ("parse", False),
    "E_UNSUPPORTED_FORMAT": ("parse", False),
    "E_EMBED_DIM_MISMATCH": ("provider", False),
    "E_EMBED_MODEL_MISMATCH": ("integrity", False),
    "E_PROVIDER_TIMEOUT": ("provider", True),
    "E_PROVIDER_UNAVAILABLE": ("provider", True),
    "E_RATE_LIMITED": ("provider", True),
    "E_CONTEXT_BUDGET": ("resource", False),
    "E_CONTEXT_TRUNCATED": ("provider", False),
    "E_DEGENERATE_ANSWER": ("provider", False),
    "E_CITATION_INVALID": ("provider", False),
    "E_FILTER_INVALID": ("user", False),
    "E_RESOURCE_GATE": ("resource", False),
    "E_STORE_LOCKED": ("storage", True),
    "E_MIGRATION_REQUIRED": ("storage", False),
    "E_INTEGRITY": ("integrity", False),
    "E_TIMEOUT": ("resource", True),
}


def test_error_code_snapshot() -> None:
    actual = {
        code.value: (error_spec(code).category.value, error_spec(code).retryable)
        for code in ErrorCode
    }
    assert actual == EXPECTED


def test_code_values_equal_their_names() -> None:
    assert all(code.value == code.name for code in ErrorCode)


def test_every_category_is_a_known_category() -> None:
    assert {error_spec(c).category for c in ErrorCode} <= set(ErrorCategory)


def test_rag_error_carries_structured_fields() -> None:
    error = RagError(
        ErrorCode.E_PROVIDER_TIMEOUT, "embedder timed out", context={"attempt": 2, "model": "m"}
    )
    assert error.code is ErrorCode.E_PROVIDER_TIMEOUT
    assert error.category is ErrorCategory.PROVIDER
    assert error.retryable is True
    assert error.message == "embedder timed out"
    assert error.context == {"attempt": 2, "model": "m"}
    assert str(error) == "E_PROVIDER_TIMEOUT: embedder timed out"
    assert error.to_dict() == {
        "code": "E_PROVIDER_TIMEOUT",
        "category": "provider",
        "retryable": True,
        "message": "embedder timed out",
        "context": {"attempt": 2, "model": "m"},
    }


def test_retryable_can_be_overridden_per_instance() -> None:
    assert RagError(ErrorCode.E_PARSE_FAILED, "x", retryable=True).retryable is True
    assert RagError(ErrorCode.E_RATE_LIMITED, "x", retryable=False).retryable is False


def test_context_is_immutable_and_detached_from_the_input() -> None:
    source = {"a": 1}
    error = RagError(ErrorCode.E_TIMEOUT, "slow", context=source)
    source["a"] = 2
    assert error.context["a"] == 1
    with pytest.raises(TypeError):
        error.context["b"] = 3  # type: ignore[index]


def test_default_context_is_empty() -> None:
    assert RagError(ErrorCode.E_TIMEOUT, "slow").context == {}


def test_require_and_violation_use_the_integrity_code() -> None:
    require(True, "fine")
    with pytest.raises(RagError) as info:
        require(False, "broken", field="x")
    assert info.value.code is ErrorCode.E_INTEGRITY
    assert info.value.category is ErrorCategory.INTEGRITY
    assert info.value.context == {"field": "x"}
    assert violation("v").code is ErrorCode.E_INTEGRITY
