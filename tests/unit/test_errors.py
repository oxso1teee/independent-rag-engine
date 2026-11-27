from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Callable

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
    "E_QUERY_INVALID": ("user", False),
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


# ------------------------------------------------- copy / pickle / frozen context

NESTED: dict[str, object] = {
    "stage": "embed",
    "attempts": [1, 2, 3],
    "limits": {"batch": 16, "tags": ["a", "b"], "inner": {"deep": [{"x": 1}]}},
    "flag": True,
    "none": None,
}


def _error() -> RagError:
    return RagError(
        ErrorCode.E_RATE_LIMITED,
        "slow down",
        retryable=False,  # overrides the code default (True) so restoration is observable
        context=NESTED,  # type: ignore[arg-type]
    )


def _assert_equivalent(original: RagError, restored: RagError) -> None:
    assert type(restored) is type(original)
    assert restored.code is original.code
    assert restored.category is original.category
    assert restored.message == original.message
    assert restored.retryable is original.retryable
    assert restored.context == original.context
    assert str(restored) == str(original)
    assert restored.to_dict() == original.to_dict()


@pytest.mark.parametrize(
    "clone",
    [
        copy.copy,
        copy.deepcopy,
        lambda e: pickle.loads(pickle.dumps(e)),  # noqa: S301
        lambda e: pickle.loads(pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL)),  # noqa: S301
    ],
    ids=["copy", "deepcopy", "pickle", "pickle-highest"],
)
def test_error_survives_cloning(clone: Callable[[RagError], RagError]) -> None:
    original = _error()
    restored = clone(original)
    assert restored is not original
    _assert_equivalent(original, restored)
    assert restored.retryable is False


def test_restored_error_keeps_its_context_frozen() -> None:
    restored = pickle.loads(pickle.dumps(_error()))  # noqa: S301
    with pytest.raises(TypeError):
        restored.context["new"] = 1
    with pytest.raises(TypeError):
        restored.context["limits"]["batch"] = 1
    with pytest.raises(TypeError):
        restored.context["attempts"][0] = 9
    with pytest.raises(AttributeError):
        restored.context["attempts"].append(4)


def test_nested_context_is_deeply_frozen_and_detached() -> None:
    source: dict[str, object] = {"items": [1, {"k": [2]}], "map": {"a": 1}}
    error = RagError(ErrorCode.E_TIMEOUT, "slow", context=source)  # type: ignore[arg-type]
    source["items"].append(3)  # type: ignore[attr-defined]
    source["map"]["a"] = 2  # type: ignore[index]
    assert error.context["items"] == (1, {"k": (2,)})
    assert error.context["map"] == {"a": 1}
    with pytest.raises(TypeError):
        error.context["map"]["a"] = 5  # type: ignore[index]


def test_to_dict_is_json_friendly_and_mutable_copy() -> None:
    data = _error().to_dict()
    assert json.loads(json.dumps(data))["context"]["limits"]["inner"] == {"deep": [{"x": 1}]}
    data["context"]["attempts"].append(99)  # type: ignore[index]
    assert _error().context["attempts"] == (1, 2, 3)


@pytest.mark.parametrize("bad", [{1: "x"}, {"s": {1, 2}}, {"b": b"raw"}, {"o": object()}])
def test_unsupported_context_values_are_rejected(bad: dict[object, object]) -> None:
    with pytest.raises(TypeError):
        RagError(ErrorCode.E_TIMEOUT, "x", context=bad)  # type: ignore[arg-type]


def test_every_code_round_trips() -> None:
    for code in ErrorCode:
        error = RagError(code, "m", context={"c": code.value})
        _assert_equivalent(error, pickle.loads(pickle.dumps(error)))  # noqa: S301


def test_subclasses_are_restored_as_their_own_type() -> None:
    class Specific(RagError):
        pass

    error = Specific(ErrorCode.E_PARSE_FAILED, "bad")
    restored = copy.deepcopy(error)
    assert isinstance(restored, Specific)


def test_parse_failures_carry_a_stable_reason() -> None:
    from rag_engine.domain import ParseReason, parse_failed

    assert {r.value for r in ParseReason} == {
        "corrupt",
        "encrypted",
        "no_text_layer",
        "too_large",
        "too_many_pages",
        "too_many_blocks",
        "too_deep",
        "zip_bomb",
        "timeout",
        "memory",
        "worker_failed",
        "encoding",
    }
    error = parse_failed(ParseReason.ENCRYPTED, "the document is encrypted", pages=3)
    assert error.code is ErrorCode.E_PARSE_FAILED
    assert error.category is ErrorCategory.PARSE
    assert error.retryable is False
    assert error.context == {"reason": "encrypted", "pages": 3}
