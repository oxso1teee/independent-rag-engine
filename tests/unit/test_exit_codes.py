"""Exit codes follow the error category (ARCHITECTURE §7); details stay in the error code."""

from __future__ import annotations

import pytest

from rag_engine.cli.exit_codes import ExitCode, exit_code_for
from rag_engine.domain import ErrorCategory, ErrorCode, RagError, error_spec

EXPECTED = {
    ErrorCategory.INTERNAL: 1,
    ErrorCategory.USER: 2,
    ErrorCategory.CONFIG: 3,
    ErrorCategory.SOURCE: 4,
    ErrorCategory.PARSE: 5,
    ErrorCategory.PROVIDER: 6,
    ErrorCategory.RESOURCE: 7,
    ErrorCategory.STORAGE: 8,
    ErrorCategory.INTEGRITY: 9,
}


def test_every_category_has_one_stable_code() -> None:
    assert set(EXPECTED) == set(ErrorCategory)
    assert {c: int(exit_code_for_category(c)) for c in ErrorCategory} == EXPECTED
    assert len(set(EXPECTED.values())) == len(EXPECTED)  # no two categories share a code
    assert int(ExitCode.OK) == 0


def exit_code_for_category(category: ErrorCategory) -> ExitCode:
    # any code of that category maps to it; INTERNAL has no error code, so it is checked directly
    if category is ErrorCategory.INTERNAL:
        return ExitCode.FAILURE
    code = next(c for c in ErrorCode if error_spec(c).category is category)
    return exit_code_for(RagError(code, "x"))


@pytest.mark.parametrize("code", list(ErrorCode))
def test_every_error_code_maps_to_the_code_of_its_category(code: ErrorCode) -> None:
    error = RagError(code, "x")
    assert int(exit_code_for(error)) == EXPECTED[error_spec(code).category]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (ErrorCode.E_MASS_DELETE_GUARD, 4),
        (ErrorCode.E_SOURCE_LOCATOR_CHANGED, 4),
        (ErrorCode.E_UNSUPPORTED_FORMAT, 5),
        (ErrorCode.E_PARSE_FAILED, 5),
        (ErrorCode.E_RATE_LIMITED, 6),
        (ErrorCode.E_RESOURCE_GATE, 7),
        (ErrorCode.E_STORE_LOCKED, 8),
        (ErrorCode.E_MIGRATION_REQUIRED, 8),
        (ErrorCode.E_INTEGRITY, 9),
        (ErrorCode.E_FILTER_INVALID, 2),
        (ErrorCode.E_QUERY_INVALID, 2),
        (ErrorCode.E_CONFIG_INVALID, 3),
    ],
)
def test_representative_codes(code: ErrorCode, expected: int) -> None:
    assert int(exit_code_for(RagError(code, "x"))) == expected
