"""Stable process exit codes of the ``rag`` command, grouped by error category.

0 success; 1 unexpected failure; 2 usage or user error. Codes 3-9 each stand for one error
*category* of the taxonomy (ARCHITECTURE §7). `rag ask` adds 10 (insufficient evidence) and
11 (unverified answer): outcomes, not failures, that a script must tell apart from an answer.
`rag eval gate` exits 12 when a metric regressed.
The detailed error code stays available in the text output and in the ``--json`` document.
"""

from __future__ import annotations

from enum import IntEnum

from rag_engine.domain import ErrorCategory, ErrorCode, RagError, error_spec


class ExitCode(IntEnum):
    OK = 0
    FAILURE = 1  # internal or unexpected
    USAGE = 2  # user: bad arguments or input
    CONFIG = 3
    SOURCE = 4
    PARSE = 5
    PROVIDER = 6
    RESOURCE = 7
    STORAGE = 8
    INTEGRITY = 9
    INSUFFICIENT = 10  # `rag ask`: a valid outcome, the documents do not contain the answer
    UNVERIFIED = 11  # `rag ask`: the model's citations did not hold up
    REGRESSION = 12  # `rag eval gate`: a metric is worse than its baseline


_BY_CATEGORY: dict[ErrorCategory, ExitCode] = {
    ErrorCategory.INTERNAL: ExitCode.FAILURE,
    ErrorCategory.USER: ExitCode.USAGE,
    ErrorCategory.CONFIG: ExitCode.CONFIG,
    ErrorCategory.SOURCE: ExitCode.SOURCE,
    ErrorCategory.PARSE: ExitCode.PARSE,
    ErrorCategory.PROVIDER: ExitCode.PROVIDER,
    ErrorCategory.RESOURCE: ExitCode.RESOURCE,
    ErrorCategory.STORAGE: ExitCode.STORAGE,
    ErrorCategory.INTEGRITY: ExitCode.INTEGRITY,
}


def exit_code_for(error: RagError) -> ExitCode:
    return _BY_CATEGORY[error.category]


def exit_code_for_error_code(code: ErrorCode) -> ExitCode:
    return _BY_CATEGORY[error_spec(code).category]
