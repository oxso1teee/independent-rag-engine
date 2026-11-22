"""Stable process exit codes of the ``rag`` command.

0 success; 1 unexpected failure; 2 usage error (reported by the argument parser). Every
other code maps one error code of the taxonomy, so scripts can react to a specific problem.
"""

from __future__ import annotations

from enum import IntEnum

from rag_engine.domain import ErrorCategory, ErrorCode, RagError


class ExitCode(IntEnum):
    OK = 0
    FAILURE = 1
    USAGE = 2
    CONFIG = 3
    SOURCE_NOT_FOUND = 4
    LOCATOR_CHANGED = 5
    MASS_DELETE_GUARD = 6
    STORE_LOCKED = 7
    MIGRATION_REQUIRED = 8
    INTEGRITY = 9
    RESOURCE = 10
    UNSUPPORTED = 11


_BY_CODE: dict[ErrorCode, ExitCode] = {
    ErrorCode.E_CONFIG_INVALID: ExitCode.CONFIG,
    ErrorCode.E_SOURCE_NOT_FOUND: ExitCode.SOURCE_NOT_FOUND,
    ErrorCode.E_SOURCE_LOCATOR_CHANGED: ExitCode.LOCATOR_CHANGED,
    ErrorCode.E_MASS_DELETE_GUARD: ExitCode.MASS_DELETE_GUARD,
    ErrorCode.E_STORE_LOCKED: ExitCode.STORE_LOCKED,
    ErrorCode.E_MIGRATION_REQUIRED: ExitCode.MIGRATION_REQUIRED,
    ErrorCode.E_UNSUPPORTED_FORMAT: ExitCode.UNSUPPORTED,
}

_BY_CATEGORY: dict[ErrorCategory, ExitCode] = {
    ErrorCategory.CONFIG: ExitCode.CONFIG,
    ErrorCategory.INTEGRITY: ExitCode.INTEGRITY,
    ErrorCategory.RESOURCE: ExitCode.RESOURCE,
    ErrorCategory.USER: ExitCode.USAGE,
}


def exit_code_for(error: RagError) -> ExitCode:
    return _BY_CODE.get(error.code) or _BY_CATEGORY.get(error.category) or ExitCode.FAILURE
