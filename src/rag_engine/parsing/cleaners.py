"""Per-source noise rules: a line that matches a cleaner entirely is removed.

Cleaners are regular expressions chosen by the user, so they are validated when a source is
registered and applied only to lines of bounded length; together this keeps a hostile or
careless pattern from stalling a scan.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_engine.domain import ErrorCode, RagError
from rag_engine.parsing.normalize import NORMALIZER_BASE

MAX_CLEANERS = 50
MAX_PATTERN_LENGTH = 300
MAX_LINE_LENGTH = 4_000  # longer lines are never tested

# A group that repeats something unbounded inside an unbounded repetition: (a+)+, (.*)*, (\w+\s*)*
_NESTED_REPETITION = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)[+*{]")


@dataclass(frozen=True, slots=True)
class CleanerSet:
    patterns: tuple[str, ...] = ()
    compiled: tuple[re.Pattern[str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.patterns)

    def apply(self, text: str) -> str:
        """``text`` without the lines that match a cleaner entirely."""
        if not self.compiled:
            return text
        kept = [
            line
            for line in text.split("\n")
            if len(line) > MAX_LINE_LENGTH or not any(p.fullmatch(line) for p in self.compiled)
        ]
        return "\n".join(kept)


def _invalid(index: int, reason: str) -> RagError:
    return RagError(
        ErrorCode.E_CONFIG_INVALID,
        f"invalid cleaner #{index + 1}: {reason}",
        context={"cleaner": index + 1, "reason": reason},
    )


def compile_cleaners(patterns: Sequence[str]) -> CleanerSet:
    """Validate and compile ``patterns`` (``E_CONFIG_INVALID`` for anything unsafe)."""
    if len(patterns) > MAX_CLEANERS:
        raise RagError(
            ErrorCode.E_CONFIG_INVALID,
            f"at most {MAX_CLEANERS} cleaners are allowed",
            context={"count": len(patterns)},
        )
    compiled: list[re.Pattern[str]] = []
    for index, pattern in enumerate(patterns):
        if not pattern.strip():
            raise _invalid(index, "empty pattern")
        if len(pattern) > MAX_PATTERN_LENGTH:
            raise _invalid(index, f"longer than {MAX_PATTERN_LENGTH} characters")
        if _NESTED_REPETITION.search(pattern):
            raise _invalid(index, "nested unbounded repetition")
        try:
            regex = re.compile(pattern)
        except re.error:
            raise _invalid(index, "not a valid regular expression") from None
        if regex.fullmatch("") or regex.fullmatch(" "):
            raise _invalid(index, "matches an empty or blank line")
        compiled.append(regex)
    return CleanerSet(tuple(patterns), tuple(compiled))


def normalizer_version(cleaners: CleanerSet | None = None) -> str:
    """Version of the normalisation, including the cleaners that change the text."""
    if not cleaners:
        return NORMALIZER_BASE
    digest = hashlib.sha256("\n".join(cleaners.patterns).encode("utf-8")).hexdigest()[:8]
    return f"{NORMALIZER_BASE}+{digest}"
