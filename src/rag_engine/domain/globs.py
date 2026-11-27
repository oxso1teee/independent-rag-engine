"""Glob patterns over source keys and document paths (``/``-separated, NFC, case-sensitive).

Supported: ``*`` (within a segment), ``?``, ``[...]`` / ``[!...]`` classes, ``**`` (any
number of segments, including none when written as ``**/``). A pattern without ``/``
matches the file name at any depth. Anything else is rejected up front instead of being
silently misinterpreted.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from rag_engine.domain.errors import ErrorCode, RagError


def _invalid(pattern: str, reason: str) -> RagError:
    return RagError(
        ErrorCode.E_CONFIG_INVALID,
        f"invalid glob pattern: {reason}",
        context={"pattern": pattern[:200]},
    )


def _translate(pattern: str) -> str:
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            if pattern.startswith("**", i):
                j = i + 2
                if pattern.startswith("/", j):
                    out.append("(?:.*/)?")  # ``**/`` also matches zero directories
                    i = j + 1
                else:
                    out.append(".*")
                    i = j
                continue
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            end = pattern.find("]", i + 2)
            if end == -1:
                raise _invalid(pattern, "unterminated character class")
            body = pattern[i + 1 : end]
            negate = body.startswith(("!", "^"))
            body = body[1:] if negate else body
            if not body or "/" in body:
                raise _invalid(pattern, "empty character class or '/' inside a class")
            out.append("[" + ("^" if negate else "") + re.escape(body).replace(r"\-", "-") + "]")
            i = end
        else:
            out.append(re.escape(char))
        i += 1
    return "".join(out)


def compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile one pattern; raises ``E_CONFIG_INVALID`` for anything malformed."""
    normalised = unicodedata.normalize("NFC", pattern)
    if not normalised.strip():
        raise _invalid(pattern, "the pattern is empty")
    if "\\" in normalised or "\x00" in normalised:
        raise _invalid(pattern, "use '/' as the separator; backslashes and NUL are not allowed")
    if normalised.startswith("/") or any(s == ".." for s in normalised.split("/")):
        raise _invalid(pattern, "patterns are relative to the source root")
    directory_only = normalised.endswith("/")
    core = normalised.rstrip("/")
    if not core:
        raise _invalid(pattern, "the pattern is empty")
    body = _translate(core + "/**" if directory_only else core)
    if "/" not in core:
        body = "(?:.*/)?" + body
    try:
        return re.compile(body, re.DOTALL)
    except re.error as exc:  # pragma: no cover - _translate escapes everything else
        raise _invalid(pattern, "the pattern cannot be compiled") from exc


class GlobSet:
    """Include/exclude rules compiled once."""

    def __init__(self, include: Sequence[str], exclude: Sequence[str]) -> None:
        self._include = tuple(compile_glob(p) for p in include)
        self._exclude = tuple(compile_glob(p) for p in exclude)

    def includes(self, key: str) -> bool:
        return any(p.fullmatch(key) for p in self._include)

    def excludes(self, key: str) -> bool:
        return any(p.fullmatch(key) for p in self._exclude)

    def prunes_directory(self, key: str) -> bool:
        """Whether nothing below directory ``key`` can be selected."""
        return any(p.fullmatch(key) or p.fullmatch(key + "/") for p in self._exclude)

    def selects(self, key: str) -> bool:
        return self.includes(key) and not self.excludes(key)
