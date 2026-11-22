"""One normalisation for every document: the text all offsets refer to.

``normalize_text`` is pure and idempotent. Prose and code differ only in how whitespace is
treated: prose collapses it, code keeps indentation and inner spacing.
"""

from __future__ import annotations

import html
import unicodedata

NORMALIZER_BASE = "norm/1"

_LINE_BREAKS = str.maketrans(
    {"\r": "\n", "\x0b": "\n", "\x0c": "\n", "\x85": "\n", " ": "\n", " ": "\n"}
)
# Zero-width space, word joiner, byte-order mark and soft hyphen carry no content. The
# zero-width joiners (U+200C, U+200D) are kept: they change how some scripts are written.
_REMOVED = dict.fromkeys(map(ord, "​⁠﻿­"))
_MAX_PASSES = 4


def _is_removable_control(char: str) -> bool:
    return unicodedata.category(char) == "Cc" and char not in "\n\t"


def _clean_characters(text: str) -> str:
    text = text.replace("\r\n", "\n").translate(_LINE_BREAKS).translate(_REMOVED)
    if text.isascii() and text.isprintable():
        return text
    return "".join(
        " " if unicodedata.category(c) == "Zs" else c for c in text if not _is_removable_control(c)
    )


def _prose(text: str) -> str:
    out: list[str] = []
    lines = [" ".join(line.split()) for line in text.split("\n")]
    blank = 0
    for line in lines:
        if line:
            blank = 0
            out.append(line)
        else:
            blank += 1
            if blank <= 1 and out:
                out.append("")
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def _code(text: str) -> str:
    lines = [line.replace("\t", "    ").rstrip(" ") for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def normalize_text(text: str, *, code: bool = False) -> str:
    """The normalised form of ``text``; applying it to its own result changes nothing."""
    shape = _code if code else _prose
    current = text
    for _ in range(_MAX_PASSES):
        following = shape(unicodedata.normalize("NFC", _clean_characters(current)))
        if following == current:
            break
        current = following
    return current


def unescape_entities_once(text: str) -> str:
    """Decode HTML entities exactly one level: ``&amp;lt;`` becomes ``&lt;``, not ``<``."""
    return html.unescape(text)
