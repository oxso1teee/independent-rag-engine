"""Helpers shared by the parsers."""

from __future__ import annotations

import re

from charset_normalizer import from_bytes

from rag_engine.domain import ParseReason, parse_failed
from rag_engine.ports import ParseHints

_TEXT_ENCODINGS = [
    "ascii",
    "utf_8",
    "utf_16",
    "utf_32",
    "cp1251",
    "koi8_r",
    "cp866",
    "iso8859_5",
    "mac_cyrillic",
    "cp1252",
    "latin_1",
    "iso8859_15",
    "cp1250",
    "iso8859_2",
]
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


def check_input_size(data: bytes, hints: ParseHints) -> None:
    if len(data) > hints.limits.max_input_bytes:
        raise parse_failed(
            ParseReason.TOO_LARGE,
            "the document is larger than allowed",
            limit=hints.limits.max_input_bytes,
        )


def extension_of(name: str | None) -> str | None:
    if not name or "." not in name.rsplit("/", 1)[-1]:
        return None
    return name.rsplit(".", 1)[-1].lower()


def decode_text(data: bytes, hints: ParseHints) -> tuple[str, list[str]]:
    """Text of ``data`` and the warnings raised while decoding.

    The encoding detected earlier (``hints.encoding``) is used first; without it UTF-8 is tried,
    then the detector. Undecodable bytes are replaced and reported, never silently dropped.
    """
    warnings: list[str] = []
    if hints.encoding:
        try:
            return data.decode(hints.encoding), warnings
        except (UnicodeDecodeError, LookupError):
            warnings.append("decoded_with_replacement")
            try:
                return data.decode(hints.encoding, errors="replace"), warnings
            except LookupError:
                warnings.clear()
    try:
        return data.decode("utf-8"), warnings
    except UnicodeDecodeError:
        pass
    best = from_bytes(data[:65_536], cp_isolation=_TEXT_ENCODINGS).best()
    if best is None:
        raise parse_failed(ParseReason.ENCODING, "the text encoding could not be determined")
    return data.decode(best.encoding, errors="replace"), ["decoded_with_replacement"]


def split_paragraphs(text: str) -> list[str]:
    """Paragraphs of ``text`` (blank-line separated); line endings must already be ``\\n``."""
    return _PARAGRAPH_BREAK.split(text.replace("\r\n", "\n").replace("\r", "\n"))


def clean_metadata(value: str | None) -> str | None:
    """A single-line, normalised metadata value, or ``None`` when nothing is left."""
    if value is None:
        return None
    from rag_engine.parsing.normalize import normalize_text

    cleaned = " ".join(normalize_text(value).split())
    return cleaned or None
