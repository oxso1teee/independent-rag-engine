"""Content-based MIME detection.

The extension is never trusted on its own. The decision is taken in this order:

1. signatures (``filetype`` plus the PDF header and the ZIP central directory for DOCX);
2. whether the bytes are text, and in which encoding (``charset-normalizer``);
3. only then the extension, to tell Markdown and HTML from plain text.

The result (``mime``, ``detected_by``, ``encoding``) is recorded with the document. Nothing
here parses a document; only the first few kilobytes (and the central directory of ZIP
containers) are read. Short non-UTF-8 files are ambiguous by nature: their single-byte
encoding is a best guess and a later parser may have to decode with replacement.
"""

from __future__ import annotations

import codecs
import zipfile
from dataclasses import dataclass
from typing import BinaryIO

import filetype
from charset_normalizer import from_bytes

from rag_engine.domain import ErrorCode, RagError

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF = "application/pdf"
HTML = "text/html"
MARKDOWN = "text/markdown"
PLAIN = "text/plain"
OCTET = "application/octet-stream"
ZIP = "application/zip"

SUPPORTED_MIME_TYPES = frozenset({PLAIN, MARKDOWN, HTML, PDF, DOCX})

_HEAD_BYTES = 8192
_MARKDOWN_EXTENSIONS = frozenset({"md", "markdown"})
_HTML_EXTENSIONS = frozenset({"html", "htm", "xhtml"})
_HTML_MARKERS = ("<!doctype html", "<html", "<head", "<body")

# Encodings a text document may plausibly use here. Anything else (including the many
# exotic guesses the detector can make for short or noisy input) is not accepted as text.
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


@dataclass(frozen=True, slots=True)
class MimeResult:
    mime: str
    detected_by: str  # "magic", "content", "charset" or "extension"
    supported: bool
    encoding: str | None = None  # of text; ``None`` for binary formats


def _extension(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base else ""


def _zip_kind(stream: BinaryIO) -> str:
    try:
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            names = set(archive.namelist())
    except (zipfile.BadZipFile, OSError, ValueError):
        return ZIP
    if "[Content_Types].xml" in names and "word/document.xml" in names:
        return DOCX
    return ZIP


def _decode_utf8(head: bytes, *, complete: bool) -> str | None:
    """Strict UTF-8; a character cut by the end of the sampled head is tolerated."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        return decoder.decode(head, final=complete)
    except UnicodeDecodeError:
        return None


def _detect_encoding(head: bytes, *, complete: bool) -> tuple[str, str] | None:
    """``(encoding, how)`` for text, ``None`` when the bytes are not text."""
    has_bom = head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE, codecs.BOM_UTF32_LE))
    if b"\x00" in head and not has_bom:
        return None
    if head.startswith(codecs.BOM_UTF8):
        head = head[len(codecs.BOM_UTF8) :]
        return ("utf_8", "content") if _decode_utf8(head, complete=complete) is not None else None
    if not has_bom and _decode_utf8(head, complete=complete) is not None:
        return "utf_8", "content"
    best = from_bytes(head, cp_isolation=_TEXT_ENCODINGS).best()
    if best is None:
        return None
    if best.encoding in ("utf_16", "utf_32") and not best.bom:
        return None  # wide encodings are only trusted when announced by a byte-order mark
    return best.encoding, "charset"


def detect_mime(stream: BinaryIO, name: str) -> MimeResult:
    """Classify a seekable binary stream; ``name`` is only a tie-breaker for text."""
    head = stream.read(_HEAD_BYTES)
    complete = len(head) < _HEAD_BYTES
    if b"%PDF-" in head[:1024]:  # PDF readers accept the header anywhere in the first kilobyte
        return MimeResult(PDF, "magic", True)
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        kind = _zip_kind(stream)
        return MimeResult(kind, "magic", kind in SUPPORTED_MIME_TYPES)
    known = filetype.guess(head)
    if known is not None:
        return MimeResult(known.mime, "magic", known.mime in SUPPORTED_MIME_TYPES)

    found = _detect_encoding(head, complete=complete)
    if found is None:
        return MimeResult(OCTET, "content", False)
    encoding, how = found
    text = head.decode(encoding, errors="replace").lower()
    if any(marker in text[:1024] for marker in _HTML_MARKERS):
        return MimeResult(HTML, "content", True, encoding)
    extension = _extension(name)
    if extension in _MARKDOWN_EXTENSIONS:
        return MimeResult(MARKDOWN, "extension", True, encoding)
    if extension in _HTML_EXTENSIONS:
        return MimeResult(HTML, "extension", True, encoding)
    return MimeResult(PLAIN, how, True, encoding)


def require_supported(result: MimeResult, key: str) -> MimeResult:
    """Return ``result`` or raise ``E_UNSUPPORTED_FORMAT`` with a precise context."""
    if not result.supported:
        raise RagError(
            ErrorCode.E_UNSUPPORTED_FORMAT,
            "the file format is not supported",
            context={"key": key, "mime": result.mime, "detected_by": result.detected_by},
        )
    return result
