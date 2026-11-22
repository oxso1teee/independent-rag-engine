"""Content-based MIME detection.

The extension is never trusted on its own: signatures and the content decide. The
extension only separates the text flavours that cannot be told apart by content
(Markdown versus plain text). Nothing here parses a document; only the first few
kilobytes (and the central directory of ZIP containers) are read.
"""

from __future__ import annotations

import codecs
import zipfile
from dataclasses import dataclass
from typing import BinaryIO

from rag_engine.domain import ErrorCode, RagError

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF = "application/pdf"
HTML = "text/html"
MARKDOWN = "text/markdown"
PLAIN = "text/plain"
OCTET = "application/octet-stream"

SUPPORTED_MIME_TYPES = frozenset({PLAIN, MARKDOWN, HTML, PDF, DOCX})

_HEAD_BYTES = 8192
_MARKDOWN_EXTENSIONS = frozenset({"md", "markdown"})
_HTML_EXTENSIONS = frozenset({"html", "htm", "xhtml"})
_HTML_MARKERS = (b"<!doctype html", b"<html", b"<head", b"<body")

# Signature -> media type, for formats that are recognised only to be rejected precisely.
_BINARY_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"\x1f\x8b", "application/gzip"),
    (b"\x7fELF", "application/x-executable"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "application/x-ole-storage"),  # legacy .doc/.xls
    (b"Rar!\x1a\x07", "application/vnd.rar"),
    (b"7z\xbc\xaf\x27\x1c", "application/x-7z-compressed"),
)


@dataclass(frozen=True, slots=True)
class MimeResult:
    mime: str
    detected_by: str  # "magic", "content" or "extension"
    supported: bool


def _extension(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base else ""


def _is_utf8_text(head: bytes, *, complete: bool) -> bool:
    if b"\x00" in head:
        return False
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        decoder.decode(head, final=complete)
    except UnicodeDecodeError:
        return False
    return True


def _zip_kind(stream: BinaryIO) -> str:
    try:
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            names = set(archive.namelist())
    except (zipfile.BadZipFile, OSError, ValueError):
        return "application/zip"
    if "[Content_Types].xml" in names and "word/document.xml" in names:
        return DOCX
    return "application/zip"


def _has_pdf_header(head: bytes) -> bool:
    """PDF readers accept the header anywhere in the first kilobyte."""
    return b"%PDF-" in head[:1024]


def detect_mime(stream: BinaryIO, name: str) -> MimeResult:
    """Classify a seekable binary stream; ``name`` is only a tie-breaker for text."""
    head = stream.read(_HEAD_BYTES)
    complete = len(head) < _HEAD_BYTES
    if _has_pdf_header(head):
        return MimeResult(PDF, "magic", True)
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        kind = _zip_kind(stream)
        return MimeResult(kind, "magic", kind in SUPPORTED_MIME_TYPES)
    for signature, media_type in _BINARY_SIGNATURES:
        if head.startswith(signature):
            return MimeResult(media_type, "magic", False)

    text_head = head
    if head.startswith(codecs.BOM_UTF8):
        text_head = head[len(codecs.BOM_UTF8) :]
    elif head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        # UTF-16 text is recognised but has no parser in the supported set.
        return MimeResult(OCTET, "magic", False)
    if not _is_utf8_text(text_head, complete=complete):
        return MimeResult(OCTET, "content", False)

    lowered = text_head[:1024].lower()
    if any(marker in lowered for marker in _HTML_MARKERS):
        return MimeResult(HTML, "content", True)
    extension = _extension(name)
    if extension in _MARKDOWN_EXTENSIONS:
        return MimeResult(MARKDOWN, "extension", True)
    if extension in _HTML_EXTENSIONS:
        return MimeResult(HTML, "extension", True)
    return MimeResult(PLAIN, "content", True)


def require_supported(result: MimeResult, key: str) -> MimeResult:
    """Return ``result`` or raise ``E_UNSUPPORTED_FORMAT`` with a precise context."""
    if not result.supported:
        raise RagError(
            ErrorCode.E_UNSUPPORTED_FORMAT,
            "the file format is not supported",
            context={"key": key, "mime": result.mime, "detected_by": result.detected_by},
        )
    return result
