"""P2.8: MIME detection table. Content decides; the extension only breaks ties."""

from __future__ import annotations

import io
import zipfile

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.ingestion.mime import (
    DOCX,
    HTML,
    MARKDOWN,
    OCTET,
    PDF,
    PLAIN,
    SUPPORTED_MIME_TYPES,
    MimeResult,
    detect_mime,
    require_supported,
)


def _zip(*names: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, "<x/>")
    return buffer.getvalue()


DOCX_BYTES = _zip("[Content_Types].xml", "word/document.xml")
LONG_TEXT = ("слово " * 3000).encode("utf-8")  # longer than the sniffed head, cut mid-character
OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# (case id, content, file name, expected mime, expected detected_by, supported)
TABLE = [
    ("plain", b"hello world\n", "a.txt", PLAIN, "content", True),
    ("plain-no-extension", b"hello", "README", PLAIN, "content", True),
    ("plain-unknown-extension", "привет".encode(), "notes.rst", PLAIN, "content", True),
    ("utf8-bom", b"\xef\xbb\xbfhello", "a.txt", PLAIN, "content", True),
    ("empty", b"", "empty.txt", PLAIN, "content", True),
    ("markdown", b"# Title\n\ntext", "a.md", MARKDOWN, "extension", True),
    ("markdown-long-extension", b"# Title", "a.markdown", MARKDOWN, "extension", True),
    ("html-by-content", b"<!DOCTYPE html><html></html>", "page.txt", HTML, "content", True),
    ("html-without-extension", b"<html><body>x</body></html>", "page", HTML, "content", True),
    ("html-by-extension", b"plain words", "page.html", HTML, "extension", True),
    ("pdf", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", "doc.pdf", PDF, "magic", True),
    ("pdf-wrong-extension", b"%PDF-1.4\n...", "doc.txt", PDF, "magic", True),
    ("pdf-with-leading-junk", b"\n\n%PDF-1.4\n", "doc.pdf", PDF, "magic", True),
    ("docx", DOCX_BYTES, "report.docx", DOCX, "magic", True),
    ("docx-wrong-extension", DOCX_BYTES, "report.bin", DOCX, "magic", True),
    ("fake-pdf-is-text", b"just some words", "fake.pdf", PLAIN, "content", True),
    ("fake-docx-is-text", b"just some words", "fake.docx", PLAIN, "content", True),
    ("long-text", LONG_TEXT, "long.txt", PLAIN, "content", True),
    ("zip", _zip("a.txt"), "archive.zip", "application/zip", "magic", False),
    ("zip-named-docx", _zip("a.txt"), "report.docx", "application/zip", "magic", False),
    ("broken-zip", b"PK\x03\x04garbage", "x.docx", "application/zip", "magic", False),
    ("png", b"\x89PNG\r\n\x1a\n\x00\x00", "pic.png", "image/png", "magic", False),
    ("jpeg", b"\xff\xd8\xff\xe0\x00", "pic.jpg", "image/jpeg", "magic", False),
    ("gzip", b"\x1f\x8b\x08\x00", "a.txt", "application/gzip", "magic", False),
    ("elf", b"\x7fELF\x02\x01", "tool", "application/x-executable", "magic", False),
    ("legacy-doc", OLE, "old.doc", "application/x-ole-storage", "magic", False),
    ("binary-noise", b"\x00\x01\x02\xff\xfe", "data.txt", OCTET, "content", False),
    ("latin1-text", "caf\xe9".encode("latin-1"), "a.txt", OCTET, "content", False),
    ("utf16", "hi".encode("utf-16"), "a.txt", OCTET, "magic", False),
]


@pytest.mark.parametrize(
    ("content", "name", "mime", "detected_by", "supported"),
    [pytest.param(*row[1:], id=row[0]) for row in TABLE],
)
def test_format_table(
    content: bytes, name: str, mime: str, detected_by: str, supported: bool
) -> None:
    result = detect_mime(io.BytesIO(content), name)
    assert result == MimeResult(mime, detected_by, supported)
    assert result.supported == (result.mime in SUPPORTED_MIME_TYPES)


def test_unsupported_formats_raise_the_documented_error() -> None:
    result = detect_mime(io.BytesIO(b"\x89PNG\r\n\x1a\n"), "photo.png")
    with pytest.raises(RagError) as info:
        require_supported(result, "photos/photo.png")
    assert info.value.code is ErrorCode.E_UNSUPPORTED_FORMAT
    assert info.value.context["mime"] == "image/png"
    assert info.value.context["key"] == "photos/photo.png"


def test_supported_results_pass_through() -> None:
    result = detect_mime(io.BytesIO(b"text"), "a.txt")
    assert require_supported(result, "a.txt") is result


def test_only_the_head_of_a_large_stream_is_read() -> None:
    class Counting(io.BytesIO):
        read_bytes = 0

        def read(self, size: int | None = -1) -> bytes:
            data = super().read(size)
            self.read_bytes += len(data)
            return data

    stream = Counting(b"x" * 10_000_000)
    detect_mime(stream, "big.txt")
    assert stream.read_bytes <= 8192
