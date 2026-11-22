"""P2.8: MIME detection table. Content decides; the extension only breaks ties."""

from __future__ import annotations

import io
import os
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
    ZIP,
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
RUSSIAN = "Привет, это достаточно длинный русский текст для определения кодировки. " * 4
FRENCH = "café au lait, un très long texte français avec des accents é è à ç ù ô. " * 4

# (case id, content, file name, mime, detected_by, supported, encoding)
TABLE = [
    ("plain", b"hello world\n", "a.txt", PLAIN, "content", True, "utf_8"),
    ("plain-no-extension", b"hello", "README", PLAIN, "content", True, "utf_8"),
    ("plain-unknown-extension", "привет".encode(), "notes.rst", PLAIN, "content", True, "utf_8"),
    ("utf8-bom", b"\xef\xbb\xbfhello", "a.txt", PLAIN, "content", True, "utf_8"),
    ("empty", b"", "empty.txt", PLAIN, "content", True, "utf_8"),
    ("markdown", b"# Title\n\ntext", "a.md", MARKDOWN, "extension", True, "utf_8"),
    ("markdown-long-ext", b"# Title", "a.markdown", MARKDOWN, "extension", True, "utf_8"),
    ("html-by-content", b"<!DOCTYPE html><html></html>", "p.txt", HTML, "content", True, "utf_8"),
    ("html-no-extension", b"<html><body>x</body></html>", "page", HTML, "content", True, "utf_8"),
    ("html-by-extension", b"plain words", "page.html", HTML, "extension", True, "utf_8"),
    ("pdf", b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", "doc.pdf", PDF, "magic", True, None),
    ("pdf-wrong-extension", b"%PDF-1.4\n...", "doc.txt", PDF, "magic", True, None),
    ("pdf-leading-junk", b"\n\n%PDF-1.4\n", "doc.pdf", PDF, "magic", True, None),
    ("docx", DOCX_BYTES, "report.docx", DOCX, "magic", True, None),
    ("docx-wrong-extension", DOCX_BYTES, "report.bin", DOCX, "magic", True, None),
    ("fake-pdf-is-text", b"just some words", "fake.pdf", PLAIN, "content", True, "utf_8"),
    ("fake-docx-is-text", b"just some words", "fake.docx", PLAIN, "content", True, "utf_8"),
    ("long-text", LONG_TEXT, "long.txt", PLAIN, "content", True, "utf_8"),
    ("cp1251", RUSSIAN.encode("cp1251"), "ru.txt", PLAIN, "charset", True, "cp1251"),
    ("cp1251-markdown", RUSSIAN.encode("cp1251"), "ru.md", MARKDOWN, "extension", True, "cp1251"),
    ("koi8-r", RUSSIAN.encode("koi8_r"), "ru.txt", PLAIN, "charset", True, "koi8_r"),
    ("cp866", RUSSIAN.encode("cp866"), "ru.txt", PLAIN, "charset", True, "cp866"),
    ("utf16-bom", RUSSIAN.encode("utf-16"), "ru.txt", PLAIN, "charset", True, "utf_16"),
    (
        "html-cp1251",
        ("<html><body>" + RUSSIAN).encode("cp1251"),
        "p",
        HTML,
        "content",
        True,
        "cp1251",
    ),
    ("zip", _zip("a.txt"), "archive.zip", ZIP, "magic", False, None),
    ("zip-named-docx", _zip("a.txt"), "report.docx", ZIP, "magic", False, None),
    ("broken-zip", b"PK\x03\x04garbage", "x.docx", ZIP, "magic", False, None),
    ("png", b"\x89PNG\r\n\x1a\n\x00\x00", "pic.png", "image/png", "magic", False, None),
    ("jpeg", b"\xff\xd8\xff\xe0\x00\x10JFIF", "pic.jpg", "image/jpeg", "magic", False, None),
    ("gif", b"GIF89a" + b"0" * 10, "pic.gif", "image/gif", "magic", False, None),
    ("gzip", b"\x1f\x8b\x08\x00" + b"0" * 20, "a.txt", "application/gzip", "magic", False, None),
    (
        "7z",
        b"7z\xbc\xaf\x27\x1c" + b"0" * 10,
        "a.7z",
        "application/x-7z-compressed",
        "magic",
        False,
        None,
    ),
    ("elf", b"\x7fELF\x02\x01" + b"\x00" * 20, "tool", OCTET, "content", False, None),
    (
        "legacy-doc",
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 30,
        "old.doc",
        OCTET,
        "content",
        False,
        None,
    ),
    ("binary-noise", b"\x00\x01\x02\xff\xfe", "data.txt", OCTET, "content", False, None),
    ("utf16-without-bom", "hi there".encode("utf-16-le"), "a.txt", OCTET, "content", False, None),
]


@pytest.mark.parametrize(
    ("content", "name", "mime", "detected_by", "supported", "encoding"),
    [pytest.param(*row[1:], id=row[0]) for row in TABLE],
)
def test_format_table(
    content: bytes, name: str, mime: str, detected_by: str, supported: bool, encoding: str | None
) -> None:
    result = detect_mime(io.BytesIO(content), name)
    assert result == MimeResult(mime, detected_by, supported, encoding)
    assert result.supported == (result.mime in SUPPORTED_MIME_TYPES)


def test_random_bytes_are_not_text() -> None:
    for _ in range(20):
        result = detect_mime(io.BytesIO(os.urandom(512)), "x.txt")
        assert not result.supported


def test_western_single_byte_text_is_accepted_with_a_best_guess_encoding() -> None:
    result = detect_mime(io.BytesIO(FRENCH.encode("latin-1")), "fr.txt")
    assert result.supported and result.mime == PLAIN and result.detected_by == "charset"
    assert result.encoding in {"latin_1", "cp1252", "cp1250", "iso8859_15", "iso8859_2"}


def test_a_character_cut_by_the_sampled_head_does_not_make_text_binary() -> None:
    text = ("я" * 4095 + "й" * 3000).encode("utf-8")  # the 8192-byte head ends inside a letter
    assert detect_mime(io.BytesIO(text), "a.txt").supported


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
