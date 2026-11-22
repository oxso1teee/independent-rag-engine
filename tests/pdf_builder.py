"""Builds small valid PDF files for tests: text in lines, metadata, pages without text.

Text is written with an unembedded two-byte font and a ``ToUnicode`` map, which is enough for
text extraction and works for any character, Cyrillic included. No network and no fixtures on
disk are needed.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from datetime import datetime

from pypdf import PdfReader, PdfWriter

Page = Sequence[str] | None  # lines of text; ``None`` is a page that only draws a rectangle
_LEADING = 14


def _cmap(codes: dict[str, int]) -> bytes:
    entries = [
        f"<{code:04X}> <{char.encode('utf-16-be').hex().upper()}>" for char, code in codes.items()
    ]
    blocks = [entries[i : i + 100] for i in range(0, len(entries), 100)]
    body = "".join(
        f"{len(block)} beginbfchar\n" + "\n".join(block) + "\nendbfchar\n" for block in blocks
    )
    return (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        f"{body}endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    ).encode("ascii")


def _content(lines: Page, codes: dict[str, int]) -> bytes:
    if lines is None:
        return b"0.8 g 100 100 200 100 re f"
    parts = [f"BT /F1 12 Tf {_LEADING} TL 72 760 Td"]
    for line in lines:
        if line:
            parts.append("<" + "".join(f"{codes[c]:04X}" for c in line) + "> Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("ascii")


def _stream(body: bytes) -> bytes:
    return b"<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream"


def _pdf_date(moment: datetime) -> str:
    return "(D:" + moment.strftime("%Y%m%d%H%M%S") + "Z)"


def build_pdf(
    pages: Sequence[Page],
    *,
    title: str | None = None,
    author: str | None = None,
    created: datetime | None = None,
) -> bytes:
    chars = sorted({c for page in pages if page for line in page for c in line} | {" "})
    codes = {char: index + 1 for index, char in enumerate(chars)}
    count = len(pages)
    kids = " ".join(f"{8 + 2 * i} 0 R" for i in range(count))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
        b"<< /Type /Font /Subtype /Type0 /BaseFont /Test /Encoding /Identity-H "
        b"/DescendantFonts [4 0 R] /ToUnicode 6 0 R >>",
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Test "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        b"/FontDescriptor 5 0 R /DW 600 >>",
        b"<< /Type /FontDescriptor /FontName /Test /Flags 4 /FontBBox [0 0 1000 1000] "
        b"/ItalicAngle 0 /Ascent 900 /Descent -200 /CapHeight 700 /StemV 80 >>",
        _stream(_cmap(codes)),
    ]
    info = []
    if title is not None:
        info.append(f"/Title <FEFF{title.encode('utf-16-be').hex().upper()}>")
    if author is not None:
        info.append(f"/Author <FEFF{author.encode('utf-16-be').hex().upper()}>")
    if created is not None:
        info.append(f"/CreationDate {_pdf_date(created)}")
    objects.append(("<< " + " ".join(info) + " >>").encode("ascii"))
    for index, lines in enumerate(pages):
        content = 9 + 2 * index
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % content
        )
        objects.append(_stream(_content(lines, codes)))

    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 7 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def encrypt_pdf(data: bytes, password: str, *, owner_only: bool = False) -> bytes:
    """``data`` encrypted with RC4 (needs no extra library); ``owner_only`` leaves the user
    password empty, so the file can still be read."""
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(data)))
    if owner_only:
        writer.encrypt("", owner_password=password, algorithm="RC4-128")
    else:
        writer.encrypt(password, algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
