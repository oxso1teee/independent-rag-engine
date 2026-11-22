"""Builds small DOCX files for tests, and the damaged variants the parser must refuse."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from datetime import datetime

from docx import Document
from docx.enum.style import WD_STYLE_TYPE

Item = tuple[str, str] | tuple[str, Sequence[Sequence[str]]]
"""``("h1", text)`` .. ``("h6", text)``, ``("title", text)``, ``("p", text)``, ``("li", text)``,
``("li2", text)`` (second level), ``("quote", text)``, ``("code", text)`` or
``("table", rows)``."""

OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def build_docx(
    items: Sequence[Item],
    *,
    title: str | None = None,
    author: str | None = None,
    created: datetime | None = None,
) -> bytes:
    document = Document()
    document.styles.add_style("Code", WD_STYLE_TYPE.PARAGRAPH)
    for kind, value in items:
        if kind == "table":
            rows = value
            assert not isinstance(rows, str)
            table = document.add_table(rows=len(rows), cols=max((len(r) for r in rows), default=1))
            for row, cells in zip(table.rows, rows, strict=False):
                for cell, text in zip(row.cells, cells, strict=False):
                    cell.text = text
            continue
        assert isinstance(value, str)
        if kind == "title":
            document.add_heading(value, 0)
        elif kind[0] == "h" and kind[1:].isdigit():
            document.add_heading(value, int(kind[1:]))
        elif kind == "li":
            document.add_paragraph(value, style="List Bullet")
        elif kind == "li2":
            document.add_paragraph(value, style="List Bullet 2")
        elif kind == "quote":
            document.add_paragraph(value, style="Quote")
        elif kind == "code":
            document.add_paragraph(value, style="Code")
        else:
            document.add_paragraph(value)
    properties = document.core_properties
    properties.title = title or ""
    properties.author = author or ""
    if created is not None:
        properties.created = created
    buffer = io.BytesIO()
    document.save(buffer)
    return _stable(buffer.getvalue())


def _stable(data: bytes) -> bytes:
    """The same archive with fixed entry timestamps, so equal input gives equal bytes."""
    out = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(data)) as source,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for info in source.infolist():
            entry = zipfile.ZipInfo(info.filename, date_time=(2020, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(entry, source.read(info.filename))
    return out.getvalue()


def zip_with(entries: dict[str, bytes], *, compress: bool = True) -> bytes:
    buffer = io.BytesIO()
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", method) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def docx_with_part(data: bytes, name: str, content: bytes) -> bytes:
    """``data`` with one more entry (for example a huge or a very compressible part)."""
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        entries = {n: source.read(n) for n in source.namelist()}
    entries[name] = content
    return zip_with(entries)


def encrypted_docx_stub() -> bytes:
    """What Word writes for a password-protected file: an OLE container, not a ZIP archive."""
    return OLE_MAGIC + b"\x00" * 512
