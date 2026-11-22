"""The messages between the process that parses and the isolated worker that does the parsing.

Everything is length-prefixed JSON: no pickle, so a compromised worker cannot attack the parent
through deserialisation. Documents are rebuilt through the domain constructors, which check
every invariant again.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict
from datetime import datetime
from typing import Any, BinaryIO

from rag_engine.domain import (
    Block,
    BlockKind,
    ErrorCode,
    Language,
    ParsedDocument,
    RagError,
    SourceKey,
)
from rag_engine.ports import ParseHints, ParseLimits

MAX_FRAME = 1 << 30  # a frame larger than this is a protocol error, not a document
_HEADER = struct.Struct(">Q")


def dumps(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8", "surrogatepass"
    )


def loads(data: bytes) -> Any:
    return json.loads(data.decode("utf-8", "surrogatepass"))


def frame(payload: bytes) -> bytes:
    return _HEADER.pack(len(payload)) + payload


def read_exactly(stream: BinaryIO, size: int) -> bytes:
    """``size`` bytes from ``stream``; ``EOFError`` when it ends first."""
    parts: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, 1 << 20))
        if not chunk:
            raise EOFError
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def read_frame(stream: BinaryIO) -> bytes:
    (size,) = _HEADER.unpack(read_exactly(stream, _HEADER.size))
    if size > MAX_FRAME:
        raise ValueError("frame too large")
    return read_exactly(stream, size)


# ------------------------------------------------------------------------ documents


def encode_document(document: ParsedDocument) -> dict[str, Any]:
    """Blocks are stored as offsets only; their text is a slice of the document text."""
    return {
        "text": document.normalized_text,
        "blocks": [
            [
                block.kind.value,
                block.char_start,
                block.char_end,
                block.level,
                block.page,
                list(block.heading_path),
                dict(block.attrs),
            ]
            for block in document.blocks
        ],
        "title": document.title,
        "author": document.author,
        "created": document.created.isoformat() if document.created else None,
        "language": document.language.value,
        "warnings": list(document.warnings),
    }


def decode_document(data: dict[str, Any]) -> ParsedDocument:
    text: str = data["text"]
    blocks = tuple(
        Block(
            kind=BlockKind(kind),
            text=text[start:end],
            char_start=start,
            char_end=end,
            level=level,
            page=page,
            heading_path=tuple(path),
            attrs=attrs,
        )
        for kind, start, end, level, page, path, attrs in data["blocks"]
    )
    created = data["created"]
    return ParsedDocument(
        normalized_text=text,
        blocks=blocks,
        title=data["title"],
        author=data["author"],
        created=datetime.fromisoformat(created) if created else None,
        language=Language(data["language"]),
        warnings=tuple(data["warnings"]),
    )


# ------------------------------------------------------------------------- requests


def encode_hints(hints: ParseHints) -> dict[str, Any]:
    return {
        "mime": hints.mime,
        "extension": hints.extension,
        "source_key": hints.source_key.value if hints.source_key else None,
        "encoding": hints.encoding,
        "limits": asdict(hints.limits),
    }


def decode_hints(data: dict[str, Any]) -> ParseHints:
    key = data["source_key"]
    return ParseHints(
        mime=data["mime"],
        extension=data["extension"],
        source_key=SourceKey(key) if key else None,
        encoding=data["encoding"],
        limits=ParseLimits(**data["limits"]),
    )


# --------------------------------------------------------------------------- errors


def encode_error(error: RagError) -> dict[str, Any]:
    return {"code": error.code.value, "message": error.message, "context": dict(error.context)}


def decode_error(data: dict[str, Any]) -> RagError:
    return RagError(ErrorCode(data["code"]), data["message"], context=data["context"])
