"""From a stored source entry to a parsed document.

``DocumentProcessor`` reads the bytes within a size limit, identifies the format from the
content, picks the parser that supports it and returns the parsed document together with the
provenance that is recorded with a version (parser, normaliser, MIME decision, content hash).
It never persists anything and never lets a parser's own exception escape: whatever goes wrong
is an ``E_PARSE_FAILED`` or ``E_UNSUPPORTED_FORMAT`` with a stable reason.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import BinaryIO

from rag_engine.domain import (
    ContentHash,
    ErrorCode,
    ParsedDocument,
    ParseReason,
    RagError,
    SourceKey,
    parse_failed,
)
from rag_engine.ingestion.mime import MimeResult, detect_mime, require_supported
from rag_engine.observability import Timer, get_logger, stage_scope
from rag_engine.ports import Clock, ParseHints, ParseLimits, Parser

_log = get_logger("rag_engine.ingestion.processing")
_READ_BLOCK = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ParsedResult:
    document: ParsedDocument
    mime: MimeResult
    parser_id: str
    parser_version: str
    normalizer_version: str
    content_hash: ContentHash
    size_bytes: int
    parse_ms: int


def read_limited(stream: BinaryIO, max_bytes: int) -> bytes:
    """All bytes of ``stream``; more than ``max_bytes`` is an error, found without reading on."""
    chunks: list[bytes] = []
    total = 0
    while block := stream.read(min(_READ_BLOCK, max_bytes + 1 - total)):
        total += len(block)
        if total > max_bytes:
            raise parse_failed(
                ParseReason.TOO_LARGE, "the document is larger than allowed", limit=max_bytes
            )
        chunks.append(block)
    return b"".join(chunks)


class DocumentProcessor:
    def __init__(self, parsers: Sequence[Parser], clock: Clock) -> None:
        self._parsers = tuple(parsers)
        self._clock = clock

    def read(self, stream: BinaryIO, limits: ParseLimits) -> bytes:
        return read_limited(stream, limits.max_input_bytes)

    def parse(self, data: bytes, key: SourceKey, limits: ParseLimits) -> ParsedResult:
        mime = require_supported(detect_mime(io.BytesIO(data), key.value), key.value)
        parser = self._select(mime, key)
        extension = key.value.rsplit(".", 1)[-1].lower() if "." in key.value else None
        hints = ParseHints(
            mime=mime.mime,
            extension=extension,
            source_key=key,
            encoding=mime.encoding,
            limits=limits,
        )
        timer = Timer(self._clock)
        with stage_scope(_log, "process.parse", clock=self._clock, parser=parser.parser_id):
            document = self._run(parser, data, hints)
        return ParsedResult(
            document=document,
            mime=mime,
            parser_id=parser.parser_id,
            parser_version=parser.version,
            normalizer_version=parser.normalizer_version,
            content_hash=ContentHash.of_bytes(data),
            size_bytes=len(data),
            parse_ms=timer.elapsed_ms(),
        )

    def _select(self, mime: MimeResult, key: SourceKey) -> Parser:
        extension = key.value.rsplit(".", 1)[-1].lower() if "." in key.value else None
        for parser in self._parsers:
            if parser.supports(mime.mime, extension):
                return parser
        raise RagError(
            ErrorCode.E_UNSUPPORTED_FORMAT,
            "no parser is available for this format (an optional extra may be missing)",
            context={"key": key.value, "mime": mime.mime},
        )

    @staticmethod
    def _run(parser: Parser, data: bytes, hints: ParseHints) -> ParsedDocument:
        try:
            return parser.parse(data, hints)
        except RagError:
            raise
        except MemoryError:
            raise parse_failed(ParseReason.MEMORY, "the parser ran out of memory") from None
        except RecursionError:
            raise parse_failed(ParseReason.TOO_DEEP, "the document is nested too deeply") from None
        except Exception as exc:
            raise parse_failed(
                ParseReason.CORRUPT,
                "the document could not be parsed",
                parser=parser.parser_id,
                error_type=type(exc).__name__,
            ) from exc
