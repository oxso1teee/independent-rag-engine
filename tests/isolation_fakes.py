"""Parsers that misbehave in the ways an isolated worker must survive.

The worker imports them by ``module:callable`` (``tests.isolation_fakes:Sleeper``), so they run
in the child process. Each one answers with a document made of the input bytes, except where its
name says what it does instead.
"""

from __future__ import annotations

import os
import sys
import time

from rag_engine.domain import BlockKind, ParsedDocument
from rag_engine.parsing.assemble import TextAssembler
from rag_engine.ports import ParseHints


class _Base:
    parser_id = "fake"
    version = "1"
    normalizer_version = "norm/1"

    def supports(self, mime: str | None, extension: str | None) -> bool:
        return True

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        assembler = TextAssembler(hints.limits)
        assembler.add(BlockKind.PARAGRAPH, data.decode("utf-8", "replace"), page=1)
        return assembler.parsed(title=f"pid:{os.getpid()}", warnings=("fake",))


class Echo(_Base):
    parser_id = "echo"


class Sleeper(_Base):
    parser_id = "sleeper"

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        time.sleep(120)
        return super().parse(data, hints)


class Crasher(_Base):
    parser_id = "crasher"

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        os._exit(7)


class Greedy(_Base):
    parser_id = "greedy"

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        blob = bytearray(2 * 1024 * 1024 * 1024)  # far above any limit the tests set
        return super().parse(data + bytes(blob[:1]), hints)


class Deep(_Base):
    parser_id = "deep"

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        raise RecursionError


class Loud(_Base):
    parser_id = "loud"

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        print("noise on stdout")
        sys.stdout.write("more noise\n")
        sys.stderr.write("noise on stderr\n")
        os.write(1, b"raw noise on descriptor 1\n")
        return super().parse(data, hints)


class Environment(_Base):
    parser_id = "environment"

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        assembler = TextAssembler(hints.limits)
        assembler.add(BlockKind.PARAGRAPH, " ".join(sorted(os.environ)) or "none")
        return assembler.parsed(title=os.getcwd())


class Buggy(_Base):
    parser_id = "buggy"

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        raise ValueError("the private text of the document")


class Rich(_Base):
    """A document with every field set, to check that nothing is lost on the way."""

    parser_id = "rich"

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        from datetime import UTC, datetime

        assembler = TextAssembler(hints.limits)
        assembler.add(BlockKind.HEADING, "Заголовок", level=1)
        assembler.add(BlockKind.PARAGRAPH, "Текст страницы.", page=2)
        assembler.add(BlockKind.TABLE, "a | b\n1 | 2", attrs={"rows": 2, "ok": True, "x": 1.5})
        return assembler.parsed(
            title="T", author="A", created=datetime(2024, 1, 2, 3, 4, tzinfo=UTC), warnings=("w",)
        )
