"""DocumentProcessor on real files: format identification, parser choice, provenance, failures."""

from __future__ import annotations

import io
import json

import pytest

from rag_engine.domain import (
    ContentHash,
    ErrorCode,
    ParsedDocument,
    RagError,
    SourceKey,
    SyncStatus,
)
from rag_engine.ingestion.processing import DocumentProcessor, ParsedResult, read_limited
from rag_engine.ingestion.sources import FilesystemSource
from rag_engine.observability import configure_logging
from rag_engine.parsing.cleaners import CleanerSet, compile_cleaners
from rag_engine.parsing.registry import default_parsers
from rag_engine.ports import ParseHints, ParseLimits
from rag_engine.testing import FakeClock
from tests.workspace import Workspace

LIMITS = ParseLimits()
SECRET = "launch codes are 0000-SECRET-BODY"


def processor(cleaners: CleanerSet | None = None) -> DocumentProcessor:
    return DocumentProcessor(default_parsers(cleaners), FakeClock())


def parse(data: bytes, key: str, limits: ParseLimits = LIMITS) -> ParsedResult:
    return processor().parse(data, SourceKey(key), limits)


def test_the_parser_is_chosen_by_content_not_by_extension() -> None:
    cases = {
        ("# Title\n\ntext", "notes.md"): ("md", "text/markdown"),
        ("<html><body><h1>T</h1><p>x</p></body></html>", "page.txt"): ("html", "text/html"),
        ("just words", "readme.md"): ("md", "text/markdown"),  # extension breaks the text tie
        ("just words", "words.txt"): ("txt", "text/plain"),
        ("just words", "no_extension"): ("txt", "text/plain"),
    }
    for (content, name), (parser_id, mime) in cases.items():
        result = parse(content.encode(), name)
        assert (result.parser_id, result.mime.mime) == (parser_id, mime), name
        assert isinstance(result.document, ParsedDocument)


def test_the_result_carries_the_provenance_of_a_version() -> None:
    data = "# Заголовок\n\nТекст документа.".encode()
    result = parse(data, "a.md")
    assert result.parser_id == "md" and result.parser_version == "1"
    assert result.normalizer_version == "norm/1"
    assert result.content_hash == ContentHash.of_bytes(data) and result.size_bytes == len(data)
    assert result.mime.encoding == "utf_8"
    assert result.document.title == "Заголовок"


def test_the_normaliser_version_follows_the_cleaners() -> None:
    plain = parse(b"text", "a.txt")
    cleaned = processor(compile_cleaners([r"noise"])).parse(b"text", SourceKey("a.txt"), LIMITS)
    assert plain.normalizer_version != cleaned.normalizer_version


def test_the_detected_encoding_reaches_the_parser() -> None:
    text = "Это достаточно длинный русский текст, записанный в старой кодировке. " * 3
    result = parse(text.encode("cp1251"), "old.txt")
    assert result.mime.encoding == "cp1251"
    assert result.document.normalized_text == text.strip()
    assert result.document.warnings == ()


def test_unsupported_content_is_refused_before_any_parser_runs() -> None:
    with pytest.raises(RagError) as png:
        parse(b"\x89PNG\r\n\x1a\n" + b"0" * 64, "pic.md")
    assert png.value.code is ErrorCode.E_UNSUPPORTED_FORMAT
    assert png.value.context["mime"] == "image/png"


def test_a_supported_format_without_an_installed_parser_says_so() -> None:
    with pytest.raises(RagError) as info:
        parse(b"%PDF-1.4\n" + b"0" * 32, "doc.pdf")
    assert info.value.code is ErrorCode.E_UNSUPPORTED_FORMAT
    assert info.value.context["mime"] == "application/pdf"
    assert "optional extra" in info.value.message


class Exploding:
    parser_id = "boom"
    version = "1"
    normalizer_version = "n"

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def supports(self, mime: str | None, extension: str | None) -> bool:
        return True

    def parse(self, data: bytes, hints: ParseHints) -> ParsedDocument:
        raise self._error


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (ValueError(SECRET), "corrupt"),
        (MemoryError(), "memory"),
        (RecursionError(), "too_deep"),
    ],
)
def test_a_parser_that_misbehaves_never_escapes_as_it_is(error: BaseException, reason: str) -> None:
    runner = DocumentProcessor([Exploding(error)], FakeClock())
    with pytest.raises(RagError) as info:
        runner.parse(b"text", SourceKey("a.txt"), LIMITS)
    assert info.value.code is ErrorCode.E_PARSE_FAILED
    assert info.value.context["reason"] == reason
    assert SECRET not in repr(info.value.to_dict())  # the message of the bug is not kept


def test_a_documented_parser_error_is_passed_on_unchanged() -> None:
    error = RagError(ErrorCode.E_PARSE_FAILED, "bad", context={"reason": "encrypted"})
    runner = DocumentProcessor([Exploding(error)], FakeClock())
    with pytest.raises(RagError) as kept:
        runner.parse(b"text", SourceKey("a.txt"), LIMITS)
    assert kept.value is error


def test_reading_stops_at_the_limit_without_reading_everything() -> None:
    class Counting(io.BytesIO):
        read_total = 0

        def read(self, size: int | None = -1) -> bytes:
            data = super().read(size)
            self.read_total += len(data)
            return data

    assert read_limited(io.BytesIO(b"x" * 10), 10) == b"x" * 10
    assert read_limited(io.BytesIO(b""), 10) == b""
    huge = Counting(b"x" * 50_000_000)
    with pytest.raises(RagError) as info:
        read_limited(huge, 1_000)
    assert info.value.context["reason"] == "too_large"
    assert huge.read_total <= 1_001


def test_parser_limits_are_passed_through() -> None:
    with pytest.raises(RagError) as info:
        parse(("word " * 100).encode(), "a.txt", ParseLimits(max_text_chars=50))
    assert info.value.context["reason"] == "too_large"


def test_files_of_a_scanned_source_are_processed_and_hash_like_the_scan(
    workspace: Workspace,
) -> None:
    workspace.write("guide/intro.md", "# Intro\n\nFirst paragraph.\n\n- item\n")
    workspace.write("guide/page.html", "<main><h1>Page</h1><p>Body &amp; more.</p></main>")
    workspace.write("notes.txt", "plain notes\n\nsecond paragraph")
    workspace.run()
    source = FilesystemSource(workspace.root)
    runner = processor()
    states = {
        s.source_key.value: s
        for s in workspace.store.reader().list_sync_states(workspace.source.id)
    }
    assert set(states) == {"guide/intro.md", "guide/page.html", "notes.txt"}
    for entry in source.scan({}).entries:
        with source.open(entry.key) as stream:
            data = runner.read(stream, LIMITS)
        result = runner.parse(data, entry.key, LIMITS)
        state = states[entry.key.value]
        assert state.status is SyncStatus.CHANGED
        assert result.content_hash == state.last_content_hash  # the same bytes the scan hashed
        assert result.document.blocks


def test_logs_describe_the_stage_but_never_the_document() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    parse(f"# {SECRET}\n\n{SECRET}".encode(), "secret.md")
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [e["event"] for e in events] == ["stage.start", "stage.done"]
    assert events[0]["stage"] == "process.parse" and events[0]["parser"] == "md"
    assert SECRET not in stream.getvalue() and "secret.md" not in stream.getvalue()


def test_processing_the_same_bytes_twice_gives_equal_results() -> None:
    data = b"# T\n\ntext & more\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    first, second = parse(data, "a.md"), parse(data, "a.md")
    assert first.document == second.document
    assert first.content_hash == second.content_hash
