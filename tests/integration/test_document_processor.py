"""DocumentProcessor on real files: format identification, parser choice, provenance, failures."""

from __future__ import annotations

import io
import json

import pytest

from rag_engine.chunking.registry import ChunkerRegistry
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import (
    ChunkPolicy,
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
from rag_engine.parsing.html import HtmlParser
from rag_engine.parsing.md import MarkdownParser
from rag_engine.parsing.registry import default_parsers
from rag_engine.parsing.txt import TxtParser
from rag_engine.ports import ParseHints, ParseLimits
from rag_engine.testing import FakeClock
from tests.chunking_checks import check_chunks
from tests.factories import make_ids
from tests.pdf_builder import build_pdf
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
    builtin = DocumentProcessor([MarkdownParser(), HtmlParser(), TxtParser()], FakeClock())
    with pytest.raises(RagError) as info:
        builtin.parse(b"%PDF-1.4\n" + b"0" * 32, SourceKey("doc.pdf"), LIMITS)
    assert info.value.code is ErrorCode.E_UNSUPPORTED_FORMAT
    assert info.value.context["mime"] == "application/pdf"
    assert "optional extra" in info.value.message


def test_a_pdf_is_parsed_when_its_extra_is_installed() -> None:
    data = build_pdf([["A line of text in a PDF file."]], title="Report")
    result = parse(data, "report.pdf")
    assert (result.parser_id, result.mime.mime) == ("pdf", "application/pdf")
    assert result.document.title == "Report" and result.document.blocks[0].page == 1
    with pytest.raises(RagError) as info:
        parse(build_pdf([None]), "scan.pdf")
    assert info.value.context["reason"] == "no_text_layer"


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


# ------------------------------------------------------------------- chunking

POLICY = ChunkPolicy(target_tokens=40, max_tokens=60, min_tokens=8, overlap_ratio=0.12)
COUNTER = HeuristicTokenCounter()
VERSION = make_ids().document_version_id()
GUIDE = (
    "# Guide\n\n" + " ".join(f"Sentence number {i} of the guide." for i in range(20)) + "\n\n"
    "## Setup\n\nRun the installer.\n\n## Usage\n\n"
    + " ".join(f"Usage sentence {i} explains one thing." for i in range(20))
)


def chunking_processor() -> DocumentProcessor:
    return DocumentProcessor(default_parsers(), FakeClock(), ChunkerRegistry(COUNTER))


def test_a_processed_document_has_chunks_and_the_strategy_that_made_them() -> None:
    done = chunking_processor().process(
        GUIDE.encode(), SourceKey("guide.md"), LIMITS, POLICY, VERSION
    )
    assert done.chunker_id == "structure_aware" and done.chunker_version == "1"
    assert done.parsed.parser_id == "md" and done.chunks
    check_chunks(done.parsed.document, done.chunks, POLICY, COUNTER, done.chunker_fingerprint)
    assert {c.level for c in done.chunks} == {0, 1}
    assert all(c.document_version_id == VERSION for c in done.chunks)


def test_the_strategy_follows_the_content_of_the_file() -> None:
    runner = chunking_processor()
    cases = {
        "a.txt": (b"first paragraph\n\nsecond paragraph", "recursive"),
        "b.txt": (b"one single paragraph of plain words", "fixed"),
        "c.md": (b"# T\n\ntext", "structure_aware"),
    }
    for name, (data, strategy) in cases.items():
        done = runner.process(data, SourceKey(name), LIMITS, POLICY, VERSION)
        assert done.chunker_id == strategy, name


def test_processing_is_deterministic_and_the_fingerprint_follows_the_policy() -> None:
    runner = chunking_processor()
    first = runner.process(GUIDE.encode(), SourceKey("g.md"), LIMITS, POLICY, VERSION)
    again = runner.process(GUIDE.encode(), SourceKey("g.md"), LIMITS, POLICY, VERSION)
    assert first.chunks == again.chunks and first.chunker_fingerprint == again.chunker_fingerprint
    other = ChunkPolicy(target_tokens=48, max_tokens=60, min_tokens=8, overlap_ratio=0.12)
    changed = runner.process(GUIDE.encode(), SourceKey("g.md"), LIMITS, other, VERSION)
    assert changed.chunker_fingerprint != first.chunker_fingerprint
    assert {c.id for c in changed.chunks}.isdisjoint({c.id for c in first.chunks})


def test_an_empty_document_has_no_chunks() -> None:
    done = chunking_processor().process(b"   \n\n", SourceKey("e.txt"), LIMITS, POLICY, VERSION)
    assert done.chunks == ()


def test_parse_failures_stop_before_chunking() -> None:
    with pytest.raises(RagError) as info:
        chunking_processor().process(
            b"\x89PNG\r\n\x1a\n" + b"0" * 64, SourceKey("p.md"), LIMITS, POLICY, VERSION
        )
    assert info.value.code is ErrorCode.E_UNSUPPORTED_FORMAT


def test_processing_without_chunkers_is_a_programming_error() -> None:
    with pytest.raises(RuntimeError, match="without chunkers"):
        processor().process(b"text", SourceKey("a.txt"), LIMITS, POLICY, VERSION)


def test_chunking_is_logged_as_a_stage_without_the_text() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    chunking_processor().process(
        f"# {SECRET}\n\n{SECRET}".encode(), SourceKey("s.md"), LIMITS, POLICY, VERSION
    )
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    stages = [(e["stage"], e["event"]) for e in events]
    assert stages == [
        ("process.parse", "stage.start"),
        ("process.parse", "stage.done"),
        ("process.chunk", "stage.start"),
        ("process.chunk", "stage.done"),
    ]
    assert events[2]["chunker"] == "structure_aware" and SECRET not in stream.getvalue()
