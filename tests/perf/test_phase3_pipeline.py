"""Phase 3 performance budgets: counting, parsing and chunking at realistic document sizes.

Inputs are generated (no network, no stored binaries). Every test prints its measurement; the
budgets are deliberate multiples of what ``make perf`` measures on the reference machine
(see docs/TESTING.md), so they catch an accidental quadratic step, not noise.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable

import pytest

from rag_engine.chunking.registry import STRATEGIES, ChunkerRegistry
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import ChunkPolicy, SourceKey
from rag_engine.ingestion.processing import DocumentProcessor
from rag_engine.parsing.isolation import ParserWorker
from rag_engine.parsing.registry import default_parsers
from rag_engine.ports import ParseHints, ParseLimits
from rag_engine.testing import FakeClock
from tests.docx_builder import build_docx
from tests.factories import make_ids
from tests.fixtures.generate_corpus_mini import EN_CLAUSES, EN_TAILS, RU_CLAUSES, RU_TAILS, Lcg
from tests.pdf_builder import build_pdf

COUNTER = HeuristicTokenCounter()
POLICY = ChunkPolicy()
LIMITS = ParseLimits()
VERSION = make_ids().document_version_id()
MEGABYTE = 1_000_000

# seconds; see docs/TESTING.md for the measurements these are derived from
BUDGET = {
    "count_6mb": 7.0,
    "markdown_1mb": 15.0,
    "html_1mb": 10.0,
    "text_1mb": 6.0,
    "chunk_1mb": 15.0,
    "pdf_100_pages": 5.0,
    "docx_2000_paragraphs": 10.0,
    "worker_start": 5.0,
    "worker_round_trip": 0.5,
}
PEAK_MEMORY_MB = 150


def timed[T](
    label: str, budget: float, action: Callable[[], T], capsys: pytest.CaptureFixture[str]
) -> T:
    started = time.perf_counter()
    result = action()
    seconds = time.perf_counter() - started
    with capsys.disabled():
        print(f"\nPhase 3 / {label}: {seconds:.3f}s (budget {budget}s)")
    assert seconds <= budget, f"{label} took {seconds:.2f}s, budget {budget}s"
    return result


def sentences(size: int, *, russian: bool = False) -> list[str]:
    rnd = Lcg(5)
    clauses, tails = (RU_CLAUSES, RU_TAILS) if russian else (EN_CLAUSES, EN_TAILS)
    out: list[str] = []
    total = 0
    while total < size:
        sentence = f"{rnd.pick(clauses)} {rnd.pick(tails)} (item {len(out)})."
        out.append(sentence)
        total += len(sentence) + 1
    return out


def markdown(size: int) -> bytes:
    parts: list[str] = []
    items = sentences(size, russian=True)
    for index in range(0, len(items), 6):
        if index % 60 == 0:
            parts.append(f"# Chapter {index // 60}")
        parts.append(f"## Section {index // 6}\n\n" + " ".join(items[index : index + 6]))
    return "\n\n".join(parts).encode()


def html_page(size: int) -> bytes:
    items = sentences(size)
    body = "".join(
        f"<h2>Part {i // 6}</h2><p>{' '.join(items[i : i + 6])}</p>"
        for i in range(0, len(items), 6)
    )
    return f"<html><body><main>{body}</main></body></html>".encode()


def paragraphs_text(size: int) -> bytes:
    items = sentences(size)
    return "\n\n".join(" ".join(items[i : i + 5]) for i in range(0, len(items), 5)).encode()


def processor(strategy: str = "auto") -> DocumentProcessor:
    return DocumentProcessor(default_parsers(), FakeClock(), ChunkerRegistry(COUNTER, strategy))


def test_counting_six_megabytes(capsys: pytest.CaptureFixture[str]) -> None:
    text = " ".join(sentences(5 * MEGABYTE)) + " ".join(sentences(MEGABYTE, russian=True))
    count = timed("count 6 MB", BUDGET["count_6mb"], lambda: COUNTER.count(text), capsys)
    assert count > 0


def test_markdown_of_one_megabyte(capsys: pytest.CaptureFixture[str]) -> None:
    data = markdown(MEGABYTE)
    done = timed(
        "markdown 1 MB parse+chunk",
        BUDGET["markdown_1mb"],
        lambda: processor().process(data, SourceKey("big.md"), LIMITS, POLICY, VERSION),
        capsys,
    )
    assert done.chunker_id == "structure_aware" and len(done.chunks) > 500


def test_html_of_one_megabyte(capsys: pytest.CaptureFixture[str]) -> None:
    data = html_page(MEGABYTE)
    done = timed(
        "html 1 MB parse+chunk",
        BUDGET["html_1mb"],
        lambda: processor().process(data, SourceKey("big.html"), LIMITS, POLICY, VERSION),
        capsys,
    )
    assert len(done.chunks) > 500


def test_plain_text_of_one_megabyte(capsys: pytest.CaptureFixture[str]) -> None:
    data = paragraphs_text(MEGABYTE)
    done = timed(
        "text 1 MB parse+chunk",
        BUDGET["text_1mb"],
        lambda: processor().process(data, SourceKey("big.txt"), LIMITS, POLICY, VERSION),
        capsys,
    )
    assert done.chunker_id == "recursive" and len(done.chunks) > 500


@pytest.mark.parametrize("strategy", [s for s in STRATEGIES if s != "auto"])
def test_every_strategy_chunks_one_megabyte(
    strategy: str, capsys: pytest.CaptureFixture[str]
) -> None:
    parsed = processor().parse(paragraphs_text(MEGABYTE), SourceKey("big.txt"), LIMITS)
    chunker = ChunkerRegistry(COUNTER, strategy).select(parsed.document)
    chunks = timed(
        f"chunk 1 MB with {strategy}",
        BUDGET["chunk_1mb"],
        lambda: chunker.chunk(
            parsed.document, POLICY, version_id=VERSION, normalizer_version="norm/1"
        ),
        capsys,
    )
    assert len(chunks) > 500


def test_pdf_of_one_hundred_pages(capsys: pytest.CaptureFixture[str]) -> None:
    items = sentences(100 * 2_000)
    pages = [
        [f"Report page {n + 1}", *[s[:70] for s in items[n * 20 : n * 20 + 20]], f"Page {n + 1}"]
        for n in range(100)
    ]
    data = build_pdf(pages)
    done = timed(
        "pdf 100 pages parse+chunk",
        BUDGET["pdf_100_pages"],
        lambda: processor().process(data, SourceKey("big.pdf"), LIMITS, POLICY, VERSION),
        capsys,
    )
    assert done.parsed.parser_id == "pdf" and done.chunks


def test_docx_of_two_thousand_paragraphs(capsys: pytest.CaptureFixture[str]) -> None:
    items = sentences(2_000 * 300)
    entries: list[tuple[str, str]] = []
    for index, sentence in enumerate(items[:2_000]):
        if index % 25 == 0:
            entries.append(("h1", f"Chapter {index // 25}"))
        entries.append(("p", sentence))
    data = build_docx(entries)
    done = timed(
        "docx 2000 paragraphs parse+chunk",
        BUDGET["docx_2000_paragraphs"],
        lambda: processor().process(data, SourceKey("big.docx"), LIMITS, POLICY, VERSION),
        capsys,
    )
    assert done.parsed.parser_id == "docx" and done.chunks


def test_isolated_worker_start_and_round_trip(capsys: pytest.CaptureFixture[str]) -> None:
    worker = ParserWorker(memory_mb=900)
    try:
        parsers = {p.parser_id: p for p in default_parsers(isolate=("html",), worker=worker)}
        hints = ParseHints(mime="text/html", limits=LIMITS)
        page = html_page(20_000)
        first = timed(
            "worker start + first document",
            BUDGET["worker_start"],
            lambda: parsers["html"].parse(page, hints),
            capsys,
        )
        again = timed(
            "worker round trip (20 KB)",
            BUDGET["worker_round_trip"],
            lambda: parsers["html"].parse(page, hints),
            capsys,
        )
        assert first == again
    finally:
        worker.close()


def test_peak_memory_of_a_large_document(capsys: pytest.CaptureFixture[str]) -> None:
    data = markdown(2 * MEGABYTE)
    runner = processor()
    tracemalloc.start()
    try:
        runner.process(data, SourceKey("big.md"), LIMITS, POLICY, VERSION)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    megabytes = peak / MEGABYTE
    with capsys.disabled():
        print(
            f"\nPhase 3 / peak memory, markdown 2 MB parse+chunk: {megabytes:.0f} MB "
            f"(budget {PEAK_MEMORY_MB} MB)"
        )
    assert megabytes <= PEAK_MEMORY_MB
