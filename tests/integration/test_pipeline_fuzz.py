"""Damaged documents through the whole pipeline.

Every file of ``corpus_mini`` is mutated with a fixed seed (flipped bytes, truncation, deleted,
inserted, duplicated and zeroed ranges). Each outcome must be a documented refusal or chunks
that satisfy every invariant; and the formats that run in the isolated worker must give exactly
the outcome they give in process.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import pytest

from rag_engine.chunking.registry import ChunkerRegistry
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import ChunkPolicy, ErrorCode, RagError, SourceKey
from rag_engine.ingestion.processing import DocumentProcessor
from rag_engine.parsing.isolation import ParserWorker
from rag_engine.parsing.registry import default_parsers
from rag_engine.ports import ParseLimits
from rag_engine.testing import FakeClock
from tests.chunking_checks import check_chunks
from tests.corpus_snapshot import CORPUS
from tests.factories import make_ids

COUNTER = HeuristicTokenCounter()
POLICY = ChunkPolicy(target_tokens=64, max_tokens=96, min_tokens=12, overlap_ratio=0.12)
LIMITS = ParseLimits(timeout_s=20)
VERSION = make_ids().document_version_id()
CLEAN = {ErrorCode.E_PARSE_FAILED, ErrorCode.E_UNSUPPORTED_FORMAT}
MUTATIONS_PER_FILE = 14
ISOLATED_FORMATS = (".pdf", ".docx", ".html")
Outcome = tuple[str, object]


def mutate(data: bytes, rng: random.Random) -> bytes:
    if not data:
        return data
    out = bytearray(data)
    kind = rng.randrange(6)
    start = rng.randrange(len(out))
    length = rng.randint(1, max(1, min(64, len(out) - start)))
    if kind == 0:
        for _ in range(rng.randint(1, 8)):
            out[rng.randrange(len(out))] = rng.randrange(256)
    elif kind == 1:
        del out[start:]
    elif kind == 2:
        del out[start : start + length]
    elif kind == 3:
        out[start:start] = bytes(rng.randrange(256) for _ in range(length))
    elif kind == 4:
        out[start:start] = out[start : start + length] * rng.randint(1, 20)
    else:
        out[start : start + length] = bytes(length)
    return bytes(out)


def mutants(name: str) -> Iterator[bytes]:
    rng = random.Random(f"fuzz:{name}")  # noqa: S311 - a fixed seed keeps the mutations reproducible
    data = (CORPUS / name).read_bytes()
    for _ in range(MUTATIONS_PER_FILE):
        yield mutate(data, rng)


def run(runner: DocumentProcessor, name: str, data: bytes) -> Outcome:
    try:
        done = runner.process(data, SourceKey(name), LIMITS, POLICY, VERSION)
    except RagError as error:
        failure = error
    else:
        check_chunks(done.parsed.document, done.chunks, POLICY, COUNTER, done.chunker_fingerprint)
        return "ok", (done.parsed.document, done.chunks)
    assert failure.code in CLEAN, failure
    return "error", (failure.code, failure.context.get("reason"))


@pytest.fixture(scope="module")
def local() -> DocumentProcessor:
    return DocumentProcessor(default_parsers(), FakeClock(), ChunkerRegistry(COUNTER))


@pytest.fixture(scope="module")
def isolated() -> Iterator[DocumentProcessor]:
    worker = ParserWorker(memory_mb=900, grace_s=2)
    parsers = default_parsers(isolate=("pdf", "docx", "html"), worker=worker)
    yield DocumentProcessor(parsers, FakeClock(), ChunkerRegistry(COUNTER))
    worker.close()


@pytest.mark.parametrize("name", sorted(p.name for p in CORPUS.iterdir()))
def test_damaged_documents_are_refused_cleanly_or_chunked_within_the_invariants(
    local: DocumentProcessor, name: str
) -> None:
    outcomes = [run(local, name, data) for data in mutants(name)]
    assert len(outcomes) == MUTATIONS_PER_FILE


def test_the_isolated_worker_gives_the_outcome_of_in_process_parsing(
    local: DocumentProcessor, isolated: DocumentProcessor
) -> None:
    names = [p.name for p in sorted(CORPUS.iterdir()) if p.name.endswith(ISOLATED_FORMATS)]
    assert len(names) >= 6
    for name in names:
        for data in mutants(name):
            assert run(isolated, name, data) == run(local, name, data), name
