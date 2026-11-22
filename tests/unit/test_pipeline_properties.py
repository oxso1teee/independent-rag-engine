"""Properties of parsing followed by chunking, for generated Markdown, HTML and text.

Whatever the input and the size policy, every strategy either refuses the document with a
documented error or returns chunks that satisfy every invariant of the data model, cover the
text, respect the size bounds and are the same on a second run.
"""

from __future__ import annotations

import html

from hypothesis import given, settings
from hypothesis import strategies as st

from rag_engine.chunking.registry import STRATEGIES, ChunkerRegistry
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import ChunkPolicy, ErrorCode, RagError, SourceKey
from rag_engine.ingestion.processing import DocumentProcessor, ProcessedDocument
from rag_engine.parsing.registry import default_parsers
from rag_engine.ports import ParseLimits
from rag_engine.testing import FakeClock
from tests.chunking_checks import check_chunks
from tests.factories import make_ids

COUNTER = HeuristicTokenCounter()
PARSERS = default_parsers()
VERSION = make_ids().document_version_id()
CLEAN_FAILURES = {ErrorCode.E_PARSE_FAILED, ErrorCode.E_UNSUPPORTED_FORMAT}

_WORDS = st.text(
    alphabet=st.characters(codec="utf-8", exclude_categories=("Cs", "Cc")), min_size=1, max_size=14
)
_SENTENCE = st.lists(_WORDS, min_size=1, max_size=40).map(" ".join)

_MARKDOWN_LINE = st.one_of(
    _SENTENCE,
    _SENTENCE.map("# {}".format),
    _SENTENCE.map("## {}".format),
    _SENTENCE.map("- {}".format),
    _SENTENCE.map("1. {}".format),
    _SENTENCE.map("> {}".format),
    _SENTENCE.map("    {}".format),
    _SENTENCE.map("| {} | b |".format),
    st.sampled_from(["", "", "```", "---", "|---|---|", "***"]),
)
MARKDOWN = st.lists(_MARKDOWN_LINE, max_size=40).map("\n".join)

_TAGS = ["p", "h1", "h2", "h3", "li", "pre", "blockquote", "td"]
HTML = st.lists(st.tuples(st.sampled_from(_TAGS), _SENTENCE), max_size=25).map(
    lambda parts: "<main>" + "".join(f"<{t}>{html.escape(s)}</{t}>" for t, s in parts) + "</main>"
)

TEXT = st.text(max_size=3000)

POLICIES = st.builds(
    lambda target, extra, low, overlap: ChunkPolicy(
        target_tokens=target,
        max_tokens=target + extra,
        min_tokens=max(1, min(low, target)),
        overlap_ratio=overlap,
    ),
    target=st.integers(min_value=8, max_value=120),
    extra=st.integers(min_value=0, max_value=80),
    low=st.integers(min_value=1, max_value=120),
    overlap=st.floats(min_value=0, max_value=0.45),
)


def process(data: bytes, key: str, policy: ChunkPolicy, strategy: str) -> ProcessedDocument | None:
    runner = DocumentProcessor(PARSERS, FakeClock(), ChunkerRegistry(COUNTER, strategy))
    try:
        return runner.process(data, SourceKey(key), ParseLimits(), policy, VERSION)
    except RagError as error:
        failure = error
    assert failure.code in CLEAN_FAILURES
    return None


def check(data: bytes, key: str, policy: ChunkPolicy) -> None:
    for strategy in STRATEGIES:
        done = process(data, key, policy, strategy)
        if done is None:
            continue
        check_chunks(done.parsed.document, done.chunks, policy, COUNTER, done.chunker_fingerprint)
        again = process(data, key, policy, strategy)
        assert again is not None and again.chunks == done.chunks


@given(MARKDOWN, POLICIES)
@settings(max_examples=60, deadline=None)
def test_markdown_always_chunks_within_the_invariants(text: str, policy: ChunkPolicy) -> None:
    check(text.encode(), "doc.md", policy)


@given(HTML, POLICIES)
@settings(max_examples=60, deadline=None)
def test_html_always_chunks_within_the_invariants(text: str, policy: ChunkPolicy) -> None:
    check(text.encode(), "page.html", policy)


@given(TEXT, POLICIES)
@settings(max_examples=60, deadline=None)
def test_plain_text_always_chunks_within_the_invariants(text: str, policy: ChunkPolicy) -> None:
    check(text.encode(), "notes.txt", policy)


def squeezed(text: str) -> str:
    return "".join(text.split())


@given(st.lists(_SENTENCE, min_size=1, max_size=30), POLICIES)
@settings(max_examples=40, deadline=None)
def test_without_overlap_the_leaves_rebuild_the_text(
    sentences: list[str], policy: ChunkPolicy
) -> None:
    exact = ChunkPolicy(
        target_tokens=policy.target_tokens,
        max_tokens=policy.max_tokens,
        min_tokens=policy.min_tokens,
        overlap_ratio=0.0,
    )
    data = "\n\n".join(sentences).encode()
    for strategy in STRATEGIES:
        done = process(data, "words.txt", exact, strategy)
        if done is None:
            continue
        leaves = [c.text for c in done.chunks if c.level == 0]
        assert squeezed("".join(leaves)) == squeezed(done.parsed.document.normalized_text)
