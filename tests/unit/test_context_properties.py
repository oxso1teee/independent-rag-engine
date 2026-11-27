"""NFR-07 as a property: over random corpora, budgets and policies nothing is silently lost."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rag_engine.context.builder import ContextBuilder, ContextPolicy
from rag_engine.domain import ErrorCode, HitScores, RagError, RetrievalHit
from rag_engine.ports import ModelLimits
from tests.context_support import Library, Words

pytestmark = pytest.mark.unit


def document_text(word_count: int, sentence_every: int) -> str:
    pieces = []
    for i in range(word_count):
        end = "." if sentence_every and i % sentence_every == sentence_every - 1 else ""
        pieces.append(f"w{i}{end}")
    return " ".join(pieces)


@st.composite
def scenarios(draw: st.DrawFn) -> tuple[Library, list[RetrievalHit], ContextPolicy, ModelLimits]:
    library = Library()
    names = ["a", "b", "c"][: draw(st.integers(1, 3))]
    for name in names:
        library.texts[name] = document_text(
            draw(st.integers(20, 500)), draw(st.sampled_from([0, 4, 9, 25]))
        )
    chunks = []
    for _ in range(draw(st.integers(1, 14))):
        name = draw(st.sampled_from(names))
        whole = library.texts[name]
        start = draw(st.integers(0, len(whole) - 2))
        end = draw(st.integers(start + 1, min(len(whole), start + 900)))
        chunks.append(library.add(name, start, end))
    if draw(st.booleans()):  # small-to-big: wrap some leaves in a section that covers them
        for leaf in list(chunks):
            name = leaf.source_key.removesuffix(".md")
            whole = library.texts[name]
            start = max(0, leaf.char_start - draw(st.integers(0, 200)))
            end = min(len(whole), leaf.char_end + draw(st.integers(0, 200)))
            parent = library.add(name, start, end, level=1)
            library.contents[leaf.ref.chunk_id] = replace(leaf, parent_id=parent.ref.chunk_id)
            chunks[chunks.index(leaf)] = library.contents[leaf.ref.chunk_id]
    order = draw(st.permutations(range(len(chunks))))
    chosen = [chunks[i] for i in order[: draw(st.integers(1, len(chunks)))]]
    hits = [
        RetrievalHit(
            chunk_ref=c.ref,
            rank=rank,
            scores=HitScores(fused=1 / rank, rerank=draw(st.one_of(st.none(), st.floats(-5, 5)))),
        )
        for rank, c in enumerate(chosen, start=1)
    ]
    if draw(st.booleans()) and chosen:  # a chunk garbage-collected after retrieval
        library.contents.pop(chosen[-1].ref.chunk_id, None)
    policy = ContextPolicy(
        max_chunks_per_document=draw(st.sampled_from([None, 1, 2, 3])),
        order=draw(st.sampled_from(["by_score", "by_document_order"])),
        small_to_big=draw(st.booleans()),
        min_evidence_items=draw(st.sampled_from([1, 1, 1, 2])),
        item_overhead_tokens=draw(st.integers(0, 10)),
    )
    limits = ModelLimits(draw(st.integers(450, 4000)), draw(st.integers(20, 200)))
    return library, hits, policy, limits


@settings(max_examples=1000, deadline=None)
@given(scenarios(), st.integers(1, 80), st.integers(10, 120))
def test_nothing_is_lost_silently_and_the_budget_is_never_exceeded(
    scenario: tuple[Library, list[RetrievalHit], ContextPolicy, ModelLimits],
    question_tokens: int,
    system_tokens: int,
) -> None:
    library, hits, policy, limits = scenario
    builder = ContextBuilder(Words(), library)
    arguments = {
        "question_tokens": question_tokens,
        "system_tokens": system_tokens,
        "limits": limits,
        "policy": policy,
    }
    failure: RagError | None = None
    try:
        context = builder.build(hits, **arguments)  # type: ignore[arg-type]
    except RagError as error:
        failure = error
    if failure is not None:
        assert failure.code is ErrorCode.E_CONTEXT_BUDGET  # the only allowed failure
        assert "hint" in failure.context
        return

    # the budget holds, and the report agrees with the evidence
    assert context.tokens_used <= context.budget.evidence_budget
    assert (
        context.report.tokens_used == context.tokens_used == sum(e.tokens for e in context.evidence)
    )

    # every retrieved chunk is accounted for exactly once: shown, merged into one, or dropped
    shown = [chunk for e in context.evidence for chunk in e.matched]
    dropped = [d.chunk_id for d in context.report.dropped]
    wanted = [h.chunk_ref.chunk_id for h in hits]
    assert len(shown) == len(set(shown)) and len(dropped) == len(set(dropped))
    assert not set(shown) & set(dropped)
    assert sorted(shown + dropped, key=str) == sorted(wanted, key=str)

    # evidence is an exact slice of its document, and a cut is always declared
    by_name = {
        c.ref.document_id: c.source_key.removesuffix(".md") for c in library.contents.values()
    }
    for item in context.evidence:
        whole = library.texts[by_name[item.document_id]]
        assert whole[item.locator.char_start : item.locator.char_end] == item.text
        assert item.truncated == (item.chunk_id in context.report.truncated)
        if item.truncated:
            assert item.tokens <= context.budget.evidence_budget
    assert len(context.evidence) >= policy.min_evidence_items

    # the same inputs give the same context
    again = builder.build(hits, **arguments)  # type: ignore[arg-type]
    assert again.context_hash == context.context_hash
