"""The context builder: budget, duplicates, overlaps, small-to-big, truncation and accounting."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_engine.context.builder import ContextBuilder, ContextPolicy
from rag_engine.domain import (
    Context,
    DropReason,
    ErrorCode,
    HitScores,
    RagError,
    RetrievalHit,
)
from rag_engine.ports import ChunkContent, ModelLimits
from tests.context_support import WHOLE, Library, Words, words

pytestmark = pytest.mark.unit


def hit(content: ChunkContent, rank: int, fused: float = 0.5) -> RetrievalHit:
    return RetrievalHit(chunk_ref=content.ref, rank=rank, scores=HitScores(fused=fused))


LIMITS = ModelLimits(context_window=1000, max_new_tokens=100)
NO_OVERHEAD = ContextPolicy(item_overhead_tokens=0)


def build(
    library: Library, hits: Sequence[RetrievalHit], policy: ContextPolicy = NO_OVERHEAD
) -> Context:
    return ContextBuilder(Words(), library).build(
        hits, question_tokens=10, system_tokens=50, limits=LIMITS, policy=policy
    )


def test_the_budget_subtracts_every_measured_term_and_a_safety_margin() -> None:
    library = Library()
    context = build(library, [hit(library.add("a", 0, 100, text=words("a", 10)), 1)])
    # margin = max(256, 10 % of 1000) = 256; 1000 - 50 - 10 - 100 - 256 = 584
    assert (context.budget.margin, context.budget.evidence_budget) == (256, 584)
    wide = ContextBuilder(Words(), library).build(
        [hit(library.add("b", 0, 100, text=words("b", 10)), 1)],
        question_tokens=10,
        system_tokens=50,
        limits=ModelLimits(context_window=8192, max_new_tokens=512),
        policy=NO_OVERHEAD,
    )
    assert wide.budget.margin == 820  # 10 % of 8192 rounded up, above the 256 floor


def test_chunks_fill_the_budget_in_rank_order_and_the_rest_are_reported() -> None:
    library = Library()
    one = library.add("a", 0, 100, text=words("a", 300))
    two = library.add("b", 0, 100, text=words("b", 200))
    three = library.add("c", 0, 100, text=words("c", 50))
    hits = [hit(one, 1), hit(two, 2), hit(three, 3)]
    context = build(library, hits)
    assert [item.evidence_id for item in context.evidence] == ["E1", "E2", "E3"]
    assert context.tokens_used == 550 and context.report.tokens_used == 550
    tight = build(library, hits, ContextPolicy(item_overhead_tokens=40))
    # overhead 40: 340 fits, 240 fits (580), 90 does not (670 > 584)
    assert [d.reason for d in tight.report.dropped] == [DropReason.BUDGET]
    assert tight.tokens_used <= tight.budget.evidence_budget


def test_a_later_smaller_chunk_can_still_fit_after_a_large_one_is_skipped() -> None:
    library = Library()
    first = library.add("a", 0, 100, text=words("a", 400))
    too_big = library.add("b", 0, 100, text=words("b", 300))
    small = library.add("c", 0, 100, text=words("c", 100))
    context = build(library, [hit(first, 1), hit(too_big, 2), hit(small, 3)])
    assert [item.chunk_id for item in context.evidence] == [first.ref.chunk_id, small.ref.chunk_id]
    assert [(d.chunk_id, d.reason) for d in context.report.dropped] == [
        (too_big.ref.chunk_id, DropReason.BUDGET)
    ]


def test_an_identical_text_is_included_once() -> None:
    library = Library()
    one = library.add("a", 0, 10, text="same words here")
    copy = library.add("b", 0, 10, text="same words here")
    context = build(library, [hit(one, 1), hit(copy, 2)])
    assert len(context.evidence) == 1
    assert [(d.chunk_id, d.reason) for d in context.report.dropped] == [
        (copy.ref.chunk_id, DropReason.DUPLICATE)
    ]


def test_a_document_cannot_exceed_its_chunk_cap() -> None:
    library = Library()
    chunks = [library.add("a", i * 100, i * 100 + 5, text=words(f"t{i}x", 5)) for i in range(5)]
    hits = [hit(c, i + 1) for i, c in enumerate(chunks)]
    context = build(library, hits)
    assert len(context.evidence) == 3  # the default cap
    assert [d.reason for d in context.report.dropped] == [DropReason.DOC_CAP] * 2
    unlimited = ContextPolicy(item_overhead_tokens=0, max_chunks_per_document=None)
    assert len(build(library, hits, unlimited).evidence) == 5


def test_a_chunk_inside_an_included_one_is_dropped_as_overlap() -> None:
    library = Library()
    wide = library.add("a", 0, 200)
    inner = library.add("a", 20, 80)
    context = build(library, [hit(wide, 1), hit(inner, 2)])
    assert len(context.evidence) == 1
    assert [(d.chunk_id, d.reason) for d in context.report.dropped] == [
        (inner.ref.chunk_id, DropReason.OVERLAP)
    ]


def test_overlapping_chunks_of_one_version_merge_into_the_exact_original_text() -> None:
    library = Library()
    whole = WHOLE["a"]
    left = library.add("a", 0, 120)
    right = library.add("a", 80, 200)
    context = build(library, [hit(left, 1), hit(right, 2)])
    assert len(context.evidence) == 1
    merged = context.evidence[0]
    assert merged.text == whole[0:200]
    assert (merged.locator.char_start, merged.locator.char_end) == (0, 200)
    assert merged.matched == (left.ref.chunk_id, right.ref.chunk_id)
    assert [(m.chunk_id, m.into) for m in context.report.merged] == [
        (right.ref.chunk_id, left.ref.chunk_id)
    ]
    assert context.report.dropped == ()


def test_small_to_big_replaces_a_leaf_by_its_section_when_it_fits() -> None:
    library = Library()
    section = words("s", 120)
    parent = library.add("a", 0, 600, text=section, level=1, headings=("Backups",))
    leaf_one = library.add("a", 0, 300, text=section[:200], parent=parent.ref.chunk_id)
    leaf_two = library.add("a", 300, 600, text=section[200:], parent=parent.ref.chunk_id)
    hits = [hit(leaf_one, 1), hit(leaf_two, 2)]
    context = build(library, hits, ContextPolicy(item_overhead_tokens=0, small_to_big=True))
    assert len(context.evidence) == 1
    item = context.evidence[0]
    assert item.chunk_id == parent.ref.chunk_id and item.text == section
    assert item.matched == (leaf_one.ref.chunk_id, leaf_two.ref.chunk_id)  # the hits it stands for
    assert item.locator.heading_path == ("Backups",)
    assert context.report.dropped == ()
    off = build(library, hits)
    assert [e.chunk_id for e in off.evidence] == [leaf_one.ref.chunk_id, leaf_two.ref.chunk_id]


def test_small_to_big_keeps_the_leaf_when_the_section_does_not_fit() -> None:
    library = Library()
    parent = library.add("a", 0, 5000, text=words("s", 900), level=1)
    leaf = library.add("a", 0, 100, text=words("l", 20), parent=parent.ref.chunk_id)
    policy = ContextPolicy(item_overhead_tokens=0, small_to_big=True)
    assert build(library, [hit(leaf, 1)], policy).evidence[0].chunk_id == leaf.ref.chunk_id


def test_by_document_order_groups_documents_and_follows_the_text() -> None:
    library = Library()
    a2 = library.add("a", 300, 310, text=words("a2x", 5))
    b1 = library.add("b", 0, 10, text=words("b1x", 5))
    a1 = library.add("a", 0, 10, text=words("a1x", 5))
    hits = [hit(a2, 1), hit(b1, 2), hit(a1, 3)]
    by_score = build(library, hits)
    assert [e.chunk_id for e in by_score.evidence] == [
        a2.ref.chunk_id,
        b1.ref.chunk_id,
        a1.ref.chunk_id,
    ]
    ordered = build(library, hits, ContextPolicy(item_overhead_tokens=0, order="by_document_order"))
    assert [e.chunk_id for e in ordered.evidence] == [
        a1.ref.chunk_id,
        a2.ref.chunk_id,
        b1.ref.chunk_id,
    ]


def test_a_top_hit_that_cannot_fit_is_cut_at_a_sentence_and_reported() -> None:
    library = Library()
    sentence = "One two three four five six seven eight nine ten."
    text = " ".join([sentence] * 100)  # 1000 words
    only = library.add("a", 0, len(text), text=text)
    context = build(library, [hit(only, 1)])
    item = context.evidence[0]
    assert item.truncated and item.text.endswith("ten.") and len(item.text) < len(text)
    assert text.startswith(item.text)  # still an exact prefix of the chunk
    assert item.locator.char_end == item.locator.char_start + len(item.text)
    assert item.tokens <= context.budget.evidence_budget
    assert context.report.truncated == (only.ref.chunk_id,)


def test_a_single_overlong_sentence_is_cut_at_a_word() -> None:
    library = Library()
    text = words("w", 900)
    only = library.add("a", 0, len(text), text=text)
    item = build(library, [hit(only, 1)]).evidence[0]
    assert item.truncated and item.tokens <= 584 and text.startswith(item.text)


def test_not_enough_room_for_the_required_evidence_is_a_budget_error_with_a_hint() -> None:
    library = Library()
    chunk = library.add("a", 0, 10, text="short text")
    tiny = ModelLimits(context_window=400, max_new_tokens=100)  # 400 - 50 - 10 - 100 - 256 < 0
    with pytest.raises(RagError) as info:
        ContextBuilder(Words(), library).build(
            [hit(chunk, 1)], question_tokens=10, system_tokens=50, limits=tiny, policy=NO_OVERHEAD
        )
    assert info.value.code is ErrorCode.E_CONTEXT_BUDGET
    assert {"window", "evidence_budget"} <= set(info.value.context)
    with pytest.raises(RagError) as second:
        build(library, [hit(chunk, 1)], ContextPolicy(item_overhead_tokens=0, min_evidence_items=2))
    assert second.value.code is ErrorCode.E_CONTEXT_BUDGET


def test_a_chunk_that_vanished_is_reported_not_ignored() -> None:
    library = Library()
    ghost = library.add("a", 0, 10, text="present text")
    ghost_hit = hit(ghost, 2)
    library.contents.pop(ghost.ref.chunk_id)
    other = library.add("b", 0, 10, text="another passage")
    context = build(library, [ghost_hit, hit(other, 3)])
    assert [(d.chunk_id, d.reason) for d in context.report.dropped] == [
        (ghost.ref.chunk_id, DropReason.MISSING)
    ]


def test_the_hash_identifies_the_exact_evidence_and_budget() -> None:
    library = Library()
    chunk = library.add("a", 0, 10, text="stable text here")
    first = build(library, [hit(chunk, 1)])
    assert first.context_hash == build(library, [hit(chunk, 1)]).context_hash
    other = library.add("a", 20, 30, text="changed text here")
    assert build(library, [hit(other, 1)]).context_hash != first.context_hash
    assert first.report.context_hash == first.context_hash


def test_evidence_carries_the_provenance_a_citation_needs_and_the_score_used() -> None:
    library = Library()
    chunk = library.add("a", 5, 15, text="provenance text", headings=("Policy", "Backups"))
    reranked = RetrievalHit(chunk_ref=chunk.ref, rank=1, scores=HitScores(fused=0.2, rerank=3.5))
    item = build(library, [reranked]).evidence[0]
    assert item.score == 3.5 and item.matched == (chunk.ref.chunk_id,)
    assert (item.document_id, item.version_id) == (chunk.ref.document_id, chunk.ref.version_id)
    assert (item.locator.source_name, item.locator.source_key) == ("notes", "a.md")
    assert item.locator.heading_path == ("Policy", "Backups")
    assert (item.locator.char_start, item.locator.char_end) == (5, 15)


def test_a_chunk_with_no_visible_text_is_reported_and_never_shown() -> None:
    library = Library()
    blank = library.add("a", 0, 5, text="   ")
    real = library.add("b", 0, 10, text="real passage here")
    context = build(library, [hit(blank, 1), hit(real, 2)])
    assert [e.chunk_id for e in context.evidence] == [real.ref.chunk_id]
    assert [(d.chunk_id, d.reason) for d in context.report.dropped] == [
        (blank.ref.chunk_id, DropReason.MISSING)
    ]
