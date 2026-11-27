"""Building the context: a token budget measured, not assumed, and every chunk accounted for.

The builder walks the retrieved chunks best first and fills the evidence budget greedily. It
removes exact duplicates, folds overlapping neighbours of one version into their exact union,
caps how many places a document may take and, if asked, replaces a leaf by its section. A chunk
that cannot be shown is never lost silently: it is dropped with a reason, merged into a named
neighbour, or shown cut at a sentence with ``truncated`` set. If the evidence budget cannot hold
the required evidence the build fails with ``E_CONTEXT_BUDGET`` and a hint, instead of handing the
model a context that was cut without saying so.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rag_engine.context.truncate import truncate_to_tokens
from rag_engine.domain import (
    ChunkId,
    Context,
    ContextBudget,
    ContextReport,
    DocumentId,
    DocumentVersionId,
    DroppedChunk,
    DropReason,
    ErrorCode,
    Evidence,
    EvidenceLocator,
    IncludedChunk,
    MergedChunk,
    RagError,
    RetrievalHit,
    TextHash,
)
from rag_engine.ports import ChunkContent, ChunkContentReader, ModelLimits, TokenCounter

_HINT = "reduce top_k, enlarge the context window or choose a model with a larger window"


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    max_chunks_per_document: int | None = 3
    order: Literal["by_score", "by_document_order"] = "by_score"
    small_to_big: bool = False
    min_evidence_items: int = 1
    safety_margin_floor: int = 256
    safety_margin_ratio: float = 0.10
    item_overhead_tokens: int = 8
    """Tokens the prompt spends around each passage (its marker and locator line)."""


@dataclass(slots=True)
class _Item:
    content: ChunkContent
    matched: list[ChunkId]
    text: str
    start: int
    end: int
    truncated: bool
    score: float
    tokens: int
    first_rank: int


def _union(first: _Item, text: str, start: int, end: int) -> tuple[str, int, int]:
    """Exact union of two overlapping slices of one normalised text."""
    if start >= first.start:
        left_text, left_start, left_end = first.text, first.start, first.end
        right_text, right_start, right_end = text, start, end
    else:
        left_text, left_start, left_end = text, start, end
        right_text, right_start, right_end = first.text, first.start, first.end
    if right_end <= left_end:
        return left_text, left_start, left_end
    return left_text + right_text[left_end - right_start :], left_start, right_end


class ContextBuilder:
    def __init__(self, counter: TokenCounter, content: ChunkContentReader) -> None:
        self._counter = counter
        self._content = content

    def build(
        self,
        hits: Sequence[RetrievalHit],
        *,
        question_tokens: int,
        system_tokens: int,
        limits: ModelLimits,
        policy: ContextPolicy | None = None,
    ) -> Context:
        policy = policy or ContextPolicy()
        window = limits.context_window
        margin = max(policy.safety_margin_floor, math.ceil(policy.safety_margin_ratio * window))
        available = window - system_tokens - question_tokens - limits.max_new_tokens - margin
        if available <= 0:
            raise RagError(
                ErrorCode.E_CONTEXT_BUDGET,
                "the prompt leaves no room for evidence",
                context={
                    "window": window,
                    "system": system_tokens,
                    "question": question_tokens,
                    "max_new": limits.max_new_tokens,
                    "margin": margin,
                    "evidence_budget": available,
                    "hint": _HINT,
                },
            )
        budget = ContextBudget(
            window, system_tokens, question_tokens, limits.max_new_tokens, margin, available
        )

        ids = [hit.chunk_ref.chunk_id for hit in hits]
        contents = dict(self._content.read(ids))
        if policy.small_to_big:
            parents = [c.parent_id for c in contents.values() if c.parent_id is not None]
            contents.update(self._content.read([p for p in parents if p not in contents]))

        selected: list[_Item] = []
        dropped: list[DroppedChunk] = []
        merged: list[MergedChunk] = []
        seen: set[TextHash] = set()
        per_document: dict[DocumentId, int] = {}
        versions: dict[DocumentVersionId, list[_Item]] = {}

        def used() -> int:
            return sum(item.tokens for item in selected)

        def cost(text: str) -> int:
            return self._counter.count(text) + policy.item_overhead_tokens

        for hit in hits:
            chunk_id = hit.chunk_ref.chunk_id
            leaf = contents.get(chunk_id)
            if leaf is None or not leaf.text.strip():  # gone, or nothing to show
                dropped.append(DroppedChunk(chunk_id, DropReason.MISSING))
                continue
            unit = leaf
            if policy.small_to_big and leaf.parent_id is not None:
                parent = contents.get(leaf.parent_id)
                if parent is not None and parent.text.strip():
                    held = next(
                        (i for i in selected if i.content.ref.chunk_id == parent.ref.chunk_id), None
                    )
                    if held is not None:
                        held.matched.append(chunk_id)
                        continue
                    if cost(parent.text) <= available - used():
                        unit = parent
            digest = TextHash.of_text(unit.text)
            if digest in seen:
                dropped.append(DroppedChunk(chunk_id, DropReason.DUPLICATE))
                continue
            neighbours = [
                i
                for i in versions.get(unit.ref.version_id, [])
                if unit.char_start < i.end and i.start < unit.char_end
            ]
            if len(neighbours) > 1:
                dropped.append(DroppedChunk(chunk_id, DropReason.OVERLAP))
                continue
            score = hit.scores.rerank if hit.scores.rerank is not None else hit.scores.fused
            if neighbours:
                item = neighbours[0]
                text, start, end = _union(item, unit.text, unit.char_start, unit.char_end)
                if text == item.text:
                    dropped.append(DroppedChunk(chunk_id, DropReason.OVERLAP))
                    continue
                grown = cost(text) - item.tokens
                if grown > available - used():
                    dropped.append(DroppedChunk(chunk_id, DropReason.BUDGET))
                    continue
                item.text, item.start, item.end = text, start, end
                item.tokens += grown
                item.matched.append(chunk_id)
                merged.append(MergedChunk(chunk_id, item.content.ref.chunk_id))
                seen.add(digest)
                continue
            document = unit.ref.document_id
            cap = policy.max_chunks_per_document
            if cap is not None and per_document.get(document, 0) >= cap:
                dropped.append(DroppedChunk(chunk_id, DropReason.DOC_CAP))
                continue
            tokens = cost(unit.text)
            text, truncated = unit.text, False
            if tokens > available - used():
                if selected:
                    dropped.append(DroppedChunk(chunk_id, DropReason.BUDGET))
                    continue
                text = truncate_to_tokens(
                    unit.text, available - used() - policy.item_overhead_tokens, self._counter.count
                )
                if not text.strip():
                    dropped.append(DroppedChunk(chunk_id, DropReason.BUDGET))
                    continue
                truncated, tokens = True, cost(text)
            item = _Item(
                unit,
                [chunk_id],
                text,
                unit.char_start,
                unit.char_start + len(text) if truncated else unit.char_end,
                truncated,
                0.0 if score is None else score,
                tokens,
                hit.rank,
            )
            selected.append(item)
            versions.setdefault(unit.ref.version_id, []).append(item)
            per_document[document] = per_document.get(document, 0) + 1
            seen.add(digest)

        if len(selected) < policy.min_evidence_items:
            raise RagError(
                ErrorCode.E_CONTEXT_BUDGET,
                "the evidence budget cannot hold the required evidence",
                context={
                    "evidence_budget": available,
                    "required": policy.min_evidence_items,
                    "included": len(selected),
                    "hint": _HINT,
                },
            )
        return self._finish(selected, dropped, merged, budget, policy)

    def _finish(
        self,
        selected: list[_Item],
        dropped: list[DroppedChunk],
        merged: list[MergedChunk],
        budget: ContextBudget,
        policy: ContextPolicy,
    ) -> Context:
        ordered = list(selected)
        if policy.order == "by_document_order":
            first: dict[DocumentId, int] = {}
            for item in selected:
                first.setdefault(item.content.ref.document_id, item.first_rank)
            ordered.sort(key=lambda i: (first[i.content.ref.document_id], i.start))
        evidence = tuple(
            Evidence(
                evidence_id=f"E{number}",
                chunk_id=item.content.ref.chunk_id,
                matched=tuple(item.matched),
                document_id=item.content.ref.document_id,
                version_id=item.content.ref.version_id,
                text=item.text,
                truncated=item.truncated,
                locator=EvidenceLocator(
                    source_name=item.content.source_name,
                    source_key=item.content.source_key,
                    title=item.content.title,
                    heading_path=item.content.heading_path,
                    page_start=item.content.page_start,
                    page_end=item.content.page_end,
                    char_start=item.start,
                    char_end=item.end,
                ),
                score=item.score,
                tokens=item.tokens,
            )
            for number, item in enumerate(ordered, start=1)
        )
        tokens_used = sum(item.tokens for item in evidence)
        digest = hashlib.sha256(
            json.dumps(
                {
                    "budget": [budget.window, budget.system, budget.question, budget.max_new],
                    "evidence": [[str(e.chunk_id), e.text, e.truncated] for e in evidence],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        report = ContextReport(
            included=tuple(IncludedChunk(e.chunk_id, e.evidence_id, e.tokens) for e in evidence),
            dropped=tuple(dropped),
            merged=tuple(merged),
            truncated=tuple(e.chunk_id for e in evidence if e.truncated),
            tokens_used=tokens_used,
            budget=budget.evidence_budget,
            context_hash=digest,
        )
        return Context(evidence, budget, tokens_used, report, digest)
