"""Arguments, display data and rendering of ``rag search`` and ``rag context``.

The data that the API shares (filter parsing, hit details, the JSON form of an outcome) lives in
``rag_engine.retrieval.presentation``; this module adds the text rendering of the command line.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_engine.domain import Context
from rag_engine.retrieval.presentation import (
    HitDetail,
    load_details,
    outcome_to_dict,
    parse_filters,
)
from rag_engine.retrieval.service import SearchOutcome

__all__ = [
    "HitDetail",
    "context_to_dict",
    "describe_context",
    "describe_hits",
    "load_details",
    "outcome_to_dict",
    "parse_filters",
]

_PREVIEW_CHARS = 200


def describe_hits(outcome: SearchOutcome, details: Sequence[HitDetail], *, full: bool) -> list[str]:
    lines: list[str] = []
    for item in details:
        score = "" if item.fused is None else f"{item.fused:.4f}"
        where = f"{item.source}/{item.path}" if item.source else item.path
        headings = f"  [{' > '.join(item.heading_path)}]" if item.heading_path else ""
        lines.append(f"{item.rank:>2}  {score:>7}  {where}{headings}")
        text = " ".join(item.text.split())
        if not full and len(text) > _PREVIEW_CHARS:
            text = text[: _PREVIEW_CHARS - 1] + "…"
        lines.append(f"    {text}")
    if not details:
        lines.append("no results")
    for name in outcome.result.fallbacks:
        lines.append(f"note: {name}")
    return lines


def context_to_dict(
    outcome: SearchOutcome, context: Context, *, system_tokens: int, question_tokens: int
) -> dict[str, object]:
    """The context, its budget arithmetic and the full accounting of what was left out."""
    budget, report = context.budget, context.report
    return {
        "request_id": str(outcome.result.request_id),
        "mode": outcome.diagnostics.mode.value,
        "index_generation": outcome.result.index_generation,
        "fallbacks": list(outcome.result.fallbacks),
        "budget": {
            "window": budget.window,
            "system": system_tokens,
            "question": question_tokens,
            "max_new": budget.max_new,
            "margin": budget.margin,
            "evidence_budget": budget.evidence_budget,
            "tokens_used": context.tokens_used,
        },
        "context_hash": context.context_hash,
        "evidence": [
            {
                "id": item.evidence_id,
                "chunk_id": str(item.chunk_id),
                "matched": [str(chunk) for chunk in item.matched],
                "document_id": str(item.document_id),
                "version_id": str(item.version_id),
                "source": item.locator.source_name,
                "path": item.locator.source_key,
                "title": item.locator.title,
                "heading_path": list(item.locator.heading_path),
                "pages": [item.locator.page_start, item.locator.page_end],
                "char_start": item.locator.char_start,
                "char_end": item.locator.char_end,
                "truncated": item.truncated,
                "score": item.score,
                "tokens": item.tokens,
                "text": item.text,
            }
            for item in context.evidence
        ],
        "report": {
            "included": [
                {"chunk_id": str(i.chunk_id), "evidence_id": i.evidence_id, "tokens": i.tokens}
                for i in report.included
            ],
            "dropped": [
                {"chunk_id": str(d.chunk_id), "reason": d.reason.value} for d in report.dropped
            ],
            "merged": [{"chunk_id": str(m.chunk_id), "into": str(m.into)} for m in report.merged],
            "truncated": [str(chunk) for chunk in report.truncated],
            "tokens_used": report.tokens_used,
            "budget": report.budget,
        },
    }


def describe_context(
    outcome: SearchOutcome,
    context: Context,
    *,
    system_tokens: int,
    question_tokens: int,
    full: bool,
) -> list[str]:
    budget = context.budget
    lines = [
        f"budget: window {budget.window} - system {system_tokens} - question {question_tokens}"
        f" - max_new {budget.max_new} - margin {budget.margin} = evidence {budget.evidence_budget};"
        f" used {context.tokens_used}",
    ]
    for item in context.evidence:
        where = f"{item.locator.source_name}/{item.locator.source_key}"
        headings = (
            f"  [{' > '.join(item.locator.heading_path)}]" if item.locator.heading_path else ""
        )
        cut = "  (truncated)" if item.truncated else ""
        lines.append(f"{item.evidence_id}  {item.tokens:>4} tok  {where}{headings}{cut}")
        body = " ".join(item.text.split())
        if not full and len(body) > _PREVIEW_CHARS:
            body = body[: _PREVIEW_CHARS - 1] + "…"
        lines.append(f"    {body}")
    for dropped in context.report.dropped:
        lines.append(f"dropped {dropped.reason.value}: {dropped.chunk_id}")
    for merged in context.report.merged:
        lines.append(f"merged {merged.chunk_id} into {merged.into}")
    for name in outcome.result.fallbacks:
        lines.append(f"note: {name}")
    return lines
