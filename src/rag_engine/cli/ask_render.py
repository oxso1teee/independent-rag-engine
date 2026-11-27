"""How a pipeline answer is shown: as a JSON document, as text, and as stream events."""

from __future__ import annotations

from collections.abc import Sequence

from rag_engine.domain import AskStatus, Citation, Evidence, EvidenceLocator
from rag_engine.generation.generator import CitationsEvent, DeltaEvent, EvidenceEvent, RetractEvent
from rag_engine.pipeline import FinalEvent, PipelineAnswer, PipelineEvent

_PREVIEW = 160


def locator_to_dict(locator: EvidenceLocator) -> dict[str, object]:
    return {
        "source": locator.source_name,
        "path": locator.source_key,
        "title": locator.title,
        "heading_path": list(locator.heading_path),
        "pages": None if locator.page_start is None else [locator.page_start, locator.page_end],
        "char_start": locator.char_start,
        "char_end": locator.char_end,
    }


def citation_to_dict(citation: Citation) -> dict[str, object]:
    return {
        "evidence_id": citation.evidence_id,
        "chunk_id": str(citation.chunk_id),
        "document_id": str(citation.document_id),
        "version_id": str(citation.version_id),
        "uses": citation.uses,
        **locator_to_dict(citation.locator),
    }


def source_to_dict(evidence: Evidence, *, cited: bool, full: bool = False) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "chunk_id": str(evidence.chunk_id),
        "cited": cited,
        "truncated": evidence.truncated,
        "tokens": evidence.tokens,
        **locator_to_dict(evidence.locator),
        **({"text": evidence.text} if full else {}),
    }


def answer_to_dict(
    answer: PipelineAnswer, *, sources: bool = False, show_unverified: bool = False
) -> dict[str, object]:
    d = answer.diagnostics
    cited = {c.evidence_id for c in answer.citations}
    document: dict[str, object] = {
        "request_id": str(answer.request_id),
        "status": answer.status.value,
        "text": answer.text,
        "reasons": list(answer.reasons),
        "citations": [citation_to_dict(c) for c in answer.citations],
        "error": None
        if answer.error is None
        else {
            "code": answer.error.code.value,
            "message": answer.error.message,
            "retryable": answer.error.retryable,
            "stage": answer.error.stage,
        },
        "diagnostics": {
            "total_ms": d.total_ms,
            "stage_ms": dict(d.stage_ms),
            "llm_called": d.llm_called,
            "attempts": d.attempts,
            "usage": None
            if d.usage is None
            else {
                "prompt_tokens": d.usage.prompt_tokens,
                "completion_tokens": d.usage.completion_tokens,
            },
            "gate": None
            if d.gate is None
            else {
                "sufficient": d.gate.sufficient,
                "calibrated": d.gate.calibrated,
                "reasons": list(d.gate.reasons),
            },
            "context": None
            if d.context is None
            else {
                "evidence": len(d.context.included),
                "tokens_used": d.context.tokens_used,
                "budget": d.context.budget,
                "truncated": len(d.context.truncated),
                "dropped": {
                    reason: sum(1 for item in d.context.dropped if item.reason.value == reason)
                    for reason in sorted({item.reason.value for item in d.context.dropped})
                },
                "merged": len(d.context.merged),
                "context_hash": d.context.context_hash,
            },
            "fallbacks": list(d.fallbacks),
            "warnings": list(d.warnings),
            "prompt_sha256": d.prompt_sha256,
            "template_id": d.template_id,
        },
    }
    if sources:
        document["sources"] = [
            source_to_dict(e, cited=e.evidence_id in cited) for e in answer.sources
        ]
    if show_unverified and answer.status is AskStatus.UNVERIFIED:
        document["unverified_text"] = answer.raw_text
    return document


def event_to_dict(event: PipelineEvent) -> dict[str, object]:
    """One stream event as a JSON object (one per line in ``--json`` streaming)."""
    if isinstance(event, EvidenceEvent):
        return {
            "event": "evidence",
            "sources": [source_to_dict(e, cited=False) for e in event.evidence],
        }
    if isinstance(event, DeltaEvent):
        return {"event": "delta", "text": event.text, "provisional": event.provisional}
    if isinstance(event, RetractEvent):
        return {"event": "retract", "code": event.code.value, "reason": event.reason}
    if isinstance(event, CitationsEvent):
        return {"event": "citations", "citations": [citation_to_dict(c) for c in event.citations]}
    assert isinstance(event, FinalEvent)  # noqa: S101 - the union is closed
    return {"event": "final", **answer_to_dict(event.answer, sources=True)}


def _where(locator: EvidenceLocator) -> str:
    parts = [f"{locator.source_name}/{locator.source_key}"]
    if locator.heading_path:
        parts.append(" > ".join(locator.heading_path))
    if locator.page_start is not None:
        pages = (
            f"p. {locator.page_start}"
            if locator.page_start == locator.page_end
            else f"pp. {locator.page_start}-{locator.page_end}"
        )
        parts.append(pages)
    return "  ".join(parts)


def describe_answer(
    answer: PipelineAnswer, *, sources: bool = False, show_unverified: bool = False
) -> list[str]:
    lines: list[str] = []
    match answer.status:
        case AskStatus.ANSWERED:
            lines.append(answer.text)
            lines.append("")
            lines.append("Sources:")
            for citation in answer.citations:
                lines.append(f"  [{citation.evidence_id}] {_where(citation.locator)}")
        case AskStatus.INSUFFICIENT_EVIDENCE:
            lines.append("The documents do not contain enough information to answer.")
            lines.append(f"({', '.join(answer.reasons)})")
        case AskStatus.UNVERIFIED:
            lines.append("No verified answer: the model's citations did not hold up.")
            lines.append(f"({', '.join(answer.reasons)})")
            if show_unverified:
                lines.append("Unverified text, do not rely on it:")
                lines.append(answer.raw_text)
        case AskStatus.ERROR:
            error = answer.error
            assert error is not None  # noqa: S101 - an error result always has one
            lines.append(f"error [{error.code.value}] in {error.stage}: {error.message}")
    if sources:
        lines.extend(_source_lines(answer.sources, {c.evidence_id for c in answer.citations}))
    for warning in answer.diagnostics.warnings:
        lines.append(f"warning: {warning}")
    return lines


def _source_lines(sources: Sequence[Evidence], cited: set[str]) -> list[str]:
    lines = ["", "Evidence shown to the model:"]
    for item in sources:
        mark = "*" if item.evidence_id in cited else " "
        text = " ".join(item.text.split())
        text = text if len(text) <= _PREVIEW else text[: _PREVIEW - 1] + "…"
        cut = " (cut)" if item.truncated else ""
        lines.append(f" {mark}[{item.evidence_id}] {_where(item.locator)}{cut}")
        lines.append(f"     {text}")
    if not sources:
        lines.append("  none")
    return lines
