"""An answer as plain data: the JSON document and the stream events.

Shared by the command line and the HTTP API; both print and send exactly these fields.
"""

from __future__ import annotations

from rag_engine.domain import AskStatus, Citation, Evidence, EvidenceLocator
from rag_engine.generation.generator import CitationsEvent, DeltaEvent, EvidenceEvent, RetractEvent
from rag_engine.pipeline.service import FinalEvent, PipelineAnswer, PipelineEvent


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
