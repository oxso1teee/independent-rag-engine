"""How a pipeline answer is shown: as a JSON document, as text, and as stream events."""

from __future__ import annotations

from collections.abc import Sequence

from rag_engine.domain import AskStatus, Evidence, EvidenceLocator
from rag_engine.pipeline import PipelineAnswer
from rag_engine.pipeline.presentation import answer_to_dict, event_to_dict

__all__ = ["answer_to_dict", "describe_answer", "event_to_dict"]

_PREVIEW = 160


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
