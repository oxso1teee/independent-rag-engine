"""A ready ``Context`` made of given passages, for tests and smoke scripts."""

from __future__ import annotations

from collections.abc import Sequence

from rag_engine.domain import (
    ChunkId,
    Context,
    ContextBudget,
    ContextReport,
    DocumentId,
    DocumentVersionId,
    Evidence,
    EvidenceLocator,
)


def evidence_context(
    texts: Sequence[str], *, document_id: DocumentId, version_id: DocumentVersionId
) -> Context:
    """Evidence ``E1..En`` over ``texts``; passage ``n`` comes from the source ``d<n>.md``."""
    items = []
    for number, text in enumerate(texts, start=1):
        chunk = ChunkId.derive(version_id, "ctx-fixture", number)
        items.append(
            Evidence(
                evidence_id=f"E{number}",
                chunk_id=chunk,
                matched=(chunk,),
                document_id=document_id,
                version_id=version_id,
                text=text,
                truncated=False,
                locator=EvidenceLocator("notes", f"d{number}.md", None, ("H",), None, None, 0, 10),
                score=0.5,
                tokens=len(text.split()),
            )
        )
    budget = ContextBudget(1000, 100, 20, 100, 100, 680)
    used = sum(item.tokens for item in items)
    report = ContextReport((), (), (), (), used, 680, "0" * 64)
    return Context(tuple(items), budget, used, report, "0" * 64)
