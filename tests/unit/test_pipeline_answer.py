"""PipelineAnswer makes a wrong combination of status, text and citations unrepresentable."""

from __future__ import annotations

import pytest

from rag_engine.domain import AskStatus, Citation, ErrorCode, RagError
from rag_engine.pipeline import PipelineAnswer, PipelineDiagnostics, PipelineError
from tests.factories import make_ids
from tests.unit.test_citations import BACKUPS, make_context

pytestmark = pytest.mark.unit

REQUEST_ID = make_ids().request_id()
CONTEXT = make_context(BACKUPS)
EVIDENCE = CONTEXT.evidence[0]
DIAG = PipelineDiagnostics(stage_ms={}, total_ms=0)
CITATION = Citation(
    "E1", EVIDENCE.chunk_id, EVIDENCE.document_id, EVIDENCE.version_id, EVIDENCE.locator, 1
)


def result(status: AskStatus, **changes: object) -> PipelineAnswer:
    base: dict[str, object] = {
        "request_id": REQUEST_ID,
        "status": status,
        "text": "",
        "citations": (),
        "sources": CONTEXT.evidence,
        "diagnostics": DIAG,
    }
    base.update(changes)
    return PipelineAnswer(**base)  # type: ignore[arg-type]


def test_a_well_formed_answer_is_accepted() -> None:
    answer = result(AskStatus.ANSWERED, text="Nightly [E1].", citations=(CITATION,))
    assert answer.citations[0].evidence_id == "E1"


@pytest.mark.parametrize(
    "changes",
    [
        {"text": "x", "citations": (CITATION,)},  # shorter than the threshold
        {"text": "Nightly.", "citations": ()},  # nothing cited
        {
            "text": "Nightly [E2].",
            "citations": (
                CITATION.__class__(
                    "E2",
                    EVIDENCE.chunk_id,
                    EVIDENCE.document_id,
                    EVIDENCE.version_id,
                    EVIDENCE.locator,
                    1,
                ),
            ),
        },  # cites a source that is not among the sources
        {"text": "Nightly [E1].", "citations": (CITATION,), "raw_text": "leftover"},
    ],
)
def test_a_malformed_answered_result_is_refused(changes: dict[str, object]) -> None:
    with pytest.raises(RagError) as info:
        result(AskStatus.ANSWERED, **changes)
    assert info.value.code is ErrorCode.E_INTEGRITY


@pytest.mark.parametrize(
    "status", [AskStatus.INSUFFICIENT_EVIDENCE, AskStatus.UNVERIFIED, AskStatus.ERROR]
)
def test_only_an_answer_may_carry_text_or_citations(status: AskStatus) -> None:
    error = PipelineError(ErrorCode.E_TIMEOUT, "slow", True, "generation")
    extra = {"error": error} if status is AskStatus.ERROR else {}
    result(status, **extra)
    with pytest.raises(RagError):
        result(status, text="An answer.", **extra)
    with pytest.raises(RagError):
        result(status, citations=(CITATION,), **extra)


def test_an_error_result_needs_an_error_and_nothing_else_may_have_one() -> None:
    with pytest.raises(RagError):
        result(AskStatus.ERROR)
    with pytest.raises(RagError):
        result(
            AskStatus.INSUFFICIENT_EVIDENCE,
            error=PipelineError(ErrorCode.E_TIMEOUT, "x", True, "s"),
        )


def test_only_an_unverified_result_keeps_the_rejected_text() -> None:
    assert result(AskStatus.UNVERIFIED, raw_text="[E9] nonsense").raw_text
    with pytest.raises(RagError):
        result(AskStatus.INSUFFICIENT_EVIDENCE, raw_text="hidden")
