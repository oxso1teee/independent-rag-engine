"""Phase 7 flow on a fixed context: prompt, scripted model, resilience, citations, refusal."""

from __future__ import annotations

import asyncio

import pytest

from rag_engine.context.prompt import REFUSAL_SENTINEL
from rag_engine.domain import AnswerStatus, ErrorCode, RagError
from rag_engine.generation.generator import AnswerGenerator, DeltaEvent, DoneEvent
from rag_engine.generation.resilience import ResiliencePolicy, ResilientLLM
from rag_engine.testing import Fail, Reply, ScriptedLLM
from tests.factories import make_ids
from tests.unit.test_citations import BACKUPS, PRINTERS, make_context

pytestmark = pytest.mark.e2e

CONTEXT = make_context(BACKUPS, PRINTERS)
REQUEST_ID = make_ids().request_id()


def build(*steps: Reply | Fail) -> tuple[AnswerGenerator, ScriptedLLM]:
    scripted = ScriptedLLM(list(steps))

    async def no_wait(_: float) -> None:
        return None

    llm = ResilientLLM(scripted, ResiliencePolicy(backoff_s=0.0), sleep=no_wait)
    return AnswerGenerator(llm), scripted


def test_a_question_with_an_answer_is_answered_with_resolvable_citations() -> None:
    gen, scripted = build(Reply("Backups run every night at 02:30 UTC [E1]."))
    answer = asyncio.run(gen.answer(REQUEST_ID, "When do backups run?", CONTEXT))
    assert answer.status is AnswerStatus.ANSWERED
    assert {c.evidence_id for c in answer.citations} <= {e.evidence_id for e in CONTEXT.evidence}
    assert answer.citations[0].locator.source_key == "d1.md"
    sent = scripted.calls[0]
    assert BACKUPS in sent.prompt and PRINTERS in sent.prompt  # the model saw the evidence


def test_a_question_without_an_answer_is_refused_with_no_citations() -> None:
    gen, _ = build(Reply(REFUSAL_SENTINEL))
    answer = asyncio.run(gen.answer(REQUEST_ID, "What is the wifi password?", CONTEXT))
    assert answer.status is AnswerStatus.REFUSED and answer.text == "" and not answer.citations


def test_the_model_cannot_cite_a_source_that_was_not_provided() -> None:
    gen, _ = build(Reply("Backups run nightly [E4]."), Reply("Backups run nightly [E4]."))
    with pytest.raises(RagError) as info:
        asyncio.run(gen.answer(REQUEST_ID, "When do backups run?", CONTEXT))
    assert info.value.code is ErrorCode.E_CITATION_INVALID


def test_streaming_a_verified_answer_matches_the_plain_answer() -> None:
    text = "Printers are on the third floor [E2]."

    async def run() -> tuple[str, str]:
        gen, _ = build(Reply(text))
        events = [
            e
            async for e in gen.stream(
                REQUEST_ID, "Where are printers?", CONTEXT, verified_only=True
            )
        ]
        done = events[-1]
        assert isinstance(done, DoneEvent)
        return "".join(e.text for e in events if isinstance(e, DeltaEvent)), done.answer.text

    streamed, final = asyncio.run(run())
    assert streamed == final == text
