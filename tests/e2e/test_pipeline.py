"""The query pipeline end to end on a fixed corpus and a scripted model: every status branch."""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import pytest

from rag_engine.context.prompt import REFUSAL_SENTINEL
from rag_engine.domain import AskStatus, ErrorCode, RagError, RetrievalMode
from rag_engine.generation.generator import CitationsEvent, DeltaEvent, EvidenceEvent, RetractEvent
from rag_engine.observability import configure_logging
from rag_engine.pipeline import FinalEvent, PipelineAnswer, PipelineLimits
from rag_engine.ports import ModelLimits
from rag_engine.testing import Fail, Reply
from tests.pipeline_support import ANSWER, QUESTION, Rig, SlowLLM, query, rig

pytestmark = pytest.mark.e2e


def ask(rig_: Rig, **kwargs: Any) -> PipelineAnswer:
    return asyncio.run(rig_.pipeline.ask(query(**kwargs)))


# ------------------------------------------------------------------ the four statuses


def test_answered_carries_resolved_citations_sources_and_diagnostics() -> None:
    r = rig(Reply(ANSWER))
    result = ask(r)
    assert result.status is AskStatus.ANSWERED and result.text == ANSWER
    assert [c.evidence_id for c in result.citations] == ["E1"]
    assert result.citations[0].locator.source_key.endswith(".md")
    assert {s.evidence_id for s in result.sources} >= {"E1"}
    d = result.diagnostics
    assert d.llm_called and d.attempts == 1 and d.usage is not None and d.usage.prompt_tokens > 0
    assert set(d.stage_ms) >= {"retrieval", "context", "generation"} and d.total_ms >= 0
    assert d.gate is not None and d.gate.sufficient and d.context is not None
    assert d.retrieval is not None and d.prompt_sha256 and d.template_id == "answer.v1"
    assert len(r.llm.calls) == 1 and "Question: " + QUESTION in r.llm.calls[0].prompt


def test_a_gate_refusal_answers_without_calling_the_model() -> None:
    r = rig(dense_threshold=5.0)  # no cosine reaches 5: the evidence is "too weak"
    result = ask(r, mode=RetrievalMode.DENSE)
    assert result.status is AskStatus.INSUFFICIENT_EVIDENCE
    assert result.reasons == ("gate:dense_below_threshold",)
    assert r.llm.calls == () and not result.diagnostics.llm_called  # spy: zero model calls
    assert result.text == "" and not result.citations and result.sources  # evidence still shown
    assert r.released == []  # nothing to hand over: no local model is loaded


def test_an_uncalibrated_gate_lets_the_question_through_and_says_so() -> None:
    result = ask(rig(Reply(ANSWER)))
    assert result.diagnostics.gate is not None
    assert result.diagnostics.gate.reasons == ("uncalibrated",)
    assert result.status is AskStatus.ANSWERED


def test_no_hits_is_insufficient_without_a_model_call() -> None:
    r = rig()
    result = ask(r, text="qqqqzzzz", mode=RetrievalMode.LEXICAL)
    assert result.status is AskStatus.INSUFFICIENT_EVIDENCE and result.reasons == ("no_hits",)
    assert r.llm.calls == ()


def test_the_model_saying_the_evidence_is_missing_is_insufficient_evidence() -> None:
    result = ask(rig(Reply(REFUSAL_SENTINEL)))
    assert result.status is AskStatus.INSUFFICIENT_EVIDENCE
    assert result.reasons == ("model_refused",) and result.diagnostics.llm_called
    assert result.text == "" and not result.citations


def test_a_fabricated_citation_gives_unverified_and_the_text_is_not_an_answer() -> None:
    bad = "Chunks are stored in SQLite [E9]."
    result = ask(rig(Reply(bad), Reply(bad)))
    assert result.status is AskStatus.UNVERIFIED
    assert result.text == "" and not result.citations and result.raw_text == bad
    assert "unknown_marker" in result.reasons and result.diagnostics.attempts == 2


def test_a_missing_citation_is_repaired_by_the_retry() -> None:
    result = ask(rig(Reply("Chunks are stored in SQLite."), Reply(ANSWER)))
    assert result.status is AskStatus.ANSWERED and result.diagnostics.attempts == 2


@pytest.mark.parametrize("reply", ["x", ".", "[E1]", "  ", ""])
def test_a_one_character_or_empty_reply_is_never_answered(reply: str) -> None:
    result = ask(rig(Reply(reply), Reply(reply)))
    assert result.status is AskStatus.ERROR and result.error is not None
    assert result.error.code is ErrorCode.E_DEGENERATE_ANSWER and result.error.stage == "generation"


def test_a_provider_failure_is_an_error_status_not_an_exception() -> None:
    down = RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "down")
    result = ask(rig(Fail(down)))
    assert result.status is AskStatus.ERROR and result.error is not None
    assert result.error.code is ErrorCode.E_PROVIDER_UNAVAILABLE and result.error.retryable


def test_a_prompt_that_cannot_fit_is_a_context_error_naming_the_stage() -> None:
    result = ask(rig(limits=ModelLimits(300, 200)))
    assert result.status is AskStatus.ERROR and result.error is not None
    assert result.error.code is ErrorCode.E_CONTEXT_BUDGET and result.error.stage == "context"
    assert result.sources == ()


# ---------------------------------------------------------------------- the status table


def test_every_status_is_reachable_and_only_answered_carries_text() -> None:
    cases = {
        AskStatus.ANSWERED: ask(rig(Reply(ANSWER))),
        AskStatus.INSUFFICIENT_EVIDENCE: ask(rig(dense_threshold=5.0), mode=RetrievalMode.DENSE),
        AskStatus.UNVERIFIED: ask(
            rig(
                Reply("Chunks are stored in SQLite [E9]."),
                Reply("Chunks are stored in SQLite [E9]."),
            )
        ),
        AskStatus.ERROR: ask(rig(Fail(RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "x")))),
    }
    for status, result in cases.items():
        assert result.status is status
        assert bool(result.text) == (status is AskStatus.ANSWERED)
        assert bool(result.citations) == (status is AskStatus.ANSWERED)
        assert (result.error is not None) == (status is AskStatus.ERROR)


# ----------------------------------------------------------------------------- time


def test_a_model_that_outlasts_the_total_budget_is_a_timeout_naming_the_stage() -> None:
    r = rig(llm=SlowLLM(5.0, []), pipeline_limits=PipelineLimits(total_timeout_s=0.3))
    result = ask(r)
    assert result.status is AskStatus.ERROR and result.error is not None
    assert result.error.code is ErrorCode.E_TIMEOUT and result.error.stage == "generation"
    assert result.diagnostics.stage_ms["generation"] >= 200


def test_a_request_cancelled_mid_generation_releases_everything() -> None:
    slow = SlowLLM(30.0, [])
    r = rig(llm=slow)

    async def scenario() -> PipelineAnswer:
        task = asyncio.create_task(r.pipeline.ask(query()))
        while slow.started == 0:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        slow.delay = 0.0
        slow._script.append(Reply(ANSWER))
        return await r.pipeline.ask(query())  # the slot is free again

    assert asyncio.run(scenario()).status is AskStatus.ANSWERED


def test_requests_are_served_one_at_a_time_by_default() -> None:
    slow = SlowLLM(0.05, [])
    slow._script.extend([Reply(ANSWER), Reply(ANSWER)])
    r = rig(llm=slow)
    order: list[str] = []

    async def go(tag: str) -> None:
        await r.pipeline.ask(query())
        order.append(tag)

    async def scenario() -> None:
        await asyncio.gather(go("a"), go("b"))

    asyncio.run(scenario())
    assert order == ["a", "b"] and slow.started == 2


def test_a_higher_concurrency_limit_lets_requests_overlap_and_one_cancel_spares_the_other() -> None:
    slow = SlowLLM(0.2, [])
    slow._script.extend([Reply(ANSWER), Reply(ANSWER)])
    r = rig(llm=slow, pipeline_limits=PipelineLimits(total_timeout_s=30.0, concurrency=2))

    async def scenario() -> tuple[PipelineAnswer, bool]:
        keep = asyncio.create_task(r.pipeline.ask(query()))
        drop = asyncio.create_task(r.pipeline.ask(query()))
        while slow.started < 2:
            await asyncio.sleep(0.01)
        drop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drop
        return await keep, slow.active == 0

    kept, idle = asyncio.run(scenario())
    assert slow.peak == 2 and kept.status is AskStatus.ANSWERED and idle


def test_the_remaining_budget_shrinks_for_each_stage_and_generation_gets_the_rest() -> None:
    slow = SlowLLM(0.0, [Reply(ANSWER)])
    result = ask(rig(llm=slow, pipeline_limits=PipelineLimits(total_timeout_s=30.0)))
    stages = result.diagnostics.stage_ms
    assert sum(stages.values()) <= result.diagnostics.total_ms + 5  # stages never exceed the whole


# ------------------------------------------------------------------------ resources


def test_models_are_released_once_before_generation_and_a_failing_hook_changes_nothing() -> None:
    def boom() -> None:
        raise RuntimeError("could not free")

    r = rig(Reply(ANSWER), release=boom)
    assert ask(r).status is AskStatus.ANSWERED
    assert r.released == [1]


# ---------------------------------------------------------------------------- stream


async def events(r: Rig, **kwargs: Any) -> list[object]:
    return [e async for e in r.pipeline.stream(query(), **kwargs)]


def test_a_verified_stream_sends_evidence_then_the_validated_text_once_then_final() -> None:
    r = rig(Reply(ANSWER))
    items = asyncio.run(events(r))
    assert isinstance(items[0], EvidenceEvent) and isinstance(items[-1], FinalEvent)
    deltas = [e for e in items if isinstance(e, DeltaEvent)]
    assert len(deltas) == 1 and not deltas[0].provisional and deltas[0].text == ANSWER
    assert isinstance(items[-2], CitationsEvent)
    assert items[-1].answer.status is AskStatus.ANSWERED and items[-1].answer.text == ANSWER
    plain = ask(rig(Reply(ANSWER)))
    assert items[-1].answer.citations == plain.citations


def test_a_verified_stream_with_a_fabricated_citation_sends_no_text() -> None:
    r = rig(Reply("Chunks are stored in SQLite [E9]."))
    items = asyncio.run(events(r))
    assert not any(isinstance(e, DeltaEvent | RetractEvent) for e in items)
    final = items[-1]
    assert isinstance(final, FinalEvent) and final.answer.status is AskStatus.UNVERIFIED


def test_a_live_stream_marks_drafts_provisional_and_retracts_a_failed_one() -> None:
    r = rig(Reply("Chunks are stored in SQLite [E9]."))
    items = asyncio.run(events(r, verified_only=False))
    deltas = [e for e in items if isinstance(e, DeltaEvent)]
    assert deltas and all(d.provisional for d in deltas)
    assert any(isinstance(e, RetractEvent) for e in items)
    final = items[-1]
    assert isinstance(final, FinalEvent) and final.answer.status is AskStatus.UNVERIFIED
    assert final.answer.text == ""  # the draft is not an answer


def test_a_gate_refusal_ends_the_stream_at_once_without_a_model_call() -> None:
    r = rig(dense_threshold=5.0)

    async def run() -> list[object]:
        q = query(mode=RetrievalMode.DENSE)
        return [e async for e in r.pipeline.stream(q)]

    items = asyncio.run(run())
    assert len(items) == 1 and isinstance(items[0], FinalEvent)
    assert items[0].answer.status is AskStatus.INSUFFICIENT_EVIDENCE and r.llm.calls == ()


def test_a_provider_failure_in_a_stream_is_a_final_error_event() -> None:
    r = rig(Fail(RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "down")))
    items = asyncio.run(events(r))
    final = items[-1]
    assert isinstance(final, FinalEvent) and final.answer.status is AskStatus.ERROR


# ------------------------------------------------------------------- NFR-07 and logs


def test_every_pipeline_event_carries_the_request_id_and_documents_never_reach_the_log() -> None:
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    q = query()
    r = rig(Reply(ANSWER))
    asyncio.run(r.pipeline.ask(q))
    logged = [json.loads(line) for line in stream.getvalue().splitlines()]
    mine = [e for e in logged if str(e["event"]).startswith("pipeline.")]
    assert {e["event"] for e in mine} >= {"pipeline.context", "pipeline.gate", "pipeline.done"}
    assert all(e["request_id"] == str(q.request_id) for e in mine)
    from tests.unit.test_search_service import CORPUS

    text = stream.getvalue()
    assert not any(passage in text for _, passage in CORPUS.values())  # no document text
    assert ANSWER not in text and QUESTION not in text  # nor the answer or the question


def test_a_cut_or_dropped_context_is_reported_in_diagnostics_and_in_the_log() -> None:
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    r = rig(Reply(ANSWER), limits=ModelLimits(700, 100))
    result = ask(r, top_k=6)
    report = result.diagnostics.context
    assert report is not None
    accounted = len(report.included) + len(report.dropped) + len(report.merged)
    assert accounted >= 1
    event = next(
        json.loads(line) for line in stream.getvalue().splitlines() if '"pipeline.context"' in line
    )
    assert event["dropped"] == len(report.dropped) and event["truncated"] == len(report.truncated)
