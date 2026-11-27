"""AnswerGenerator: grounded answers, refusals, retries, truncation and streaming."""

from __future__ import annotations

import asyncio

import pytest

from rag_engine.context.prompt import REFUSAL_SENTINEL
from rag_engine.domain import AnswerStatus, CitationProblem, ErrorCode, RagError
from rag_engine.generation.generator import (
    AnswerEvent,
    AnswerGenerator,
    CitationsEvent,
    DeltaEvent,
    DoneEvent,
    EvidenceEvent,
    GenerationPolicy,
    RetractEvent,
)
from rag_engine.ports import Delta, FinishReason, ModelLimits
from rag_engine.testing import Fail, Reply, ScriptedLLM
from tests.factories import make_ids
from tests.unit.test_citations import BACKUPS, PRINTERS, make_context

pytestmark = pytest.mark.unit

REQUEST_ID = make_ids().request_id()
GOOD = "Backups run every night at 02:30 UTC [E1]."
CONTEXT = make_context(BACKUPS, PRINTERS)


def generator(
    *steps: Reply | Fail, policy: GenerationPolicy | None = None, **limits: int
) -> tuple[AnswerGenerator, ScriptedLLM]:
    llm = ScriptedLLM(
        list(steps),
        limits=ModelLimits(limits.get("window", 8192), limits.get("max_new", 1024)),
    )
    return AnswerGenerator(llm, policy=policy or GenerationPolicy()), llm


def answer(gen: AnswerGenerator, question: str = "When do backups run?"):  # type: ignore[no-untyped-def]
    return asyncio.run(gen.answer(REQUEST_ID, question, CONTEXT))


def test_a_grounded_answer_carries_resolved_citations_and_accounting() -> None:
    gen, llm = generator(Reply(GOOD))
    result = answer(gen)
    assert result.status is AnswerStatus.ANSWERED and result.text == GOOD
    assert [c.evidence_id for c in result.citations] == ["E1"]
    assert result.citations[0].locator.source_key == "d1.md" and result.citations[0].uses == 1
    assert result.check is not None and result.check.ok and result.attempts == 1
    assert result.usage.prompt_tokens > 0 and result.usage.completion_tokens > 0
    assert len(result.prompt_sha256) == 64 and result.template_id == "answer.v1"
    request = llm.calls[0]
    assert request.params.temperature == 0.0 and "Question: When do backups run?" in request.prompt


def test_the_sentinel_is_a_refusal_without_citations() -> None:
    gen, _ = generator(Reply(f"  {REFUSAL_SENTINEL}\n"))
    result = answer(gen)
    assert result.status is AnswerStatus.REFUSED and result.text == "" and not result.citations


def test_an_empty_context_refuses_without_calling_the_model() -> None:
    gen, llm = generator()
    result = asyncio.run(gen.answer(REQUEST_ID, "Q?", make_context()))
    assert result.status is AnswerStatus.REFUSED and "no_evidence" in result.warnings
    assert llm.calls == ()


def test_an_empty_reply_is_retried_once_with_a_warmer_temperature() -> None:
    gen, llm = generator(Reply("   "), Reply(GOOD))
    result = answer(gen)
    assert result.attempts == 2 and result.status is AnswerStatus.ANSWERED
    assert [c.params.temperature for c in llm.calls] == [0.0, 0.3]
    assert result.usage.completion_tokens >= 7  # usage is summed over attempts


def test_a_degenerate_answer_twice_is_an_error() -> None:
    gen, _ = generator(Reply(""), Reply("the the the the the the the the the the the the the"))
    with pytest.raises(RagError) as info:
        answer(gen)
    assert info.value.code is ErrorCode.E_DEGENERATE_ANSWER


def test_a_sentinel_mixed_with_prose_is_degenerate() -> None:
    gen, _ = generator(Reply(f"{REFUSAL_SENTINEL} but backups run nightly [E1]."), Reply(GOOD))
    assert answer(gen).attempts == 2


def test_an_invented_marker_is_retried_and_then_fails_loudly() -> None:
    gen, _ = generator(Reply("Backups run nightly [E5]."), Reply("Backups run nightly [E9]."))
    with pytest.raises(RagError) as info:
        answer(gen)
    assert info.value.code is ErrorCode.E_CITATION_INVALID
    assert info.value.context["problems"] == ("unknown_marker", "no_citation")
    assert info.value.context["unknown"] == ("E9",)


def test_a_missing_citation_is_repaired_by_the_retry() -> None:
    gen, _ = generator(Reply("Backups run every night."), Reply(GOOD))
    result = answer(gen)
    assert result.attempts == 2 and [c.evidence_id for c in result.citations] == ["E1"]


def test_a_low_overlap_sentence_is_reported_not_fatal_by_default() -> None:
    text = "The printers are encrypted with a hardware token [E1]."
    result = answer(generator(Reply(text))[0])
    assert result.status is AnswerStatus.ANSWERED
    assert result.check is not None and CitationProblem.LOW_OVERLAP in result.check.problems


def test_strict_overlap_treats_an_ungrounded_sentence_as_invalid() -> None:
    text = "The printers are encrypted with a hardware token [E1]."
    gen, _ = generator(Reply(text), Reply(text), policy=GenerationPolicy(strict_overlap=True))
    with pytest.raises(RagError) as info:
        answer(gen)
    assert info.value.code is ErrorCode.E_CITATION_INVALID


def test_a_prompt_that_cannot_fit_the_window_is_refused_before_the_call() -> None:
    gen, llm = generator(Reply(GOOD), window=60, max_new=20)
    with pytest.raises(RagError) as info:
        answer(gen)
    assert info.value.code is ErrorCode.E_CONTEXT_BUDGET and llm.calls == ()


def test_a_silently_truncated_prompt_is_detected_from_the_reported_tokens() -> None:
    gen, _ = generator(Reply(GOOD, prompt_tokens=8191), window=8192, max_new=512)
    with pytest.raises(RagError) as info:
        answer(gen)
    assert info.value.code is ErrorCode.E_CONTEXT_TRUNCATED


def test_a_large_gap_between_counted_and_reported_tokens_is_a_warning() -> None:
    gen, llm = generator(Reply(GOOD))
    counted = llm.count_tokens(CONTEXT.evidence[0].text) * 40
    gen, _ = generator(Reply(GOOD, prompt_tokens=counted))
    assert "token_count_drift" in answer(gen).warnings


def test_an_answer_cut_by_the_length_limit_is_flagged() -> None:
    gen, _ = generator(Reply(GOOD, finish_reason=FinishReason.LENGTH))
    result = answer(gen)
    assert "answer_truncated" in result.warnings and result.status is AnswerStatus.ANSWERED


def test_provider_errors_propagate_unchanged() -> None:
    error = RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "down")
    gen, _ = generator(Fail(error))
    with pytest.raises(RagError) as info:
        answer(gen)
    assert info.value is error


# --------------------------------------------------------------------------- streaming


async def collect(gen: AnswerGenerator, *, verified_only: bool = False) -> list[AnswerEvent]:
    return [
        e
        async for e in gen.stream(
            REQUEST_ID, "When do backups run?", CONTEXT, verified_only=verified_only
        )
    ]


async def collect_until_error(
    gen: AnswerGenerator, *, verified_only: bool = False
) -> tuple[list[AnswerEvent], RagError]:
    seen: list[AnswerEvent] = []
    try:
        async for event in gen.stream(
            REQUEST_ID, "When do backups run?", CONTEXT, verified_only=verified_only
        ):
            seen.append(event)
    except RagError as error:
        return seen, error
    raise AssertionError("the stream was expected to fail")


def test_a_stream_orders_evidence_deltas_citations_and_done() -> None:
    events = asyncio.run(collect(generator(Reply(GOOD))[0]))
    kinds = [type(e) for e in events]
    assert kinds[0] is EvidenceEvent and kinds[-1] is DoneEvent and kinds[-2] is CitationsEvent
    assert all(k is DeltaEvent for k in kinds[1:-2]) and len(kinds) > 4
    assert len(events[0].evidence) == 2  # type: ignore[union-attr]


def test_live_deltas_are_marked_provisional() -> None:
    events = asyncio.run(collect(generator(Reply(GOOD))[0]))
    deltas = [e for e in events if isinstance(e, DeltaEvent)]
    assert deltas and all(d.provisional for d in deltas)


def test_the_stream_equals_the_non_streamed_answer() -> None:
    events = asyncio.run(collect(generator(Reply(GOOD))[0]))
    streamed = "".join(e.text for e in events if isinstance(e, DeltaEvent))
    done = events[-1]
    assert isinstance(done, DoneEvent)
    plain = answer(generator(Reply(GOOD))[0])
    assert streamed == plain.text == done.answer.text
    assert done.answer.citations == plain.citations and "usage_estimated" in done.answer.warnings


def test_a_refusal_streams_no_text() -> None:
    events = asyncio.run(collect(generator(Reply(REFUSAL_SENTINEL))[0]))
    assert not any(isinstance(e, DeltaEvent) for e in events)
    done = events[-1]
    assert isinstance(done, DoneEvent) and done.answer.status is AnswerStatus.REFUSED


def test_a_live_stream_retracts_text_that_failed_validation_before_raising() -> None:
    gen, _ = generator(Reply("Backups run nightly [E5]."))
    seen, error = asyncio.run(collect_until_error(gen))
    assert error.code is ErrorCode.E_CITATION_INVALID
    assert any(isinstance(e, DeltaEvent) for e in seen)  # the draft did reach the client ...
    assert isinstance(seen[-1], RetractEvent)  # ... and was withdrawn before the error
    assert seen[-1].code is ErrorCode.E_CITATION_INVALID
    assert not any(isinstance(e, DoneEvent | CitationsEvent) for e in seen)


def test_a_provider_failure_after_text_was_sent_also_retracts() -> None:
    class Dies(ScriptedLLM):
        async def stream(self, request):  # type: ignore[no-untyped-def]
            yield Delta("Backups run ")
            raise RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "gone")

    gen = AnswerGenerator(Dies([]))
    seen, error = asyncio.run(collect_until_error(gen))
    assert error.code is ErrorCode.E_PROVIDER_UNAVAILABLE and isinstance(seen[-1], RetractEvent)


def test_verified_only_never_sends_unvalidated_text() -> None:
    gen, _ = generator(Reply("Backups run nightly [E5]."))
    seen, error = asyncio.run(collect_until_error(gen, verified_only=True))
    assert error.code is ErrorCode.E_CITATION_INVALID
    assert [type(e) for e in seen] == [EvidenceEvent]  # nothing but the evidence list


def test_verified_only_sends_the_validated_text_once_as_final() -> None:
    events = asyncio.run(collect(generator(Reply(GOOD))[0], verified_only=True))
    deltas = [e for e in events if isinstance(e, DeltaEvent)]
    assert len(deltas) == 1 and not deltas[0].provisional and deltas[0].text == GOOD
    assert isinstance(events[-1], DoneEvent) and events[-1].answer.text == GOOD


# ------------------------------------------------------------ acceptance scenarios (P7.5, P7.6)


@pytest.mark.parametrize("window", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("gap", [0, 1])
def test_a_prompt_cut_to_the_window_is_always_caught(window: int, gap: int) -> None:
    gen, _ = generator(Reply(GOOD, prompt_tokens=window - gap), window=window, max_new=256)
    with pytest.raises(RagError) as info:
        answer(gen)
    assert info.value.code is ErrorCode.E_CONTEXT_TRUNCATED


@pytest.mark.parametrize(
    "reply",
    [
        Reply("x"),
        Reply(""),
        Reply(" \n"),
        Reply("?"),
        Reply("[E1]"),
        Reply("-", finish_reason=FinishReason.LENGTH),
        Reply("", finish_reason=FinishReason.LENGTH),
    ],
)
def test_no_answer_shorter_than_the_threshold_is_ever_returned(reply: Reply) -> None:
    gen, _ = generator(reply, reply)
    with pytest.raises(RagError) as info:
        answer(gen)
    assert info.value.code is ErrorCode.E_DEGENERATE_ANSWER


def test_the_threshold_counts_letters_not_markers_and_is_configurable() -> None:
    assert answer(generator(Reply("No [E1]."))[0]).status is AnswerStatus.ANSWERED  # 2 letters
    gen, _ = generator(
        Reply("No [E1]."), Reply("No [E1]."), policy=GenerationPolicy(min_answer_chars=5)
    )
    with pytest.raises(RagError):
        answer(gen)


@pytest.mark.parametrize("text", [GOOD, "Backups [E1]\n\nrun every night [E1]", " lead " + GOOD])
def test_stream_and_non_stream_agree_for_any_scripted_text(text: str) -> None:
    plain = answer(generator(Reply(text))[0])
    events = asyncio.run(collect(generator(Reply(text))[0]))
    streamed = "".join(e.text for e in events if isinstance(e, DeltaEvent))
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.answer.text == plain.text == streamed.strip()
    assert done.answer.citations == plain.citations and done.answer.status == plain.status
