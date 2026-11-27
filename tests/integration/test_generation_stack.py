"""The generator on top of the resilience wrapper and a scripted provider, end to end in memory."""

from __future__ import annotations

import asyncio

import pytest

from rag_engine.domain import AnswerStatus, ErrorCode, RagError
from rag_engine.generation.generator import AnswerGenerator
from rag_engine.generation.resilience import ResiliencePolicy, ResilientLLM
from rag_engine.testing import Fail, Reply, ScriptedLLM
from tests.factories import make_ids
from tests.unit.test_citations import BACKUPS, PRINTERS, make_context

pytestmark = pytest.mark.integration

REQUEST_ID = make_ids().request_id()
CONTEXT = make_context(BACKUPS, PRINTERS)


def stack(*steps: Reply | Fail) -> tuple[AnswerGenerator, ScriptedLLM]:
    scripted = ScriptedLLM(list(steps))

    async def no_wait(_: float) -> None:
        return None

    llm = ResilientLLM(scripted, ResiliencePolicy(backoff_s=0.0), sleep=no_wait)
    return AnswerGenerator(llm), scripted


def test_a_transient_outage_is_absorbed_and_the_answer_is_still_cited() -> None:
    gen, scripted = stack(
        Fail(RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "starting")),
        Reply("Printers are on the third floor [E2]."),
    )
    result = asyncio.run(gen.answer(REQUEST_ID, "Where are the printers?", CONTEXT))
    assert result.status is AnswerStatus.ANSWERED and result.citations[0].evidence_id == "E2"
    assert len(scripted.calls) == 2


def test_an_outage_that_persists_surfaces_the_provider_error() -> None:
    down = RagError(ErrorCode.E_PROVIDER_UNAVAILABLE, "down")
    gen, _ = stack(Fail(down), Fail(down), Fail(down))
    with pytest.raises(RagError) as info:
        asyncio.run(gen.answer(REQUEST_ID, "Where?", CONTEXT))
    assert info.value.code is ErrorCode.E_PROVIDER_UNAVAILABLE
