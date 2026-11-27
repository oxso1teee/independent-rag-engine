"""A whole pipeline over in-memory pieces: a real search service, context builder and generator."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.context.builder import ContextBuilder, ContextPolicy
from rag_engine.domain import RetrievalMode, RetrievalQuery
from rag_engine.generation.generator import AnswerGenerator, GenerationPolicy
from rag_engine.pipeline import PipelineLimits, RagPipeline
from rag_engine.ports import (
    GenerationRequest,
    GenerationResult,
    ModelLimits,
)
from rag_engine.retrieval.sufficiency import SufficiencyGate, Thresholds
from rag_engine.system import SystemClock
from rag_engine.testing import Fail, Reply, ScriptedLLM
from tests.factories import make_ids
from tests.unit.test_search_service import Lookups, build

IDS = make_ids(1_767_300_000_000)
QUESTION = "how are chunks stored in SQLite"
ANSWER = "Chunks are stored in SQLite [E1]."


class SlowLLM(ScriptedLLM):
    """A model that takes ``delay`` seconds before it answers, to exercise time and cancellation."""

    def __init__(self, delay: float, script: list[Reply | Fail]) -> None:
        super().__init__(script)
        self.delay = delay
        self.started = 0
        self.active = 0
        self.peak = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.started += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            return await super().generate(request)
        finally:
            self.active -= 1


@dataclass
class Rig:
    pipeline: RagPipeline
    llm: ScriptedLLM
    released: list[int] = field(default_factory=list)


def query(
    text: str = QUESTION, mode: RetrievalMode = RetrievalMode.HYBRID, top_k: int = 3
) -> RetrievalQuery:
    return RetrievalQuery(request_id=IDS.request_id(), text=text, mode=mode, top_k=top_k)


def rig(
    *steps: Reply | Fail,
    llm: ScriptedLLM | None = None,
    dense_threshold: float | None = None,
    limits: ModelLimits | None = None,
    pipeline_limits: PipelineLimits | None = None,
    policy: ContextPolicy | None = None,
    release: Callable[[], None] | None = None,
) -> Rig:
    service, embedder, _ = build()
    lookups = Lookups()
    model = llm or ScriptedLLM(list(steps), limits=limits or ModelLimits(8192, 1024))
    counter = HeuristicTokenCounter()
    thresholds = (
        None
        if dense_threshold is None
        else Thresholds(model_id=str(embedder.model_id), dense=dense_threshold)
    )
    out = Rig.__new__(Rig)
    out.llm = model
    out.released = []

    def hook() -> None:
        out.released.append(1)
        if release is not None:
            release()

    out.pipeline = RagPipeline(
        retrieval=service,
        builder=ContextBuilder(counter, lookups),
        context_policy=policy or ContextPolicy(),
        counter=counter,
        gate=lambda: SufficiencyGate(thresholds, str(embedder.model_id)),
        generator=AnswerGenerator(model, policy=GenerationPolicy(unverified_on_invalid=True)),
        clock=SystemClock(),
        limits=pipeline_limits or PipelineLimits(total_timeout_s=30.0),
        release_models=hook,
    )
    return out
