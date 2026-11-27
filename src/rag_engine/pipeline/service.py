"""The query pipeline: from a question to an answer with a status and checked citations.

Stages, in order: retrieval, context assembly, the sufficiency gate, generation, composition.
Every stage runs inside one shared time budget. The outcome is always one of four statuses and
there is no fifth way to fail:

* ``answered``: the model answered and every citation resolves to evidence of this request;
* ``insufficient_evidence``: nothing was found, the gate judged the evidence too weak (the model
  is then never called), or the model itself said the evidence does not contain the answer;
* ``unverified``: the model produced text but its citations did not hold up; the text is kept
  apart as ``raw_text`` and is never presented as the answer;
* ``error``: a stage failed (including running out of time); the stable code says which.

Cancellation propagates (it is not an error status) and leaves nothing held: the request slot and
the model provider's slot are released by their context managers.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field

from rag_engine.context.builder import ContextBuilder, ContextPolicy
from rag_engine.context.prompt import system_prompt_tokens
from rag_engine.domain import (
    AnswerStatus,
    AskStatus,
    Citation,
    CitationProblem,
    Context,
    ContextBudget,
    ContextReport,
    ErrorCode,
    Evidence,
    RagError,
    RequestId,
    RetrievalQuery,
    require,
)
from rag_engine.generation.generator import (
    AnswerGenerator,
    CitationsEvent,
    DeltaEvent,
    DoneEvent,
    EvidenceEvent,
    GeneratedAnswer,
    RetractEvent,
)
from rag_engine.observability import get_logger, request_scope
from rag_engine.pipeline.budget import StageShares, TimeBudget
from rag_engine.ports import Clock, TokenCounter, TokenUsage
from rag_engine.retrieval.service import RetrievalDiagnostics, RetrievalService, SearchOutcome
from rag_engine.retrieval.sufficiency import GateDecision, SufficiencyGate, signals_from

_log = get_logger("rag_engine.pipeline")
_MIN_ANSWER_CHARS = 2


@dataclass(frozen=True, slots=True)
class PipelineLimits:
    total_timeout_s: float = 300.0
    shares: StageShares = field(default_factory=StageShares)
    concurrency: int = 1

    def __post_init__(self) -> None:
        require(self.total_timeout_s > 0 and self.concurrency >= 1, "invalid pipeline limits")


@dataclass(frozen=True, slots=True)
class PipelineError:
    code: ErrorCode
    message: str
    retryable: bool
    stage: str


@dataclass(frozen=True, slots=True)
class PipelineDiagnostics:
    stage_ms: Mapping[str, int]
    total_ms: int
    retrieval: RetrievalDiagnostics | None = None
    fallbacks: tuple[str, ...] = ()
    context: ContextReport | None = None
    context_budget: ContextBudget | None = None
    gate: GateDecision | None = None
    llm_called: bool = False
    usage: TokenUsage | None = None
    prompt_sha256: str = ""
    template_id: str = ""
    attempts: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineAnswer:
    """The result of a request. The invariants make a wrong combination unrepresentable."""

    request_id: RequestId
    status: AskStatus
    text: str
    citations: tuple[Citation, ...]
    sources: tuple[Evidence, ...]
    diagnostics: PipelineDiagnostics
    reasons: tuple[str, ...] = ()
    raw_text: str = ""
    error: PipelineError | None = None

    def __post_init__(self) -> None:
        known = {item.evidence_id: item for item in self.sources}
        if self.status is AskStatus.ANSWERED:
            visible = sum(char.isalnum() for char in self.text)
            require(visible >= _MIN_ANSWER_CHARS, "an answered result needs a real answer")
            require(len(self.citations) > 0, "an answered result must cite its evidence")
            require(self.error is None and not self.raw_text, "an answer carries no error")
            for citation in self.citations:
                source = known.get(citation.evidence_id)
                require(
                    source is not None
                    and source.chunk_id == citation.chunk_id
                    and source.locator == citation.locator,
                    "a citation must point at the evidence it names",
                )
        else:
            require(not self.text and not self.citations, "only an answer has text and citations")
        require(
            (self.status is AskStatus.ERROR) == (self.error is not None),
            "exactly an error result carries an error",
        )
        require(
            self.status is AskStatus.UNVERIFIED or not self.raw_text,
            "only an unverified result keeps the rejected text",
        )


@dataclass(frozen=True, slots=True)
class FinalEvent:
    """The last event of a stream: the one validated outcome."""

    answer: PipelineAnswer


type PipelineEvent = EvidenceEvent | DeltaEvent | RetractEvent | CitationsEvent | FinalEvent


@dataclass(frozen=True, slots=True)
class _Ready:
    """Everything generation needs, once retrieval, context and the gate have passed."""

    outcome: SearchOutcome
    context: Context
    gate: GateDecision


class RagPipeline:
    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        builder: ContextBuilder,
        context_policy: ContextPolicy,
        counter: TokenCounter,
        gate: Callable[[], SufficiencyGate],
        generator: AnswerGenerator,
        clock: Clock,
        limits: PipelineLimits | None = None,
        release_models: Callable[[], None] | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._builder = builder
        self._policy = context_policy
        self._counter = counter
        self._gate = gate
        self._generator = generator
        self._clock = clock
        self._limits = limits or PipelineLimits()
        self._release = release_models
        self._slot = asyncio.Semaphore(self._limits.concurrency)

    # ------------------------------------------------------------------ one answer

    async def ask(self, query: RetrievalQuery) -> PipelineAnswer:
        async with self._slot:
            with request_scope(query.request_id):
                budget = TimeBudget(self._limits.total_timeout_s, self._clock, self._limits.shares)
                try:
                    prepared = await self._prepare(query, budget)
                    if isinstance(prepared, PipelineAnswer):
                        return self._logged(prepared)
                    self._release_models()
                    try:
                        generated = await self._generate(query, prepared, budget)
                    except RagError as error:
                        return self._logged(
                            self._failed(query, "generation", error, budget, prepared)
                        )
                    return self._logged(self._compose(query, generated, prepared, budget))
                except asyncio.CancelledError:
                    _log.warning("pipeline.cancelled", elapsed_ms=budget.elapsed_ms())
                    raise

    async def _generate(
        self, query: RetrievalQuery, ready: _Ready, budget: TimeBudget
    ) -> GeneratedAnswer:
        timeout = budget.require_time("generation")
        began = budget.now_ns()
        try:
            async with asyncio.timeout(timeout):
                return await self._generator.answer(query.request_id, query.text, ready.context)
        except TimeoutError:
            raise budget.timeout("generation") from None
        finally:
            budget.record("generation", began)

    # ------------------------------------------------------------------- streaming

    async def stream(
        self, query: RetrievalQuery, *, verified_only: bool = True
    ) -> AsyncIterator[PipelineEvent]:
        """Events: evidence, deltas, citations, then one ``FinalEvent`` (never an exception).

        ``verified_only`` (the default) sends the answer text once, after its citations were
        validated, so unvalidated text never reaches the client. With ``verified_only=False``
        deltas are live and *provisional*: a failed validation is announced by a ``RetractEvent``
        and the final event then reports ``unverified`` or ``error``. Only ``FinalEvent`` is
        authoritative. Insufficient evidence ends the stream at once, without a model call.
        """
        async with self._slot:
            with request_scope(query.request_id):
                budget = TimeBudget(self._limits.total_timeout_s, self._clock, self._limits.shares)
                try:
                    prepared = await self._prepare(query, budget)
                    if isinstance(prepared, PipelineAnswer):
                        yield FinalEvent(self._logged(prepared))
                        return
                    yield EvidenceEvent(prepared.context.evidence)
                    self._release_models()
                    async for item in self._generated_events(
                        query, prepared, budget, verified_only
                    ):
                        if isinstance(item, PipelineAnswer):
                            yield FinalEvent(self._logged(item))
                            return
                        yield item
                except asyncio.CancelledError:
                    _log.warning("pipeline.cancelled", elapsed_ms=budget.elapsed_ms())
                    raise

    async def _generated_events(
        self, query: RetrievalQuery, ready: _Ready, budget: TimeBudget, verified_only: bool
    ) -> AsyncIterator[PipelineEvent | PipelineAnswer]:
        """Generation as a stream, ending in the one validated ``PipelineAnswer``."""
        began = budget.now_ns()
        deadline = time.monotonic() + budget.allowance_s("generation")
        stream = self._generator.stream(
            query.request_id, query.text, ready.context, verified_only=verified_only
        ).__aiter__()
        try:
            if budget.allowance_s("generation") <= 0:
                raise budget.timeout("generation")
            while True:
                try:
                    event = await asyncio.wait_for(
                        stream.__anext__(), timeout=max(0.001, deadline - time.monotonic())
                    )
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    raise budget.timeout("generation") from None
                if isinstance(event, EvidenceEvent):
                    continue  # the pipeline already sent the evidence
                if isinstance(event, DoneEvent):
                    budget.record("generation", began)
                    yield self._compose(query, event.answer, ready, budget)
                    return
                yield event
            raise RagError(ErrorCode.E_INTEGRITY, "the generation stream ended without a result")
        except RagError as error:
            budget.record("generation", began)
            yield self._failed(query, "generation", error, budget, ready)
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

    # ---------------------------------------------------------------------- stages

    async def _prepare(self, query: RetrievalQuery, budget: TimeBudget) -> _Ready | PipelineAnswer:
        try:
            outcome = await self._search(query, budget)
        except RagError as error:
            return self._failed(query, "retrieval", error, budget)
        hits = outcome.result.hits
        if not hits:
            return self._insufficient(query, budget, ("no_hits",), outcome=outcome)
        limits = self._generator.limits()
        try:
            budget.require_time("context")
            began = budget.now_ns()
            context = self._builder.build(
                hits,
                question_tokens=self._counter.count(query.text),
                system_tokens=system_prompt_tokens(self._counter),
                limits=limits,
                policy=self._policy,
            )
            budget.record("context", began)
            if budget.remaining_s() <= 0:
                raise budget.timeout("context")
        except RagError as error:
            return self._failed(query, "context", error, budget, outcome=outcome)
        self._log_context(context)
        if not context.evidence:
            return self._insufficient(query, budget, ("no_evidence",), outcome, context)
        signals = signals_from(
            hits, query.text, context.evidence[0].text, outcome.diagnostics.language
        )
        decision = self._gate().decide(signals)
        _log.info(
            "pipeline.gate",
            sufficient=decision.sufficient,
            calibrated=decision.calibrated,
            reasons=list(decision.reasons),
        )
        if not decision.sufficient:
            reasons = tuple(f"gate:{reason}" for reason in decision.reasons)
            return self._insufficient(query, budget, reasons, outcome, context, decision)
        return _Ready(outcome, context, decision)

    async def _search(self, query: RetrievalQuery, budget: TimeBudget) -> SearchOutcome:
        timeout = budget.require_time("retrieval")
        began = budget.now_ns()
        try:
            async with asyncio.timeout(timeout):
                return await self._retrieval.search(query)
        except TimeoutError:
            raise budget.timeout("retrieval") from None
        finally:
            budget.record("retrieval", began)

    def _release_models(self) -> None:
        """Hand the memory of the in-process models back before a local model is loaded."""
        if self._release is None:
            return
        try:
            self._release()
        except Exception as error:
            _log.warning("pipeline.release_failed", error_type=type(error).__name__)

    @staticmethod
    def _log_context(context: Context) -> None:
        """NFR-07: every cut and every dropped chunk is in the diagnostics and in the log."""
        report = context.report
        _log.info(
            "pipeline.context",
            evidence=len(context.evidence),
            tokens_used=report.tokens_used,
            budget=report.budget,
            truncated=len(report.truncated),
            dropped=len(report.dropped),
            merged=len(report.merged),
        )

    # ----------------------------------------------------------------- composition

    def _compose(
        self,
        query: RetrievalQuery,
        generated: GeneratedAnswer,
        ready: _Ready,
        budget: TimeBudget,
    ) -> PipelineAnswer:
        warnings = generated.warnings
        if generated.check is not None and CitationProblem.LOW_OVERLAP in generated.check.problems:
            warnings = (*warnings, "low_overlap")
        diagnostics = self._diagnostics(
            budget,
            ready.outcome,
            ready.context,
            ready.gate,
            usage=generated.usage,
            prompt_sha256=generated.prompt_sha256,
            template_id=generated.template_id,
            attempts=generated.attempts,
            llm_called=generated.attempts > 0,
            warnings=warnings,
        )
        if generated.status is AnswerStatus.REFUSED:
            reason = "no_evidence" if "no_evidence" in generated.warnings else "model_refused"
            return self._answer(
                query, AskStatus.INSUFFICIENT_EVIDENCE, ready, diagnostics, (reason,)
            )
        if generated.status is AnswerStatus.UNVERIFIED:
            problems = tuple(p.value for p in (generated.check.problems if generated.check else ()))
            return self._answer(
                query,
                AskStatus.UNVERIFIED,
                ready,
                diagnostics,
                problems,
                raw_text=generated.raw_text,
            )
        return self._answer(
            query,
            AskStatus.ANSWERED,
            ready,
            diagnostics,
            (),
            text=generated.text,
            citations=generated.citations,
        )

    def _answer(
        self,
        query: RetrievalQuery,
        status: AskStatus,
        ready: _Ready,
        diagnostics: PipelineDiagnostics,
        reasons: tuple[str, ...],
        *,
        text: str = "",
        citations: tuple[Citation, ...] = (),
        raw_text: str = "",
    ) -> PipelineAnswer:
        return PipelineAnswer(
            query.request_id,
            status,
            text,
            citations,
            ready.context.evidence,
            diagnostics,
            reasons,
            raw_text,
        )

    def _insufficient(
        self,
        query: RetrievalQuery,
        budget: TimeBudget,
        reasons: tuple[str, ...],
        outcome: SearchOutcome | None = None,
        context: Context | None = None,
        gate: GateDecision | None = None,
    ) -> PipelineAnswer:
        return PipelineAnswer(
            query.request_id,
            AskStatus.INSUFFICIENT_EVIDENCE,
            "",
            (),
            context.evidence if context else (),
            self._diagnostics(budget, outcome, context, gate),
            reasons,
        )

    def _failed(
        self,
        query: RetrievalQuery,
        stage: str,
        error: RagError,
        budget: TimeBudget,
        ready: _Ready | None = None,
        outcome: SearchOutcome | None = None,
    ) -> PipelineAnswer:
        _log.error("pipeline.failed", stage=stage, error_code=error.code.value)
        known = ready.outcome if ready else outcome
        context = ready.context if ready else None
        return PipelineAnswer(
            query.request_id,
            AskStatus.ERROR,
            "",
            (),
            context.evidence if context else (),
            self._diagnostics(budget, known, context, ready.gate if ready else None),
            (),
            error=PipelineError(error.code, error.message, error.retryable, stage),
        )

    def _diagnostics(
        self,
        budget: TimeBudget,
        outcome: SearchOutcome | None,
        context: Context | None,
        gate: GateDecision | None,
        *,
        usage: TokenUsage | None = None,
        prompt_sha256: str = "",
        template_id: str = "",
        attempts: int = 0,
        llm_called: bool = False,
        warnings: tuple[str, ...] = (),
    ) -> PipelineDiagnostics:
        return PipelineDiagnostics(
            stage_ms=budget.stage_ms,
            total_ms=budget.elapsed_ms(),
            retrieval=outcome.diagnostics if outcome else None,
            fallbacks=outcome.result.fallbacks if outcome else (),
            context=context.report if context else None,
            context_budget=context.budget if context else None,
            gate=gate,
            llm_called=llm_called,
            usage=usage,
            prompt_sha256=prompt_sha256,
            template_id=template_id,
            attempts=attempts,
            warnings=warnings,
        )

    @staticmethod
    def _logged(answer: PipelineAnswer) -> PipelineAnswer:
        _log.info(
            "pipeline.done",
            status=answer.status.value,
            reasons=list(answer.reasons),
            llm_called=answer.diagnostics.llm_called,
            total_ms=answer.diagnostics.total_ms,
            citations=len(answer.citations),
        )
        return answer
