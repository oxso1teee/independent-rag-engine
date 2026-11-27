"""Evidence-grounded answer generation.

One call renders the ``answer.v1`` prompt from a ``Context``, checks that it fits the model's
window, asks the provider, and turns the reply into one of two outcomes: a refusal (the model
said the evidence does not contain the answer) or an answer whose every citation names evidence
that was in the context. Nothing here invents a source: a marker that does not resolve fails the
answer after one retry rather than being passed on.

Failures are separated by cause. A prompt that cannot fit is refused before the call; a prompt the
provider silently cut is detected from the token count it reports; an empty, looping or
self-contradicting reply is a degenerate answer; an unresolvable or missing citation is invalid.
Degenerate and invalid replies get one retry at a slightly higher temperature, since the first
attempt at temperature 0 would repeat itself. Streaming cannot retry what it already sent, so a
stream ends in an error instead.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

from rag_engine.context.prompt import REFUSAL_SENTINEL
from rag_engine.domain import (
    AnswerStatus,
    Citation,
    CitationCheck,
    CitationProblem,
    Context,
    ErrorCode,
    Evidence,
    RagError,
    RequestId,
    require,
)
from rag_engine.generation.citations import CitationValidator
from rag_engine.generation.templates import RenderedPrompt, render_answer
from rag_engine.observability import get_logger
from rag_engine.ports import (
    FinishReason,
    GenerationParams,
    GenerationRequest,
    LLMProvider,
    TokenUsage,
)

_log = get_logger("rag_engine.generation")
_WRAPPING = " \t\r\n\"'`.*"
_MARKER = re.compile(r"\[E([1-9][0-9]*)\]")


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    max_new_tokens: int = 512
    temperature: float = 0.0
    retry_temperature: float = 0.3
    max_attempts: int = 2
    strict_overlap: bool = False
    min_answer_chars: int = 2
    """An answer with fewer letters or digits (markers excluded) is degenerate."""
    drift_ratio: float = 0.15

    def __post_init__(self) -> None:
        require(
            self.max_new_tokens > 0 and self.max_attempts >= 1 and self.min_answer_chars >= 1,
            "invalid generation policy",
        )
        require(0.0 <= self.temperature <= self.retry_temperature, "invalid temperatures")
        require(0.0 < self.drift_ratio < 1.0, "invalid drift ratio")


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    status: AnswerStatus
    text: str
    citations: tuple[Citation, ...]
    check: CitationCheck | None
    usage: TokenUsage
    prompt_sha256: str
    template_id: str
    attempts: int
    latency_ms: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class DeltaEvent:
    """A piece of the answer text.

    ``provisional`` is True while the stream is live: the text has not been validated yet and may
    be withdrawn by a ``RetractEvent``. It is False in ``verified_only`` mode, where the text is
    sent only after the citations were checked.
    """

    text: str
    provisional: bool = True


@dataclass(frozen=True, slots=True)
class RetractEvent:
    """Everything streamed so far must be discarded; an error follows.

    Sent when the answer failed validation, or the provider failed, after text was already
    delivered. A client must not keep or show the retracted text as an answer.
    """

    code: ErrorCode
    reason: str


@dataclass(frozen=True, slots=True)
class CitationsEvent:
    citations: tuple[Citation, ...]


@dataclass(frozen=True, slots=True)
class DoneEvent:
    answer: GeneratedAnswer


type AnswerEvent = EvidenceEvent | DeltaEvent | RetractEvent | CitationsEvent | DoneEvent


def is_refusal(text: str) -> bool:
    return text.strip(_WRAPPING) == REFUSAL_SENTINEL.strip(_WRAPPING)


def is_degenerate(text: str, min_chars: int = 2) -> bool:
    """Too short to be an answer, mixed with the refusal sentinel, or stuck in a loop.

    Length is counted in letters and digits with the ``[E<n>]`` markers removed, so a reply that is
    only a citation, a single character or punctuation never counts as an answer.
    """
    stripped = text.strip()
    visible = sum(char.isalnum() for char in _MARKER.sub("", stripped))
    if visible < min_chars:
        return True
    if REFUSAL_SENTINEL in stripped and not is_refusal(stripped):
        return True
    words = stripped.split()
    return len(words) >= 12 and len(set(words)) / len(words) < 0.15


class AnswerGenerator:
    def __init__(
        self,
        llm: LLMProvider,
        validator: CitationValidator | None = None,
        policy: GenerationPolicy | None = None,
    ) -> None:
        self._llm = llm
        self._validator = validator or CitationValidator()
        self._policy = policy or GenerationPolicy()

    # ----------------------------------------------------------------- one-shot

    async def answer(
        self, request_id: RequestId, question: str, context: Context
    ) -> GeneratedAnswer:
        if not context.evidence:
            return self._refusal_without_model(context)
        rendered = render_answer(question, context)
        max_new, counted = self._fit(rendered)
        usage = TokenUsage(0, 0)
        latency = 0
        warnings: list[str] = []
        failure: RagError | None = None
        for attempt in range(1, self._policy.max_attempts + 1):
            temperature = (
                self._policy.temperature if attempt == 1 else self._policy.retry_temperature
            )
            request = GenerationRequest(
                request_id,
                rendered.system_prompt,
                rendered.prompt,
                GenerationParams(temperature, max_new, self._llm.limits().context_window),
            )
            result = await self._llm.generate(request)
            usage = TokenUsage(
                usage.prompt_tokens + result.usage.prompt_tokens,
                usage.completion_tokens + result.usage.completion_tokens,
            )
            latency += result.latency_ms
            warnings = self._inspect(result.usage.prompt_tokens, counted, result.finish_reason)
            try:
                outcome = self._finalize(result.text, context, rendered, usage, attempt, latency)
            except RagError as error:
                if error.code not in {ErrorCode.E_DEGENERATE_ANSWER, ErrorCode.E_CITATION_INVALID}:
                    raise
                failure = error
                _log.warning("generation.retry", attempt=attempt, code=error.code.value)
                continue
            self._record(outcome)
            return _with_warnings(outcome, warnings)
        assert failure is not None  # noqa: S101 - max_attempts >= 1, so a failure was recorded
        raise failure

    # ----------------------------------------------------------------- streaming

    async def stream(
        self,
        request_id: RequestId,
        question: str,
        context: Context,
        *,
        verified_only: bool = False,
    ) -> AsyncIterator[AnswerEvent]:
        """Events: evidence, deltas, citations, done; a failure ends in an error.

        Live (default): deltas arrive as the model writes them and are *provisional*, because the
        citations can only be checked on the finished text. If the check fails, or the provider
        fails after text was sent, a ``RetractEvent`` is emitted and then the error is raised; a
        client shows live text as a draft and replaces it with ``DoneEvent.answer`` (the only
        validated text). The refusal sentinel is held back, so a refusal never streams text.

        ``verified_only``: the reply is collected and validated first; the text is then sent once
        as a non-provisional delta, so unvalidated text never reaches the client and a failure
        emits no text at all. The cost is the first-token latency.
        """
        yield EvidenceEvent(context.evidence)
        if not context.evidence:
            yield CitationsEvent(())
            yield DoneEvent(self._refusal_without_model(context))
            return
        rendered = render_answer(question, context)
        max_new, counted = self._fit(rendered)
        request = GenerationRequest(
            request_id,
            rendered.system_prompt,
            rendered.prompt,
            GenerationParams(self._policy.temperature, max_new, self._llm.limits().context_window),
        )
        full = ""
        held = ""
        deciding = True
        sent = False
        try:
            async for delta in self._llm.stream(request):
                full += delta.text
                if verified_only:
                    continue
                if not deciding:
                    sent = True
                    yield DeltaEvent(delta.text)
                    continue
                held += delta.text
                probe = held.lstrip(_WRAPPING)
                if probe and not REFUSAL_SENTINEL.startswith(probe):
                    deciding = False
                    sent = True
                    yield DeltaEvent(held)
            if not verified_only and deciding and held and not is_refusal(held):
                sent = True
                yield DeltaEvent(held)
            # A stream reports no usage, so it is counted here, not measured by the provider.
            usage = TokenUsage(counted, self._llm.count_tokens(full))
            outcome = self._finalize(full, context, rendered, usage, 1, 0)
        except RagError as error:
            if sent:
                yield RetractEvent(error.code, error.message)
            raise
        self._record(outcome)
        outcome = _with_warnings(outcome, ["usage_estimated"])
        if verified_only and outcome.status is AnswerStatus.ANSWERED:
            yield DeltaEvent(outcome.text, provisional=False)
        yield CitationsEvent(outcome.citations)
        yield DoneEvent(outcome)

    # ----------------------------------------------------------------- internals

    def _fit(self, rendered: RenderedPrompt) -> tuple[int, int]:
        """The tokens the model may produce and the prompt's own measured size."""
        limits = self._llm.limits()
        counted = self._llm.count_tokens(rendered.system_prompt) + self._llm.count_tokens(
            rendered.prompt
        )
        max_new = min(self._policy.max_new_tokens, limits.max_new_tokens)
        if counted + max_new > limits.context_window:
            raise RagError(
                ErrorCode.E_CONTEXT_BUDGET,
                "the prompt and the answer allowance do not fit the model's context window",
                context={
                    "prompt_tokens": counted,
                    "max_new_tokens": max_new,
                    "context_window": limits.context_window,
                },
            )
        return max_new, counted

    def _inspect(self, reported: int, counted: int, finish: FinishReason) -> list[str]:
        window = self._llm.limits().context_window
        if reported >= window - 1:
            raise RagError(
                ErrorCode.E_CONTEXT_TRUNCATED,
                "the provider reports a prompt as long as its window: it was probably cut",
                context={"prompt_tokens": reported, "context_window": window},
            )
        notes: list[str] = []
        if abs(reported - counted) > self._policy.drift_ratio * max(reported, counted):
            notes.append("token_count_drift")
        if finish is FinishReason.LENGTH:
            notes.append("answer_truncated")
        return notes

    def _finalize(
        self,
        text: str,
        context: Context,
        rendered: RenderedPrompt,
        usage: TokenUsage,
        attempts: int,
        latency: int,
    ) -> GeneratedAnswer:
        if is_refusal(text):
            return GeneratedAnswer(
                AnswerStatus.REFUSED, "", (), None, usage, rendered.prompt_sha256,
                rendered.template_id, attempts, latency,
            )  # fmt: skip
        if is_degenerate(text, self._policy.min_answer_chars):
            raise RagError(ErrorCode.E_DEGENERATE_ANSWER, "the model returned no usable answer")
        answer = text.strip()
        check = self._validator.validate(answer, context)
        fatal = [
            p
            for p in check.problems
            if p is not CitationProblem.LOW_OVERLAP or self._policy.strict_overlap
        ]
        if fatal:
            raise RagError(
                ErrorCode.E_CITATION_INVALID,
                "the answer cites evidence that does not exist or does not support it",
                context={
                    "problems": [p.value for p in fatal],
                    "unknown": list(check.unknown),
                },
            )
        return GeneratedAnswer(
            AnswerStatus.ANSWERED,
            answer,
            _resolve(answer, check, context),
            check,
            usage,
            rendered.prompt_sha256,
            rendered.template_id,
            attempts,
            latency,
        )

    @staticmethod
    def _refusal_without_model(context: Context) -> GeneratedAnswer:
        """No evidence at all: answering would only invite a guess, so the model is not called."""
        require(not context.evidence, "only an empty context is refused without the model")
        return GeneratedAnswer(
            AnswerStatus.REFUSED, "", (), None, TokenUsage(0, 0), "", "answer.v1", 0, 0,
            ("no_evidence",),
        )  # fmt: skip

    def _record(self, answer: GeneratedAnswer) -> None:
        _log.info(
            "generation.done",
            status=answer.status.value,
            attempts=answer.attempts,
            prompt_tokens=answer.usage.prompt_tokens,
            completion_tokens=answer.usage.completion_tokens,
            citations=len(answer.citations),
            prompt_sha256=answer.prompt_sha256,
        )


def _with_warnings(answer: GeneratedAnswer, extra: list[str]) -> GeneratedAnswer:
    return replace(answer, warnings=(*answer.warnings, *extra))


def _resolve(text: str, check: CitationCheck, context: Context) -> tuple[Citation, ...]:
    by_id = {item.evidence_id: item for item in context.evidence}
    uses: dict[str, int] = {}
    for number in _MARKER.findall(text):
        uses[f"E{number}"] = uses.get(f"E{number}", 0) + 1
    return tuple(
        Citation(
            evidence_id,
            by_id[evidence_id].chunk_id,
            by_id[evidence_id].document_id,
            by_id[evidence_id].version_id,
            by_id[evidence_id].locator,
            uses[evidence_id],
        )
        for evidence_id in check.cited
    )
