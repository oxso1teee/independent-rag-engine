"""A judge for claims, behind a small port, and its calibration against people.

An answer is split into claims (its sentences without citation markers); the judge says for each
claim whether the cited evidence supports it. Two judges exist: ``LexicalJudge``, a deterministic
overlap screen that needs no model, and ``LLMJudge``, which asks a language model. Neither is
trusted until it has been compared with a human sample: ``calibrate_judge`` returns the agreement
and Cohen's kappa, and a judge counts as calibrated only at the specified thresholds (agreement of
at least 90 % and kappa of at least 0.8). The judge should not be the model that wrote the answers;
the run manifest records which model was which.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from rag_engine.domain import RequestId
from rag_engine.evaluation.stats import cohen_kappa, percent_agreement
from rag_engine.generation.citations import lexical_coverage, split_sentences
from rag_engine.ports import GenerationParams, GenerationRequest, LLMProvider

MIN_AGREEMENT = 0.90
MIN_KAPPA = 0.80
_MARKER = re.compile(r"\[E[1-9][0-9]*\]")

JUDGE_SYSTEM = (
    "You check one claim against evidence. Answer with exactly one word: SUPPORTED if the evidence "
    "states or directly implies the claim, otherwise UNSUPPORTED. Use only the evidence."
)


class Verdict(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class ClaimJudge(Protocol):
    @property
    def judge_id(self) -> str: ...

    async def judge(self, claim: str, evidence: Sequence[str]) -> Verdict: ...


def split_claims(answer: str) -> list[str]:
    """The sentences of an answer without their citation markers; empty ones are dropped."""
    claims = []
    for sentence in split_sentences(answer):
        plain = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(_MARKER.sub("", sentence).split()))
        if any(char.isalnum() for char in plain):
            claims.append(plain)
    return claims


class LexicalJudge:
    """Deterministic: a claim is supported when it shares enough vocabulary with the evidence.

    The same screen the pipeline uses for citations: it is blind to meaning and across scripts, so
    it is a baseline and an offline stand-in, not an arbiter of truth. A claim it cannot judge
    (too few terms, other script than the evidence) is called unsupported here, the cautious side.
    """

    judge_id = "lexical/1"

    def __init__(self, min_coverage: float = 0.5) -> None:
        self._min_coverage = min_coverage

    async def judge(self, claim: str, evidence: Sequence[str]) -> Verdict:
        coverage = lexical_coverage(claim, evidence)
        ok = coverage is not None and coverage >= self._min_coverage
        return Verdict.SUPPORTED if ok else Verdict.UNSUPPORTED


class LLMJudge:
    """Asks a language model; an unclear reply is ``UNSUPPORTED`` (never silently supported)."""

    def __init__(self, llm: LLMProvider, request_id: RequestId, *, name: str) -> None:
        self._llm = llm
        self._request_id = request_id
        self.judge_id = f"llm:{name}/judge.v1"

    async def judge(self, claim: str, evidence: Sequence[str]) -> Verdict:
        numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(evidence, start=1))
        prompt = f"Evidence:\n{numbered}\n\nClaim: {claim}\n\nAnswer SUPPORTED or UNSUPPORTED."
        result = await self._llm.generate(
            GenerationRequest(
                self._request_id, JUDGE_SYSTEM, prompt, GenerationParams(max_new_tokens=8)
            )
        )
        word = result.text.strip().strip(".").upper()
        return Verdict.SUPPORTED if word == "SUPPORTED" else Verdict.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class HumanLabel:
    claim: str
    evidence: tuple[str, ...]
    verdict: Verdict
    reviewer: str


@dataclass(frozen=True, slots=True)
class JudgeCalibration:
    judge_id: str
    items: int
    agreement: float
    kappa: float

    @property
    def calibrated(self) -> bool:
        return self.items > 0 and self.agreement >= MIN_AGREEMENT and self.kappa >= MIN_KAPPA


async def calibrate_judge(judge: ClaimJudge, labels: Sequence[HumanLabel]) -> JudgeCalibration:
    """Agreement and kappa of ``judge`` with people on a labelled sample."""
    if not labels:
        raise ValueError("calibration needs a human-labelled sample")
    given = [(await judge.judge(item.claim, item.evidence)).value for item in labels]
    human = [item.verdict.value for item in labels]
    return JudgeCalibration(
        judge.judge_id, len(labels), percent_agreement(given, human), cohen_kappa(given, human)
    )
