"""Metrics of generated answers: faithfulness, citation accuracy, correctness, refusals.

All functions are pure. What needs a judgement (does the evidence support this claim) comes in as a
verdict, so the same metrics work with a human, a model or the lexical screen, and a report can say
which. Nothing here decides that a label is correct: dataset labels count only when reviewed.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_ENTITY = re.compile(r"\b(?:[A-Z][\w-]*\d[\w-]*|[A-Z]{2,}[\w-]*|[A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b")


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().replace("ё", "е").split())


# ----------------------------------------------------------------------------- faithfulness


@dataclass(frozen=True, slots=True)
class ProxySupport:
    checked: int
    unsupported: tuple[str, ...]

    @property
    def ratio(self) -> float | None:
        """Share of the answer's numbers and named things found in the cited evidence."""
        return None if self.checked == 0 else 1 - len(self.unsupported) / self.checked


def deterministic_support(answer: str, cited_evidence: Sequence[str]) -> ProxySupport:
    """Faithfulness proxy (a): every number and named thing of the answer occurs in the evidence."""
    body = re.sub(r"\[E[1-9][0-9]*\]", " ", answer)
    joined = " ".join(cited_evidence)
    numbers = {m.group().rstrip(".,:") for m in _NUMBER.finditer(body)}
    known_numbers = {m.group().rstrip(".,:") for m in _NUMBER.finditer(joined)}
    names = {m.group() for m in _ENTITY.finditer(body)}
    haystack = _fold(joined)
    # A number is a whole token ("30" is not found inside "02:30"); a name may be a phrase.
    missing = sorted(n for n in numbers if n not in known_numbers) + sorted(
        n for n in names if _fold(n) not in haystack
    )
    return ProxySupport(len(numbers) + len(names), tuple(sorted(missing)))


def faithfulness(verdicts: Sequence[bool]) -> float | None:
    """Share of claims judged supported; ``None`` for an answer with no claims."""
    return None if not verdicts else sum(verdicts) / len(verdicts)


def hallucination_rate(answered_claims: Sequence[Sequence[bool]]) -> float | None:
    """Share of answered replies with at least one unsupported claim (denominator: answered)."""
    if not answered_claims:
        return None
    return sum(not all(claims) for claims in answered_claims) / len(answered_claims)


# ------------------------------------------------------------------------ citation accuracy


@dataclass(frozen=True, slots=True)
class CitationScore:
    precision: float | None
    recall: float | None
    valid_markers: float | None


def citation_score(
    supported: Sequence[bool],
    cited_documents: Sequence[str],
    gold_documents: Sequence[str],
    *,
    valid_markers: int,
    total_markers: int,
) -> CitationScore:
    """Precision: cited chunks that support the claim. Recall: gold documents that were cited."""
    gold = set(gold_documents)
    return CitationScore(
        precision=None if not supported else sum(supported) / len(supported),
        recall=None if not gold else len(gold & set(cited_documents)) / len(gold),
        valid_markers=None if total_markers == 0 else valid_markers / total_markers,
    )


# ----------------------------------------------------------------------------- correctness


@dataclass(frozen=True, slots=True)
class Correctness:
    exact: bool | None
    completeness: float | None


def correctness(
    answer: str, reference_answer: str | None, reference_facts: Sequence[str]
) -> Correctness:
    """Normalised match against the reference answer, and the share of reference facts present."""
    plain = _fold(re.sub(r"\[E[1-9][0-9]*\]", " ", answer))
    exact = None if reference_answer is None else _fold(reference_answer) in plain
    completeness = (
        None
        if not reference_facts
        else sum(_fold(f) in plain for f in reference_facts) / len(reference_facts)
    )
    return Correctness(exact, completeness)


# --------------------------------------------------------------------------- no-answer matrix


@dataclass(frozen=True, slots=True)
class NoAnswerMatrix:
    answered_answerable: int
    refused_answerable: int  # false refusals
    answered_unanswerable: int  # false answers
    refused_unanswerable: int  # correct refusals

    @property
    def refusal_precision(self) -> float | None:
        refused = self.refused_answerable + self.refused_unanswerable
        return None if refused == 0 else self.refused_unanswerable / refused

    @property
    def refusal_recall(self) -> float | None:
        unanswerable = self.answered_unanswerable + self.refused_unanswerable
        return None if unanswerable == 0 else self.refused_unanswerable / unanswerable

    @property
    def false_refusal_rate(self) -> float | None:
        answerable = self.answered_answerable + self.refused_answerable
        return None if answerable == 0 else self.refused_answerable / answerable

    @property
    def false_answer_rate(self) -> float | None:
        unanswerable = self.answered_unanswerable + self.refused_unanswerable
        return None if unanswerable == 0 else self.answered_unanswerable / unanswerable


def no_answer_matrix(outcomes: Sequence[tuple[bool, bool]]) -> NoAnswerMatrix:
    """``outcomes`` are ``(answerable, refused)`` pairs."""
    return NoAnswerMatrix(
        answered_answerable=sum(a and not r for a, r in outcomes),
        refused_answerable=sum(a and r for a, r in outcomes),
        answered_unanswerable=sum(not a and not r for a, r in outcomes),
        refused_unanswerable=sum(not a and r for a, r in outcomes),
    )
