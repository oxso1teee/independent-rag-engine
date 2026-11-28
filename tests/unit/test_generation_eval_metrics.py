"""Generation metrics with hand-computed values, and the judge with its calibration."""

from __future__ import annotations

import asyncio

import pytest

from rag_engine.domain import RagError
from rag_engine.evaluation.generation_metrics import (
    citation_score,
    correctness,
    deterministic_support,
    faithfulness,
    hallucination_rate,
    no_answer_matrix,
)
from rag_engine.evaluation.judge import (
    ClaimJudge,
    HumanLabel,
    LexicalJudge,
    LLMJudge,
    Verdict,
    calibrate_judge,
    split_claims,
)
from rag_engine.testing import Reply, ScriptedLLM
from tests.factories import make_ids

pytestmark = pytest.mark.unit

EVIDENCE = ["Backups run every night at 02:30 UTC and are kept for 14 days at Site Alpha."]


def test_the_deterministic_proxy_finds_numbers_and_names_missing_from_the_evidence() -> None:
    good = deterministic_support("Backups run at 02:30 UTC and are kept 14 days [E1].", EVIDENCE)
    assert good.unsupported == () and good.ratio == 1.0 and good.checked >= 3
    bad = deterministic_support("Backups are kept 30 days at Site Beta [E1].", EVIDENCE)
    assert set(bad.unsupported) == {"30", "Site Beta"} and bad.ratio is not None and bad.ratio < 1
    assert deterministic_support("nothing checkable here", EVIDENCE).ratio is None


def test_faithfulness_and_hallucination_rate_use_the_right_denominators() -> None:
    assert faithfulness([True, True, False, True]) == 0.75
    assert faithfulness([]) is None
    answered = [[True, True], [True, False], [True], []]  # the empty one has no unsupported claim
    assert hallucination_rate(answered) == pytest.approx(0.25)
    assert hallucination_rate([]) is None


def test_citation_accuracy_is_precision_recall_and_marker_validity() -> None:
    score = citation_score(
        [True, True, False, True],
        cited_documents=["a.md", "b.md", "x.md"],
        gold_documents=["a.md", "b.md", "c.md"],
        valid_markers=3,
        total_markers=4,
    )
    assert score.precision == 0.75 and score.valid_markers == 0.75
    assert score.recall == pytest.approx(2 / 3)
    empty = citation_score([], [], [], valid_markers=0, total_markers=0)
    assert empty == citation_score([], [], [], valid_markers=0, total_markers=0)
    assert empty.precision is None and empty.recall is None and empty.valid_markers is None


def test_correctness_matches_the_reference_after_normalisation() -> None:
    result = correctness(
        "Резервные копии создаются в 02:30 UTC [E1].",
        "в 02:30 utc",
        ["02:30 UTC", "14 дней", "резервные копии"],
    )
    assert result.exact is True and result.completeness == pytest.approx(2 / 3)
    assert correctness("anything", None, []) == correctness("else", None, [])
    assert correctness("anything", None, []).exact is None


def test_the_no_answer_matrix_separates_false_refusals_from_false_answers() -> None:
    # (answerable, refused): 6 answerable (5 answered, 1 refused), 4 unanswerable (1+3)
    outcomes = [(True, False)] * 5 + [(True, True)] + [(False, False)] + [(False, True)] * 3
    matrix = no_answer_matrix(outcomes)
    assert matrix.false_refusal_rate == pytest.approx(1 / 6)
    assert matrix.false_answer_rate == pytest.approx(1 / 4)
    assert matrix.refusal_recall == pytest.approx(3 / 4)
    assert matrix.refusal_precision == pytest.approx(3 / 4)
    assert no_answer_matrix([]).false_refusal_rate is None


def test_claims_are_sentences_without_markers() -> None:
    assert split_claims(
        "Backups run nightly [E1]. They are kept 14 days [E1][E2].\n- Printers upstairs"
    ) == [
        "Backups run nightly.",
        "They are kept 14 days.",
        "- Printers upstairs",
    ]
    assert split_claims("[E1]") == []


def judge(claim: str, judge_: ClaimJudge) -> Verdict:
    return asyncio.run(judge_.judge(claim, EVIDENCE))


def test_the_lexical_judge_supports_what_overlaps_and_is_cautious_otherwise() -> None:
    lexical = LexicalJudge()
    assert judge("Backups run every night and are kept for 14 days", lexical) is Verdict.SUPPORTED
    assert judge("The printers are encrypted with a hardware token", lexical) is Verdict.UNSUPPORTED
    assert judge("Да", lexical) is Verdict.UNSUPPORTED  # cannot be judged: not "supported"


def test_the_llm_judge_reads_one_word_and_treats_anything_else_as_unsupported() -> None:
    llm = ScriptedLLM([Reply("SUPPORTED."), Reply("unsupported"), Reply("Maybe, I think so")])
    llm_judge = LLMJudge(llm, make_ids().request_id(), name="judge-model")
    assert [judge("c", llm_judge) for _ in range(3)] == [
        Verdict.SUPPORTED,
        Verdict.UNSUPPORTED,
        Verdict.UNSUPPORTED,
    ]
    assert "judge-model" in llm_judge.judge_id
    assert "Claim: c" in llm.calls[0].prompt and EVIDENCE[0] in llm.calls[0].prompt


def labels(pattern: str) -> list[HumanLabel]:
    return [
        HumanLabel(
            f"claim {i}",
            tuple(EVIDENCE),
            Verdict.SUPPORTED if c == "s" else Verdict.UNSUPPORTED,
            "A. Reviewer",
        )
        for i, c in enumerate(pattern)
    ]


class Fixed:
    judge_id = "fixed/1"

    def __init__(self, verdicts: str) -> None:
        self._verdicts = list(verdicts)

    async def judge(self, claim: str, evidence: object) -> Verdict:
        return Verdict.SUPPORTED if self._verdicts.pop(0) == "s" else Verdict.UNSUPPORTED


def test_calibration_requires_ninety_percent_agreement_and_kappa_of_point_eight() -> None:
    human = "ssssssssuu" * 2  # 20 items: 16 supported, 4 unsupported
    perfect = asyncio.run(calibrate_judge(Fixed(human), labels(human)))
    assert perfect.agreement == 1.0 and perfect.kappa == 1.0 and perfect.calibrated
    one_miss = human[:-1] + "s"  # 19 of 20 agree
    good = asyncio.run(calibrate_judge(Fixed(one_miss), labels(human)))
    assert good.agreement == 0.95 and good.kappa >= 0.8 and good.calibrated
    lenient = "s" * 20  # agrees on the 16 supported, misses every unsupported one
    weak = asyncio.run(calibrate_judge(Fixed(lenient), labels(human)))
    assert weak.agreement == 0.8 and weak.kappa == 0.0 and not weak.calibrated


def test_calibration_needs_a_labelled_sample() -> None:
    with pytest.raises(ValueError, match="human-labelled"):
        asyncio.run(calibrate_judge(Fixed(""), []))
    assert RagError is not None
