"""The automatic question audit: three verdicts, honest flags, and no effect on any dataset."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rag_engine.evaluation.dataset import parse_dataset
from rag_engine.evaluation.question_audit import QuestionAudit, Verdict, audit_source, summarize
from rag_engine.evaluation.review_kit import SourceDataset, build_kit, load_source
from rag_engine.testing import HashingEmbedder

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

CORPUS = {
    "en_backup.md": "# Backups\n\nProduction database backups run every night at 01:15 UTC. "
    "Backups are encrypted and kept for 30 days.\n\n# Laptops\n\nLaptop backups run every Friday "
    "at 18:00 local time and are kept for 7 days.\n",
    "en_meals.md": "# Meals\n\nThe meal allowance is 25 euro per day when travelling abroad.\n",
    "en_parking.md": "# Parking\n\nVisitors park in the north lot; employees need a badge.\n",
    "ru_mail.md": "# Почта\n\nКопии почтового сервера создаются каждую ночь в 03:20 "
    "и хранятся 21 день.\n",
    "ru_office.md": "# Офис\n\nОфис открыт с понедельника по пятницу с 9:00 до 18:00.\n",
}


def question(identifier: str, **changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": identifier, "split": "dev", "type": "keyword", "language": "en",
        "question": "When do the production database backups run?", "answerable": True,
        "gold": [{"doc": "en_backup.md", "anchor": "every night at 01:15 UTC", "min_support": 1}],
        "reference_answer": "Production database backups run every night at 01:15 UTC.",
        "reference_facts": ["every night at 01:15 UTC"], "absence_terms": [],
        "provenance": {"created_by": "llm-assisted", "method": "test", "created": "2026-11-28"},
        "review": {"status": "unreviewed", "reviewer": None, "date": None, "notes": ""},
    }  # fmt: skip
    return {**base, **changes}


def unanswerable(identifier: str, text: str, terms: list[str]) -> dict[str, Any]:
    return question(
        identifier, answerable=False, type="no_answer", gold=[], reference_answer=None,
        reference_facts=[], question=text, absence_terms=terms,
    )  # fmt: skip


def source(*questions: dict[str, Any]) -> SourceDataset:
    raw = {"name": "t", "version": 1, "corpus": "x", "questions": list(questions)}
    return SourceDataset("t_v1", raw, parse_dataset(raw), CORPUS)


def audit(*questions: dict[str, Any], encoder: Any = None) -> dict[str, QuestionAudit]:
    return {a.key.split(":", 1)[1]: a for a in audit_source(source(*questions), encoder)}


def flags(a: QuestionAudit, severity: str | None = None) -> set[str]:
    return {f.id for f in a.flags if severity is None or f.severity == severity}


class HashingEncoder:
    """A real, deterministic encoder: texts that share words are close."""

    def __init__(self) -> None:
        self._e = HashingEmbedder(256)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        return asyncio.run(self._e.embed_documents(list(texts))).tolist()  # type: ignore[no-any-return]

    def queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)


# --------------------------------------------------------------------------------- verdicts


def test_a_question_linked_to_its_evidence_with_nothing_against_it_is_correct() -> None:
    result = audit(question("ok"))["ok"]
    assert (
        result.verdict is Verdict.CORRECT
        and not flags(result, "hard")
        and not flags(result, "soft")
    )
    assert "lexical-link" in result.passed and "figures-in-evidence" in result.passed


def test_an_evidence_quote_unrelated_to_the_question_makes_it_incorrect() -> None:
    padded = question(
        "multi", type="multi_doc", language="ru",
        question="Сколько хранятся копии почтового сервера и когда их создают?",
        gold=[
            {"doc": "ru_mail.md", "anchor": "хранятся 21 день", "min_support": 1},
            {"doc": "ru_mail.md", "anchor": "каждую ночь в 03:20", "min_support": 1},
            {"doc": "ru_office.md", "anchor": "с 9:00 до 18:00", "min_support": 1},
        ],
        reference_answer="Копии хранятся 21 день и создаются каждую ночь в 03:20.",
        reference_facts=["хранятся 21 день", "каждую ночь в 03:20"],
    )  # fmt: skip
    result = audit(padded)["multi"]
    assert result.verdict is Verdict.INCORRECT and "unrelated-evidence" in flags(result, "hard")
    assert "ru_office.md" in result.hint


def test_a_figure_that_no_gold_document_contains_is_incorrect() -> None:
    result = audit(question("fig", reference_answer="Backups run every night at 02:45 UTC."))["fig"]
    assert result.verdict is Verdict.INCORRECT and "figure-not-in-documents" in flags(
        result, "hard"
    )
    outside = audit(
        question(
            "out", reference_answer="Backups run every night at 01:15 UTC and laptops keep 7 days."
        )
    )["out"]
    assert (
        "figure-outside-evidence" in flags(outside, "soft") and outside.verdict is Verdict.DOUBTFUL
    )


def test_structural_contradictions_are_incorrect() -> None:
    bad = audit(
        question("nogold", gold=[]),
        question("yesgold", answerable=False, type="no_answer", absence_terms=["zebra"]),
        question("onedoc", type="multi_doc"),
        question("lang", language="ru"),
        question("noans", reference_answer=""),
    )  # fmt: skip
    assert "no-gold" in flags(bad["nogold"], "hard")
    assert "gold-on-unanswerable" in flags(bad["yesgold"], "hard")
    assert "one-document" in flags(bad["onedoc"], "hard")
    assert "language-label" in flags(bad["lang"], "hard")
    assert "no-answer-text" in flags(bad["noans"], "hard")
    assert all(a.verdict is Verdict.INCORRECT for a in bad.values())


def test_an_absent_term_in_the_corpus_is_incorrect_and_a_missing_reason_doubtful() -> None:
    present = audit(unanswerable("p", "Which lot do visitors use?", ["visitors"]))["p"]
    assert "absence-term-present" in flags(present, "hard")
    bare = audit(unanswerable("b", "What is the dental plan?", []))["b"]
    assert "no-absence-terms" in flags(bare, "soft") and bare.verdict is Verdict.DOUBTFUL


def test_an_unanswerable_question_that_a_passage_nearly_answers_is_doubtful() -> None:
    near = audit(unanswerable("n", "What is the meal allowance per day abroad?", ["dental"]))["n"]
    assert "looks-answerable" in flags(near, "soft")
    far = audit(unanswerable("f", "What is the dental plan?", ["dental"]))["f"]
    assert far.verdict is Verdict.CORRECT and "no-passage-covers-the-question" in far.passed


def test_a_link_that_cannot_be_shown_is_doubtful_not_correct() -> None:
    text = "В какое время создаются резервные копии производственной базы?"
    cross = question("x", language="ru", type="paraphrase", question=text)
    result = audit(cross)["x"]
    assert result.verdict is Verdict.DOUBTFUL and "link-unverified" in flags(result, "soft")
    assert "no semantic layer" in " ".join(f.message for f in result.flags)


def test_a_better_matching_passage_and_near_duplicates_are_flagged() -> None:
    worse = question(
        "w", question="How long are laptop backups kept after they run on Friday?",
        gold=[{"doc": "en_backup.md", "anchor": "kept for 30 days", "min_support": 1}],
        reference_answer="Backups are kept for 30 days.", reference_facts=["kept for 30 days"],
    )  # fmt: skip
    assert "other-passage-fits-better" in flags(audit(worse)["w"], "soft")
    twins = audit(
        question("a"), question("b", question="When do the production database backups run??")
    )
    assert "near-duplicate" in flags(twins["a"], "soft") and twins["a"].verdict is Verdict.DOUBTFUL


def test_a_multi_document_answer_that_drops_evidence_is_incomplete() -> None:
    q = question(
        "m", type="multi_doc", question="How long are database and laptop backups kept?",
        gold=[{"doc": "en_backup.md", "anchor": "kept for 30 days", "min_support": 1},
              {"doc": "en_backup.md", "anchor": "kept for 7 days", "min_support": 1},
              {"doc": "en_meals.md", "anchor": "25 euro per day", "min_support": 1}],
        reference_answer="Database backups are kept for 30 days.",
        reference_facts=["kept for 30 days"],
    )  # fmt: skip
    result = audit(q)["m"]
    assert {"incomplete", "few-facts"} <= flags(result, "soft")


# --------------------------------------------------------------------------- semantic layer


def many(n: int) -> list[dict[str, Any]]:
    """Enough answerable questions for the semantic baseline, each about its own document."""
    return [question(f"q{i}") for i in range(n)]


def test_the_semantic_layer_confirms_a_gold_passage_that_ranks_first() -> None:
    result = audit(question("ok"), encoder=HashingEncoder())["ok"]
    assert result.verdict is Verdict.CORRECT and "dense-gold-first" in result.passed
    assert result.metrics["dense_gold_rank"] == 1


def test_the_semantic_layer_doubts_evidence_that_ranks_low() -> None:
    wrong = question(
        "far", question="Where do visitors park and what badge do employees need?",
        gold=[{"doc": "en_backup.md", "anchor": "kept for 7 days", "min_support": 1}],
        reference_answer="Laptop backups are kept for 7 days.", reference_facts=["kept for 7 days"],
    )  # fmt: skip
    result = audit(wrong, encoder=HashingEncoder())["far"]
    assert result.verdict is Verdict.DOUBTFUL
    assert flags(result, "soft") & {"dense-gold-not-first", "dense-gold-far"}


# --------------------------------------------------------------- it changes and reviews nothing


def test_the_audit_never_touches_the_dataset_or_marks_anything_reviewed() -> None:
    src = source(question("a"), unanswerable("b", "What is the dental plan?", ["dental"]))
    before = copy.deepcopy(dict(src.raw))
    results = audit_source(src)
    assert dict(src.raw) == before
    assert {q.review.status for q in src.dataset.questions} == {"unreviewed"}
    assert summarize(results)["questions"] == 2
    assert set(summarize(results)["by_verdict"]) == {"correct", "doubtful", "incorrect"}


def test_the_page_carries_audits_as_hints_and_no_prefilled_decision() -> None:
    src = source(question("a"))
    audits = {a.key: a.as_dict() for a in audit_source(src)}
    page = build_kit([src], audits=audits)
    found = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)
    assert found is not None
    payload = json.loads(found.group(1).replace("<\\/", "</"))
    assert payload["cards"][0]["audit"]["verdict"] == "correct"
    assert payload["columns"][-1] == "audit" and "decision" in payload["columns"]
    assert all("decision" not in c for c in payload["cards"])  # nothing is decided for the person
    assert {c["status"] for c in payload["cards"]} == {"unreviewed"}
    for control in ("b-accept", "b-sel-visible", "b-open", "view-table"):
        assert f'id="{control}"' in page
    assert "Type the number" in page  # a bulk accept asks for the count


# ----------------------------------------------------------------- the shipped datasets


def test_the_shipped_datasets_audit_cleanly_and_the_known_bad_ones_are_found() -> None:
    hard = load_source(ROOT / "eval/datasets/retrieval_hard_v1.json", ROOT)
    by_id = {a.key.split(":", 1)[1]: a for a in audit_source(hard)}
    assert len(by_id) == 141
    assert by_id["multi_doc-3"].verdict is Verdict.INCORRECT
    assert by_id["multi_doc-5"].verdict is Verdict.INCORRECT
    assert by_id["multi_doc-1"].verdict is not Verdict.INCORRECT
