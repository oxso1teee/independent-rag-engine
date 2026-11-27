"""Dataset schema v2: parsing, the rules the validator enforces, and metric eligibility."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.evaluation.dataset import (
    Dataset,
    accepted_questions,
    load_dataset,
    parse_dataset,
    validate_dataset,
)

pytestmark = pytest.mark.unit

CORPUS = {
    "docs/backup.md": "Backups run every night at 02:30 UTC and are kept for 14 days.",
    "docs/vacation.md": "Employees receive 28 calendar days of paid vacation per year.",
    "docs/oncall.md": "The on-call engineer hands over the pager every Monday at 10:00.",
}


def question(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "q1",
        "split": "dev",
        "type": "keyword",
        "language": "en",
        "question": "When do the backups run?",
        "answerable": True,
        "gold": [{"doc": "docs/backup.md", "anchor": "every night at 02:30 UTC", "min_support": 1}],
        "reference_answer": "Every night at 02:30 UTC.",
        "reference_facts": ["02:30 UTC"],
        "absence_terms": [],
        "provenance": {
            "created_by": "human",
            "method": "written from the document",
            "created": "2026-11-27",
        },
        "review": {
            "status": "accepted",
            "reviewer": "Ada Reviewer",
            "date": "2026-11-28",
            "notes": "",
        },
    }
    base.update(changes)
    return base


def dataset(*questions: dict[str, Any]) -> dict[str, Any]:
    return {"name": "demo", "version": 1, "corpus": "docs", "questions": list(questions)}


def report(*questions: dict[str, Any]) -> tuple[list[str], list[str]]:
    result = validate_dataset(parse_dataset(dataset(*questions)), CORPUS)
    return list(result.errors), list(result.warnings)


def test_a_valid_dataset_has_no_errors_and_a_stable_hash() -> None:
    data = dataset(question())
    parsed = parse_dataset(data)
    assert isinstance(parsed, Dataset) and parsed.identifier == "demo@v1"
    assert validate_dataset(parsed, CORPUS).errors == ()
    assert parsed.manifest_hash == parse_dataset(copy.deepcopy(data)).manifest_hash
    other = copy.deepcopy(data)
    other["questions"][0]["question"] = "Different?"
    assert parse_dataset(other).manifest_hash != parsed.manifest_hash


@pytest.mark.parametrize(
    "broken",
    [
        {"type": "trivia"},
        {"split": "train"},
        {"language": "de"},
        {"question": "   "},
        {"gold": [{"doc": "docs/backup.md"}]},
        {"review": {"status": "approved"}},
        {"id": ""},
    ],
)
def test_malformed_records_are_refused_when_parsing(broken: dict[str, Any]) -> None:
    with pytest.raises(RagError) as info:
        parse_dataset(dataset(question(**broken)))
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


def test_unknown_top_level_and_record_keys_are_refused() -> None:
    data = dataset(question())
    data["extra"] = 1
    with pytest.raises(RagError):
        parse_dataset(data)
    with pytest.raises(RagError):
        parse_dataset(dataset({**question(), "surprise": True}))


def test_duplicate_ids_are_an_error() -> None:
    errors, _ = report(question(), question())
    assert any("duplicate question id" in item for item in errors)


def test_an_anchor_must_be_a_verbatim_quote_of_the_document() -> None:
    gold = [{"doc": "docs/backup.md", "anchor": "every night at 03:30 UTC", "min_support": 1}]
    errors, _ = report(question(gold=gold))
    assert any("anchor" in item and "q1" in item for item in errors)


def test_anchor_matching_ignores_case_and_runs_of_whitespace() -> None:
    gold = [{"doc": "docs/backup.md", "anchor": "Every  NIGHT\nat 02:30 UTC", "min_support": 1}]
    assert report(question(gold=gold))[0] == []


def test_gold_documents_must_exist_in_the_corpus() -> None:
    gold = [{"doc": "docs/missing.md", "anchor": "x", "min_support": 1}]
    errors, _ = report(question(gold=gold))
    assert any("not in the corpus" in item for item in errors)


def test_answerable_questions_need_gold_and_unanswerable_ones_need_absence_terms() -> None:
    assert any("needs gold" in item for item in report(question(gold=[]))[0])
    no_answer = question(
        id="q2", type="no_answer", answerable=False, gold=[], absence_terms=["bitcoin"]
    )
    assert report(no_answer)[0] == []
    assert any("absence_terms" in item for item in report({**no_answer, "absence_terms": []})[0])
    assert any("gold" in item for item in report({**no_answer, "gold": question()["gold"]})[0])
    assert any("type" in item for item in report({**no_answer, "type": "keyword"})[0])


def test_an_absence_term_that_the_corpus_contains_is_an_error() -> None:
    leaky = question(
        id="q2", type="no_answer", answerable=False, gold=[], absence_terms=["Vacation"]
    )
    assert any("appears in the corpus" in item for item in report(leaky)[0])


def test_multi_doc_needs_two_gold_documents() -> None:
    assert any("two" in item for item in report(question(type="multi_doc"))[0])
    both = [
        {"doc": "docs/backup.md", "anchor": "every night", "min_support": 1},
        {"doc": "docs/vacation.md", "anchor": "28 calendar days", "min_support": 1},
    ]
    assert report(question(type="multi_doc", gold=both))[0] == []


def test_questions_sharing_a_gold_document_must_share_a_split() -> None:
    first = question()
    second = question(id="q2", split="test", question="How long are backups kept?")
    errors, _ = report(first, second)
    assert any("split" in item for item in errors)
    separate = question(
        id="q3",
        split="test",
        gold=[{"doc": "docs/oncall.md", "anchor": "every Monday at 10:00", "min_support": 1}],
    )
    assert report(first, separate)[0] == []


def test_an_accepted_review_needs_a_human_reviewer_and_a_date() -> None:
    for review in (
        {"status": "accepted", "reviewer": "", "date": "2026-11-28", "notes": ""},
        {"status": "accepted", "reviewer": "Ada", "date": "", "notes": ""},
        {"status": "edited", "reviewer": "gpt-review-bot", "date": "2026-11-28", "notes": ""},
        {"status": "accepted", "reviewer": "claude", "date": "2026-11-28", "notes": ""},
    ):
        assert report(question(review=review))[0], review
    unreviewed = {"status": "unreviewed", "reviewer": None, "date": None, "notes": ""}
    assert report(question(review=unreviewed))[0] == []


def test_only_accepted_and_edited_questions_count_for_metrics() -> None:
    reviews = {
        "q1": "accepted",
        "q2": "edited",
        "q3": "unreviewed",
        "q4": "rejected",
    }
    items = []
    for identifier, status in reviews.items():
        reviewer = "Ada" if status in {"accepted", "edited"} else None
        date = "2026-11-28" if reviewer else None
        items.append(
            question(
                id=identifier,
                review={"status": status, "reviewer": reviewer, "date": date, "notes": ""},
            )
        )
    kept = accepted_questions(parse_dataset(dataset(*items)))
    assert [item.id for item in kept] == ["q1", "q2"]


def test_a_paraphrase_that_repeats_its_anchor_is_a_warning_and_a_small_set_is_anecdotal() -> None:
    echo = question(
        type="paraphrase",
        question="every night at 02:30 UTC",
        gold=[{"doc": "docs/backup.md", "anchor": "every night at 02:30 UTC", "min_support": 1}],
    )
    errors, warnings = report(echo)
    assert errors == []
    assert any("overlap" in item for item in warnings)
    assert any("anecdotal" in item for item in warnings)


def test_load_dataset_reads_a_json_file_and_reports_bad_files(tmp_path: Path) -> None:
    path = tmp_path / "set.json"
    path.write_text(json.dumps(dataset(question())), encoding="utf-8")
    assert load_dataset(path).identifier == "demo@v1"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RagError) as info:
        load_dataset(path)
    assert info.value.code is ErrorCode.E_CONFIG_INVALID
    with pytest.raises(RagError):
        load_dataset(tmp_path / "absent.json")
