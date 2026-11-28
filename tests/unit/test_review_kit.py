"""The review kit shows questions to a person and takes back only that person's decisions."""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from rag_engine.domain import RagError
from rag_engine.evaluation.dataset import accepted_questions, parse_dataset
from rag_engine.evaluation.review import COLUMNS
from rag_engine.evaluation.review_kit import (
    SHEET_COLUMNS,
    SourceDataset,
    apply_kit_sheet,
    build_kit,
    load_source,
    question_card,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
HARD = ROOT / "eval" / "datasets" / "retrieval_hard_v1.json"
MINI = ROOT / "eval" / "datasets" / "retrieval_mini_v1.json"


@pytest.fixture(scope="module")
def sources() -> list[SourceDataset]:
    return [load_source(HARD, ROOT), load_source(MINI, ROOT)]


def sheet(sources: list[SourceDataset], decisions: dict[str, dict[str, str]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(SHEET_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for source in sources:
        for q in source.dataset.questions:
            row = dict.fromkeys(SHEET_COLUMNS, "")
            row.update({"dataset": source.name, "id": q.id})
            row.update(decisions.get(f"{source.name}:{q.id}", {}))
            writer.writerow(row)
    return out.getvalue()


def test_the_kit_holds_every_question_of_both_datasets_and_marks_nothing_reviewed(
    sources: list[SourceDataset],
) -> None:
    page = build_kit(sources)
    found = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)
    assert found is not None
    payload = json.loads(found.group(1).replace("<\\/", "</"))
    assert len(payload["cards"]) == 185 == 141 + 44
    assert {c["status"] for c in payload["cards"]} == {"unreviewed"}
    assert len(payload["documents"]) == 26 + 18
    assert "Export decisions (CSV)" in page and "accept" in page and "reject" in page
    for source in sources:  # building the kit changed nothing on disk
        assert accepted_questions(source.dataset) == []


def test_a_card_shows_the_evidence_in_context_or_the_closest_passages_for_a_no_answer(
    sources: list[SourceDataset],
) -> None:
    hard = sources[0]
    answerable = next(q for q in hard.dataset.questions if q.id == "en_db_backup-1-same")
    card = question_card(hard, answerable)
    assert card["evidence"][0]["anchor"] == "every night at 01:15 UTC"
    assert card["evidence"][0]["doc"] == "en_db_backup.md" and card["closest"] == []
    unanswerable = next(q for q in hard.dataset.questions if q.id == "no-answer-3")
    card = question_card(hard, unanswerable)
    assert card["evidence"] == [] and card["absence_terms"] == ["gym"]
    assert isinstance(card["closest"], list)


def test_applying_a_sheet_records_only_the_decisions_under_the_reviewers_name(
    sources: list[SourceDataset], tmp_path: Path
) -> None:
    decisions = {
        "retrieval_hard_v1:en_db_backup-1-same": {"decision": "accept"},
        "retrieval_hard_v1:en_db_backup-2-same": {
            "decision": "edit", "edited_question": "How many days are database backups kept?",
        },
        "retrieval_mini_v1:en-n1": {"decision": "reject", "notes": "ambiguous"},
    }  # fmt: skip
    applied = apply_kit_sheet(
        sources,
        sheet(sources, decisions),
        reviewer="Ada Lovelace",
        outdir=tmp_path,
        on=date(2026, 12, 1),
    )
    by_name = {a.name: a for a in applied}
    hard, mini = by_name["retrieval_hard_v1"], by_name["retrieval_mini_v1"]
    assert (hard.accepted, hard.edited, hard.rejected, hard.undecided) == (1, 1, 0, 139)
    assert (mini.accepted, mini.edited, mini.rejected, mini.undecided) == (0, 0, 1, 43)
    assert hard.version == mini.version == 2 and hard.path.name == "retrieval_hard_v2.json"
    written = json.loads(hard.path.read_text())
    record = next(q for q in written["questions"] if q["id"] == "en_db_backup-1-same")
    assert record["review"] == {
        "status": "accepted",
        "reviewer": "Ada Lovelace",
        "date": "2026-12-01",
        "notes": "",
    }
    edited = next(q for q in written["questions"] if q["id"] == "en_db_backup-2-same")
    assert (
        edited["question"] == "How many days are database backups kept?"
        and edited["review"]["status"] == "edited"
    )
    assert sum(q["review"]["status"] == "unreviewed" for q in written["questions"]) == 139
    assert len(accepted_questions(parse_dataset(written))) == 2  # accepted and edited now count


def test_the_old_versions_are_never_overwritten_and_a_second_apply_is_refused(
    sources: list[SourceDataset], tmp_path: Path
) -> None:
    before = HARD.read_bytes()
    args: dict[str, Any] = {"reviewer": "Ada Lovelace", "outdir": tmp_path, "on": date(2026, 12, 1)}
    apply_kit_sheet(sources, sheet(sources, {}), **args)
    assert HARD.read_bytes() == before
    with pytest.raises(RagError, match="already exists"):
        apply_kit_sheet(sources, sheet(sources, {}), **args)


@pytest.mark.parametrize("reviewer", ["", "  ", "ChatGPT", "claude", "review-bot", "assistant"])
def test_a_program_or_an_empty_name_cannot_review(
    sources: list[SourceDataset], tmp_path: Path, reviewer: str
) -> None:
    decisions = {"retrieval_hard_v1:en_db_backup-1-same": {"decision": "accept"}}
    with pytest.raises(RagError):
        apply_kit_sheet(
            sources,
            sheet(sources, decisions),
            reviewer=reviewer,
            outdir=tmp_path,
            on=date(2026, 12, 1),
        )
    assert not list(tmp_path.glob("*.json"))


def test_unclear_decisions_are_errors_not_guesses(
    sources: list[SourceDataset], tmp_path: Path
) -> None:
    for decision in ("maybe", "reject", "edit"):
        decisions = {"retrieval_hard_v1:en_db_backup-1-same": {"decision": decision}}
        with pytest.raises(RagError):
            apply_kit_sheet(
                sources,
                sheet(sources, decisions),
                reviewer="Ada Lovelace",
                outdir=tmp_path / decision,
                on=date(2026, 12, 1),
            )


def test_a_sheet_for_an_unknown_dataset_or_without_the_column_is_refused(
    sources: list[SourceDataset], tmp_path: Path
) -> None:
    with pytest.raises(RagError, match="dataset"):
        apply_kit_sheet(
            sources,
            ",".join(COLUMNS) + "\n" + "x," * (len(COLUMNS) - 1) + "x\n",
            reviewer="Ada Lovelace",
            outdir=tmp_path,
            on=date(2026, 12, 1),
        )
    bad = sheet(sources, {}).replace("retrieval_mini_v1", "other_dataset", 1)
    with pytest.raises(RagError, match="unknown dataset"):
        apply_kit_sheet(
            sources, bad, reviewer="Ada Lovelace", outdir=tmp_path, on=date(2026, 12, 1)
        )
