"""Version 2 of the retrieval datasets: the corrections, their history, and the guards.

The first hard dataset carried two multi-document questions (``multi_doc-3``, ``multi_doc-5``) with
a gold quote that had nothing to do with the question, added only to satisfy "a multi-document
question needs two documents", plus ambiguous cross-language questions, answers that dropped part
of the evidence and absence terms given in one language only. Version 2 fixes them; these tests pin
the frozen version 1, the exact set of changes, and the generator's guard against a repeat.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

from rag_engine.evaluation.dataset import load_dataset, validate_dataset
from rag_engine.evaluation.question_audit import Verdict, audit_source
from rag_engine.evaluation.review_kit import load_source

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "eval" / "datasets"
FROZEN = {
    "retrieval_hard_v1.json": "e0d0998f2e6c83b2a0f3c1d13f74039d0ba9bff31011c07e53a72349d74d0569",
    "retrieval_mini_v1.json": "b99073fbc1bb1da226ca8a8ce48c2c67b506df4edb8a42a7a7d26df840f7998c",
}
PAIRS = {"retrieval-hard": ("retrieval_hard_v1", "retrieval_hard_v2"),
         "retrieval-mini": ("retrieval_mini_v1", "retrieval_mini_v2")}  # fmt: skip


def script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(ROOT / "eval" / "scripts"))
    spec.loader.exec_module(module)
    return module


def read(name: str) -> dict[str, Any]:
    value = json.loads((DATASETS / f"{name}.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


# ------------------------------------------------------------------------- the frozen version


def test_version_1_is_frozen_byte_for_byte() -> None:
    for name, digest in FROZEN.items():
        assert hashlib.sha256((DATASETS / name).read_bytes()).hexdigest() == digest, (
            f"{name} was edited; a correction is a new version, never a change of a frozen one"
        )


# ------------------------------------------------------------------------------ the history


def test_the_recorded_changes_are_exactly_what_differs_between_the_versions() -> None:
    history = read("CHANGES_v2")
    assert set(history["datasets"]) == set(PAIRS)
    for dataset, (old_name, new_name) in PAIRS.items():
        old = {q["id"]: q for q in read(old_name)["questions"]}
        new = {q["id"]: q for q in read(new_name)["questions"]}
        assert old.keys() == new.keys(), "no question was added or removed"
        actual = {
            i: sorted(
                k for k in new[i] if k not in ("provenance", "review") and new[i][k] != old[i][k]
            )
            for i in new
        }
        actual = {i: fields for i, fields in actual.items() if fields}
        declared = {c["id"]: sorted(c["fields"]) for c in history["datasets"][dataset]["changes"]}
        assert actual == declared
        for change in history["datasets"][dataset]["changes"]:
            assert change["reason"].strip() and change["kind"] in {
                "gold", "anchor", "ambiguity", "completeness", "absence", "facts",
            }  # fmt: skip
            assert change["before"] is not None and change["after"] is not None
            assert change["before"] != change["after"]


def test_untouched_questions_are_identical_and_changed_ones_say_so_in_their_provenance() -> None:
    history = read("CHANGES_v2")
    for dataset, (old_name, new_name) in PAIRS.items():
        old = {q["id"]: q for q in read(old_name)["questions"]}
        changed = {c["id"] for c in history["datasets"][dataset]["changes"]}
        for question in read(new_name)["questions"]:
            before = old[question["id"]]
            if question["id"] in changed:
                assert question["provenance"]["method"].startswith("revised in v2 (")
                assert question["provenance"]["created_by"] == before["provenance"]["created_by"]
            else:
                assert question == before


def test_nothing_in_version_2_is_reviewed_and_the_corpus_is_unchanged() -> None:
    for _, (_, new_name) in PAIRS.items():
        questions = read(new_name)["questions"]
        assert {q["review"]["status"] for q in questions} == {"unreviewed"}
        assert all(
            q["review"]["reviewer"] is None and q["review"]["date"] is None for q in questions
        )
        assert read(new_name)["version"] == 2


# --------------------------------------------------------------------- the errors found in v1


def test_the_two_multi_document_questions_now_need_two_real_documents() -> None:
    questions = {q["id"]: q for q in read("retrieval_hard_v2")["questions"]}
    for identifier in ("multi_doc-3", "multi_doc-5"):
        q = questions[identifier]
        assert len({g["doc"] for g in q["gold"]}) == 2 and len(q["reference_facts"]) == 2
    texts = {
        i: json.dumps(questions[i], ensure_ascii=False) for i in ("multi_doc-3", "multi_doc-5")
    }
    assert "ru_office_hours" not in texts["multi_doc-3"]  # the unrelated filler is gone
    assert "каждый четверг" not in texts["multi_doc-5"]
    old = {q["id"]: q for q in read("retrieval_hard_v1")["questions"]}
    assert len({g["doc"] for g in old["multi_doc-3"]["gold"]}) == 2  # why the validator missed it
    assert len(old["multi_doc-3"]["gold"]) == 3


def test_the_audit_finds_the_original_error_and_nothing_wrong_remains() -> None:
    old = audit_source(load_source(DATASETS / "retrieval_hard_v1.json", ROOT))
    wrong = {a.key.split(":", 1)[1] for a in old if a.verdict is Verdict.INCORRECT}
    assert wrong == {"multi_doc-3", "multi_doc-5"}
    for name in ("retrieval_hard_v2", "retrieval_mini_v2"):
        verdicts = [a.verdict for a in audit_source(load_source(DATASETS / f"{name}.json", ROOT))]
        assert Verdict.INCORRECT not in verdicts


def test_every_unanswerable_question_has_absence_terms_in_both_languages() -> None:
    for name in ("retrieval_hard_v2", "retrieval_mini_v2"):
        for q in read(name)["questions"]:
            if not q["answerable"]:
                scripts = {
                    "ru" if re.search("[а-яё]", t, re.I) else "en" for t in q["absence_terms"]
                }
                assert scripts == {"ru", "en"}, f"{name}:{q['id']} has terms in one language only"


def test_a_multi_document_question_lists_one_fact_per_evidence_quote() -> None:
    for name in ("retrieval_hard_v2", "retrieval_mini_v2"):
        for q in read(name)["questions"]:
            if q["type"] in {"multi_doc", "conflicting"}:
                assert len(q["reference_facts"]) >= len(q["gold"]), f"{name}:{q['id']}"


def test_a_conflicting_answer_says_which_version_replaces_which() -> None:
    questions = {q["id"]: q for q in read("retrieval_hard_v2")["questions"]}
    for identifier in ("conflicting-6", "conflicting-7", "conflicting-8", "conflicting-9"):
        assert "replaces" in questions[identifier]["reference_answer"]


def test_the_versions_validate_against_their_corpora() -> None:
    for name in ("retrieval_hard_v2", "retrieval_mini_v2"):
        dataset = load_dataset(DATASETS / f"{name}.json")
        corpus = {
            p.name: p.read_text(encoding="utf-8") for p in (ROOT / dataset.corpus).glob("*.md")
        }
        assert validate_dataset(dataset, corpus).ok


# ------------------------------------------------------------------------- the generator guard


def test_the_generator_refuses_a_gold_quote_that_has_nothing_to_do_with_the_question() -> None:
    generator = script("build_hard_dataset")
    corpus, dataset, _, _ = generator.produce()
    broken = copy.deepcopy(dataset)
    target = next(q for q in broken["questions"] if q["id"] == "multi_doc-3")
    target["gold"].append(
        {"doc": "ru_office_hours.md", "anchor": "с 9:00 до 18:00", "min_support": 1}
    )
    target["reference_facts"].append("с 9:00 до 18:00")
    with pytest.raises(SystemExit, match="multi_doc-3"):
        generator.check(broken, corpus)


def test_the_generator_refuses_a_figure_that_is_in_no_gold_document() -> None:
    generator = script("build_hard_dataset")
    corpus, dataset, _, _ = generator.produce()
    broken = copy.deepcopy(dataset)
    next(q for q in broken["questions"] if q["id"] == "en_vpn-2-same")["reference_answer"] = (
        "A lost token must be reported within 9 hours."
    )
    with pytest.raises(SystemExit, match="en_vpn-2-same"):
        generator.check(broken, corpus)


# ---------------------------------------------------------------------------- the revision tool


def test_a_revision_needs_a_reason_for_every_change_and_changes_no_review() -> None:
    revision = script("dataset_revision")
    previous: dict[str, Any] = {"version": 1, "questions": [
        {"id": "a", "text": "old", "provenance": {"method": "m", "created": "d"},
         "review": {"status": "unreviewed", "reviewer": None, "date": None, "notes": ""}},
        {"id": "b", "text": "same", "provenance": {"method": "m", "created": "d"},
         "review": {"status": "unreviewed", "reviewer": None, "date": None, "notes": ""}},
    ]}  # fmt: skip

    def current() -> dict[str, Any]:
        data = copy.deepcopy(previous)
        data["version"] = 2
        data["questions"][0]["text"] = "new"
        return data

    with pytest.raises(SystemExit, match="without a stated reason"):
        revision.revise(previous, current(), {})
    with pytest.raises(SystemExit, match="did not change"):
        revision.revise(previous, current(), {"a": ("wording", "x"), "b": ("wording", "y")})
    data = current()
    changes = revision.revise(previous, data, {"a": ("wording", "clearer")})
    assert [c["id"] for c in changes] == ["a"] and changes[0]["after"] == {"text": "new"}
    assert data["questions"][0]["provenance"]["method"].startswith("revised in v2 (wording)")
    assert data["questions"][1] == previous["questions"][1]
    assert {q["review"]["status"] for q in data["questions"]} == {"unreviewed"}
