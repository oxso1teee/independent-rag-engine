"""The citation validator: markers must exist in the context and the text must support them."""

from __future__ import annotations

import pytest

from rag_engine.domain import (
    Context,
)
from rag_engine.domain.answer import CitationProblem
from rag_engine.generation.citations import CitationValidator, extract_markers, split_sentences
from rag_engine.testing import evidence_context
from tests.factories import make_ids

pytestmark = pytest.mark.unit

IDS = make_ids()
VERSION = IDS.document_version_id()
DOC = IDS.document_id()

BACKUPS = "Backups run every night at 02:30 UTC and are kept for 14 days."
PRINTERS = "Printers are on the third floor next to the kitchen."
RUSSIAN = "Резервные копии создаются ежедневно в три часа ночи и хранятся четырнадцать суток."


def make_context(*texts: str) -> Context:
    return evidence_context(texts, document_id=DOC, version_id=VERSION)


VALIDATOR = CitationValidator()


def test_markers_are_extracted_in_order_without_repeats() -> None:
    assert extract_markers("A [E2]. B [E1][E2]. C [E10].") == ("E2", "E1", "E10")
    assert extract_markers("no markers, [E0] and [Ex] and [e1] do not count") == ()


def test_a_marker_after_the_full_stop_stays_with_its_sentence() -> None:
    parts = split_sentences("Backups run nightly. [E1] They are kept 14 days [E1]. Done")
    assert parts == ["Backups run nightly. [E1]", "They are kept 14 days [E1].", "Done"]


def test_a_supported_cited_answer_is_valid() -> None:
    context = make_context(BACKUPS, PRINTERS)
    check = VALIDATOR.validate("Backups run every night at 02:30 UTC [E1].", context)
    assert check.ok and check.cited == ("E1",) and not check.unknown
    assert check.sentences[0].overlap_ok and check.sentences[0].coverage == pytest.approx(1.0)


def test_a_marker_that_names_no_evidence_is_invalid() -> None:
    context = make_context(BACKUPS)
    check = VALIDATOR.validate("Backups run every night [E1] and are mirrored [E7].", context)
    assert not check.ok
    assert check.unknown == ("E7",) and CitationProblem.UNKNOWN_MARKER in check.problems


def test_an_answer_without_any_citation_is_invalid() -> None:
    check = VALIDATOR.validate("Backups run every night.", make_context(BACKUPS))
    assert check.problems == (CitationProblem.NO_CITATION,)
    assert check.uncited_sentences == ("Backups run every night.",)


def test_a_sentence_that_the_cited_passage_does_not_support_has_low_overlap() -> None:
    context = make_context(BACKUPS, PRINTERS)
    check = VALIDATOR.validate("The printers are encrypted with a hardware token [E1].", context)
    assert CitationProblem.LOW_OVERLAP in check.problems
    assert check.sentences[0].coverage is not None and check.sentences[0].coverage < 0.5


def test_the_right_marker_on_the_wrong_passage_is_caught() -> None:
    context = make_context(BACKUPS, PRINTERS)
    wrong = VALIDATOR.validate("Backups are kept for 14 days [E2].", context)
    right = VALIDATOR.validate("Backups are kept for 14 days [E1].", context)
    assert not wrong.ok and right.ok


def test_several_markers_pool_their_evidence() -> None:
    context = make_context(BACKUPS, PRINTERS)
    check = VALIDATOR.validate(
        "Backups run at 02:30 UTC and printers are on the third floor [E1][E2].", context
    )
    assert check.ok and check.cited == ("E1", "E2")


def test_uncited_sentences_are_reported_but_do_not_fail_the_answer() -> None:
    context = make_context(BACKUPS)
    check = VALIDATOR.validate(
        "Short answer below. Backups run every night at 02:30 UTC [E1].", context
    )
    assert check.ok and check.uncited_sentences == ("Short answer below.",)


def test_a_sentence_in_another_script_than_its_evidence_is_not_judged() -> None:
    context = make_context(BACKUPS)
    check = VALIDATOR.validate("Резервные копии создаются каждую ночь [E1].", context)
    assert check.ok and check.unjudged == 1 and check.sentences[0].coverage is None


def test_russian_answers_are_checked_with_russian_stemming() -> None:
    context = make_context(RUSSIAN)
    good = VALIDATOR.validate(
        "Резервная копия создается ежедневно, хранится четырнадцать суток [E1].", context
    )
    bad = VALIDATOR.validate("Принтеры расположены на третьем этаже рядом с кухней [E1].", context)
    assert good.ok and not bad.ok


def test_a_sentence_with_too_few_terms_is_not_judged() -> None:
    check = VALIDATOR.validate("Yes [E1].", make_context(BACKUPS))
    assert check.ok and check.unjudged == 1


def test_the_threshold_is_configurable_and_validated() -> None:
    context = make_context(BACKUPS)
    text = "Backups run every night and are encrypted with a hardware token [E1]."
    assert not CitationValidator(min_coverage=0.9).validate(text, context).ok
    assert CitationValidator(min_coverage=0.3).validate(text, context).ok
    with pytest.raises(ValueError, match="min_coverage"):
        CitationValidator(min_coverage=1.5)


def test_the_check_is_deterministic() -> None:
    context = make_context(BACKUPS, PRINTERS)
    text = "Backups run every night [E1]. Printers are on the third floor [E2]."
    assert VALIDATOR.validate(text, context) == VALIDATOR.validate(text, context)
