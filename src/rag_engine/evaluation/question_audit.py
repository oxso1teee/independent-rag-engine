# ruff: noqa: E501  (the flag messages are sentences)
"""A preliminary, automatic audit of evaluation questions, to guide a person and never to replace one.

Every question is checked against its own documents and sorted into one of three groups:

* ``correct``: nothing found against it and its evidence is linked to the question;
* ``doubtful``: something a person should look at (another passage fits as well, the answer
  contains words the evidence does not, the link between question and evidence could not be shown);
* ``incorrect``: something is wrong on its face (a figure in the expected answer that no gold
  document contains, an answerable question without gold, a "missing" term that is in the corpus).

Two layers of checks. The **lexical** layer needs nothing but the corpus: it compares stems, figures
and identifiers, and it is blind across languages and for paraphrases by design. The **semantic**
layer ranks every passage of the corpus for the question with an embedding model supplied by the
caller, and is what can say that the gold passage is, or is not, the best match; without it a
question that cannot be linked lexically is ``doubtful`` rather than ``correct``. Neither layer
understands the question. A ``correct`` verdict means "no automatic objection", not "reviewed":
nothing here writes to a dataset or marks a question as reviewed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from rag_engine.domain import Language, detect_language
from rag_engine.evaluation.dataset import Question
from rag_engine.evaluation.review_kit import SourceDataset
from rag_engine.lexical.analysis import analyze

_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_CYRILLIC = re.compile(r"[Ѐ-ԯ]")
_LATIN = re.compile(r"[A-Za-z]")

SAME_LANGUAGE_LINK = 0.25
"""Idf-weighted share of the question's terms found in its gold passage that counts as a link."""
OTHER_PASSAGE_MARGIN = 0.10
UNSUPPORTED_WORDS = 0.5
NEAR_NO_ANSWER_COVERAGE = 0.75
DENSE_MARGIN = 0.01
NEAR_DUPLICATE = 0.85
SHORT_ANCHOR = 12


class Verdict(StrEnum):
    CORRECT = "correct"
    DOUBTFUL = "doubtful"
    INCORRECT = "incorrect"


@dataclass(frozen=True, slots=True)
class Flag:
    id: str
    severity: str  # "hard" makes a question incorrect, "soft" doubtful, "info" changes nothing
    message: str


@dataclass(frozen=True, slots=True)
class QuestionAudit:
    key: str
    verdict: Verdict
    flags: tuple[Flag, ...]
    passed: tuple[str, ...]
    metrics: Mapping[str, float | int | str | None]
    hint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "verdict": self.verdict.value,
            "flags": [
                {"id": f.id, "severity": f.severity, "message": f.message} for f in self.flags
            ],
            "passed": list(self.passed),
            "metrics": dict(self.metrics),
            "hint": self.hint,
        }


class Encoder(Protocol):
    """Embeds texts for the semantic layer; rows are unit vectors."""

    def queries(self, texts: Sequence[str]) -> list[list[float]]: ...
    def passages(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class Passage:
    doc: str
    index: int
    text: str
    language: Language
    terms: frozenset[str]


# ------------------------------------------------------------------------------------ helpers


def _terms(text: str) -> list[str]:
    return [t for t in analyze(text, Language.MIXED).split() if len(t) >= 3 or t.isdigit()]


def _numbers(text: str) -> set[str]:
    return {n.rstrip(".,:") for n in _NUMBER.findall(text)}


def _fold(text: str) -> str:
    return " ".join(text.casefold().split())


def _script(text: str) -> str:
    cyrillic, latin = len(_CYRILLIC.findall(text)), len(_LATIN.findall(text))
    if cyrillic + latin == 0:
        return "none"
    return "ru" if cyrillic >= latin else "en"


def split_passages(corpus: Mapping[str, str]) -> list[Passage]:
    """Paragraphs of every document, in order; a heading is kept with the paragraph after it."""
    out: list[Passage] = []
    for doc, text in corpus.items():
        pending = ""
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if not block:
                continue
            if block.startswith("#") and "\n" not in block:
                pending = block.lstrip("# ").strip()
                continue
            body = f"{pending}. {block}" if pending else block
            pending = ""
            out.append(Passage(doc, len(out), body, detect_language(body), frozenset(_terms(body))))
        if pending:
            out.append(
                Passage(
                    doc, len(out), pending, detect_language(pending), frozenset(_terms(pending))
                )
            )
    return out


def _idf(passages: Sequence[Passage]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for passage in passages:
        df.update(passage.terms)
    n = max(len(passages), 1)
    return {t: math.log(1 + n / c) for t, c in df.items()}


def _coverage(wanted: Sequence[str], passage: Passage, idf: Mapping[str, float]) -> float:
    unique = set(wanted)
    total = sum(idf.get(t, 1.0) for t in unique)
    if total == 0:
        return 0.0
    return sum(idf.get(t, 1.0) for t in unique if t in passage.terms) / total


def _passage_of(doc: str, anchor: str, passages: Sequence[Passage]) -> Passage | None:
    folded = _fold(anchor)
    return next((p for p in passages if p.doc == doc and folded in _fold(p.text)), None)


def _gold_passages(question: Question, passages: Sequence[Passage]) -> list[Passage]:
    found = (_passage_of(ref.doc, ref.anchor, passages) for ref in question.gold)
    return [p for p in found if p is not None]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# -------------------------------------------------------------------------------------- audit


@dataclass
class _Run:
    question: Question
    flags: list[Flag] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)

    def hard(self, id_: str, message: str) -> None:
        self.flags.append(Flag(id_, "hard", message))

    def soft(self, id_: str, message: str) -> None:
        self.flags.append(Flag(id_, "soft", message))

    def info(self, id_: str, message: str) -> None:
        self.flags.append(Flag(id_, "info", message))


def audit_source(source: SourceDataset, encoder: Encoder | None = None) -> list[QuestionAudit]:
    """Audit every question of ``source``; ``encoder`` switches the semantic layer on."""
    passages = split_passages(source.corpus)
    idf = _idf(passages)
    questions = list(source.dataset.questions)
    dense = _dense_scores(questions, passages, encoder) if encoder is not None else None
    typical_gold = _typical_gold_cosine(questions, passages, dense) if dense is not None else None
    duplicates = _duplicates(questions)
    anchor_splits = _anchor_splits(questions)
    audits = []
    for i, question in enumerate(questions):
        run = _Run(question)
        _structure(run, passages, source.corpus)
        _evidence_text(run, passages, source.corpus)
        _lexical_link(run, passages, idf)
        _unrelated_evidence(run, passages, idf)
        if question.answerable:
            _completeness(run)
            _anchor_quality(run, passages)
        else:
            _no_answer(run, passages, idf, source.corpus)
        _batch_checks(run, duplicates, anchor_splits)
        if dense is not None:
            _semantic(run, passages, dense[i], typical_gold)
        audits.append(_finish(source, run, semantic=dense is not None))
    return audits


def _dense_scores(
    questions: Sequence[Question], passages: Sequence[Passage], encoder: Encoder
) -> list[list[float]]:
    q = encoder.queries([item.question for item in questions])
    p = encoder.passages([item.text for item in passages])
    return [[_cosine(row, vector) for vector in p] for row in q]


def _typical_gold_cosine(
    questions: Sequence[Question], passages: Sequence[Passage], dense: list[list[float]]
) -> float | None:
    """The lower quartile of the best gold cosine of answerable questions: "looks as relevant"."""
    values: list[float] = []
    for question, scores in zip(questions, dense, strict=True):
        gold = _gold_passages(question, passages)
        if question.answerable and gold:
            values.append(max(scores[p.index] for p in gold))
    if len(values) < 10:
        return None
    values.sort()
    return values[len(values) // 4]


def _structure(run: _Run, passages: Sequence[Passage], corpus: Mapping[str, str]) -> None:
    q = run.question
    if q.answerable and not q.gold:
        run.hard("no-gold", "an answerable question has no gold evidence")
    if not q.answerable and q.gold:
        run.hard("gold-on-unanswerable", "a question marked unanswerable has gold evidence")
    if q.type in {"multi_doc", "conflicting"} and len({r.doc for r in q.gold}) < 2:
        run.hard("one-document", f"a {q.type} question needs gold in at least two documents")
    for ref in q.gold:
        if _fold(ref.anchor) not in _fold(corpus.get(ref.doc, "")):
            run.hard("anchor-missing", f"the quote is not in {ref.doc}")
    if q.answerable and not (q.reference_answer or "").strip():
        run.hard("no-answer-text", "an answerable question has no expected answer")
    if q.answerable and not q.reference_facts:
        run.soft("no-facts", "no facts to check an answer against")
    written, label = _script(q.question), q.language
    if written != "none" and written != label:
        run.hard("language-label", f"the question is written in {written} but labelled {label}")
    if run.question.gold and not _gold_passages(q, passages):
        run.soft("anchor-across-paragraphs", "the quote does not sit inside a single paragraph")


def _evidence_text(run: _Run, passages: Sequence[Passage], corpus: Mapping[str, str]) -> None:
    """Figures and words of the expected answer against the evidence."""
    q = run.question
    if not q.answerable or not q.gold:
        return
    gold = _gold_passages(q, passages)
    gold_text = " ".join(p.text for p in gold)
    docs_text = " ".join(corpus.get(doc, "") for doc in {r.doc for r in q.gold})
    answer = q.reference_answer or ""
    figures = _numbers(answer)
    outside_docs = {n for n in figures if n not in _numbers(docs_text)}
    outside_gold = {n for n in figures if n not in _numbers(gold_text)} - outside_docs
    if outside_docs:
        run.hard(
            "figure-not-in-documents",
            f"the expected answer has {sorted(outside_docs)}, found in no gold document",
        )
    elif outside_gold:
        run.soft(
            "figure-outside-evidence",
            f"{sorted(outside_gold)} is in the gold document but not in the quoted passage",
        )
    elif figures:
        run.passed.append("figures-in-evidence")
    for fact in q.reference_facts:
        numbers = _numbers(fact)
        if numbers and not numbers <= _numbers(docs_text):
            run.hard(
                "fact-figure-missing",
                f"a fact has {sorted(numbers - _numbers(docs_text))} that no gold document contains",
            )
    words = [t for t in _terms(answer) if not t.isdigit()]
    if words and gold:
        known = {t for p in gold for t in p.terms} | set(_terms(docs_text))
        unsupported = [t for t in words if t not in known]
        share = len(unsupported) / len(words)
        run.metrics["answer_words_unsupported"] = round(share, 2)
        if share > UNSUPPORTED_WORDS and _script(answer) == detect_language(gold_text).value:
            run.soft(
                "answer-words",
                f"{share:.0%} of the expected answer's words are in no gold document",
            )
        elif share <= UNSUPPORTED_WORDS:
            run.passed.append("answer-words-in-evidence")


def _completeness(run: _Run) -> None:
    q = run.question
    answer = _fold(q.reference_answer or "")
    missing = [ref.doc for ref in q.gold if not _anchor_in_answer(ref.anchor, answer)]
    if len(q.gold) > 1 and missing:
        run.soft(
            "incomplete",
            f"the expected answer does not carry the evidence of {', '.join(sorted(set(missing)))}",
        )
    elif q.gold:
        run.passed.append("answer-carries-evidence")
    if len(q.gold) > 1 and len(q.reference_facts) < len(q.gold):
        run.soft("few-facts", f"{len(q.gold)} evidence quotes but {len(q.reference_facts)} facts")


def _anchor_in_answer(anchor: str, answer_folded: str) -> bool:
    figures = _numbers(anchor)
    if figures:
        return figures <= _numbers(answer_folded)
    terms = set(_terms(anchor))
    return bool(terms) and len(terms & set(_terms(answer_folded))) / len(terms) >= 0.4


def _anchor_quality(run: _Run, passages: Sequence[Passage]) -> None:
    for ref in run.question.gold:
        folded = _fold(ref.anchor)
        if len(ref.anchor) < SHORT_ANCHOR:
            run.soft("short-anchor", f"the quote {ref.anchor!r} is very short")
        elsewhere = {p.doc for p in passages if folded in _fold(p.text) and p.doc != ref.doc}
        if elsewhere:
            run.soft(
                "anchor-in-other-documents",
                f"the same quote also occurs in {', '.join(sorted(elsewhere))}",
            )


def _lexical_link(run: _Run, passages: Sequence[Passage], idf: Mapping[str, float]) -> None:
    q = run.question
    wanted = _terms(q.question)
    run.metrics["question_terms"] = len(wanted)
    gold = _gold_passages(q, passages)
    if not q.answerable or not gold:
        return
    same = [
        p for p in passages if p.language in (Language(q.language), Language.MIXED, Language.UND)
    ]
    run.metrics["same_language_documents"] = len({p.doc for p in same})
    best_gold = max(_coverage(wanted, p, idf) for p in gold)
    run.metrics["lexical_gold_coverage"] = round(best_gold, 2)
    overlap = _anchor_overlap(q)
    if q.type == "paraphrase" and overlap > 0.5:
        run.soft(
            "weak-paraphrase",
            f"{overlap:.0%} of the question's words are in the quote: hardly a paraphrase",
        )
    if q.type == "keyword" and best_gold < 0.3:
        run.soft(
            "weak-keyword",
            f"only {best_gold:.0%} of the question's terms are in the gold passage: not a keyword question",
        )
    if any(p.language is Language(q.language) for p in gold):
        others = [p for p in same if p not in gold and p.language is Language(q.language)]
        best_other = max((_coverage(wanted, p, idf) for p in others), default=0.0)
        run.metrics["lexical_best_other"] = round(best_other, 2)
        if best_gold >= SAME_LANGUAGE_LINK and best_gold >= best_other:
            run.passed.append("lexical-link")
        elif best_other > best_gold + OTHER_PASSAGE_MARGIN:
            run.soft(
                "other-passage-fits-better",
                f"another passage shares more of the question's terms ({best_other:.2f} against {best_gold:.2f})",
            )


def _unrelated_evidence(run: _Run, passages: Sequence[Passage], idf: Mapping[str, float]) -> None:
    """An evidence quote that neither the question nor the expected answer has anything to do with.

    The typical cause is a gold entry added only to satisfy "a multi-document question needs two
    documents". It is only claimed when the quote shares no term with the question (same language)
    and its figures and words are absent from the expected answer.
    """
    q = run.question
    if not q.answerable or len(q.gold) < 2:
        return
    wanted = _terms(q.question)
    answer = _fold(q.reference_answer or "")
    for ref in q.gold:
        passage = _passage_of(ref.doc, ref.anchor, passages)
        if passage is None or passage.language is not Language(q.language):
            continue
        if _coverage(wanted, passage, idf) == 0.0 and not _anchor_in_answer(ref.anchor, answer):
            run.hard(
                "unrelated-evidence",
                f"the quote {ref.anchor!r} of {ref.doc} shares nothing with the question or the expected answer",
            )


def _anchor_overlap(q: Question) -> float:
    wanted = set(_terms(q.question))
    anchor_terms = {t for r in q.gold for t in _terms(r.anchor)}
    return len(wanted & anchor_terms) / len(wanted) if wanted else 0.0


def _no_answer(
    run: _Run, passages: Sequence[Passage], idf: Mapping[str, float], corpus: Mapping[str, str]
) -> None:
    q = run.question
    stems = {t for p in passages for t in p.terms}
    for term in q.absence_terms:
        found = [t for t in _terms(term) if t in stems]
        if found and len(found) == len(_terms(term)):
            run.hard(
                "absence-term-present",
                f"the term {term!r} occurs in the corpus (as {', '.join(found)})",
            )
    if not q.absence_terms:
        run.soft("no-absence-terms", "nothing records why this question has no answer")
    wanted = _terms(q.question)
    best, where = 0.0, ""
    for p in passages:
        if p.language is Language(q.language) or p.language in (Language.MIXED, Language.UND):
            c = _coverage(wanted, p, idf)
            if c > best:
                best, where = c, p.doc
    run.metrics["lexical_best_passage"] = round(best, 2)
    if best >= NEAR_NO_ANSWER_COVERAGE:
        run.soft(
            "looks-answerable", f"a passage of {where} shares {best:.0%} of the question's terms"
        )
    else:
        run.passed.append("no-passage-covers-the-question")
    del corpus


def _duplicates(questions: Sequence[Question]) -> dict[str, list[str]]:
    sets = {q.id: frozenset(_terms(q.question)) for q in questions}
    found: dict[str, list[str]] = {}
    items = list(questions)
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            x, y = sets[a.id], sets[b.id]
            if len(x) >= 3 and len(y) >= 3 and len(x & y) / len(x | y) >= NEAR_DUPLICATE:
                found.setdefault(a.id, []).append(b.id)
                found.setdefault(b.id, []).append(a.id)
    return found


def _anchor_splits(questions: Sequence[Question]) -> dict[str, set[str]]:
    splits: dict[str, set[str]] = {}
    for q in questions:
        for ref in q.gold:
            splits.setdefault(f"{ref.doc}::{_fold(ref.anchor)}", set()).add(q.split)
    return splits


def _batch_checks(
    run: _Run, duplicates: Mapping[str, list[str]], anchor_splits: Mapping[str, set[str]]
) -> None:
    q = run.question
    twins = duplicates.get(q.id, [])
    if twins:
        run.soft("near-duplicate", f"nearly the same question as {', '.join(twins[:3])}")
    for ref in q.gold:
        if len(anchor_splits.get(f"{ref.doc}::{_fold(ref.anchor)}", set())) > 1:
            run.soft("split-leak", "the same evidence is used in both the dev and the test split")
            break


def _semantic(
    run: _Run, passages: Sequence[Passage], scores: Sequence[float], typical_gold: float | None
) -> None:
    q = run.question
    if not q.answerable:
        best = max(range(len(passages)), key=lambda i: scores[i])
        run.metrics["dense_best"] = round(scores[best], 3)
        if typical_gold is not None and scores[best] >= typical_gold:
            run.soft(
                "dense-looks-answerable",
                f"{passages[best].doc} scores {scores[best]:.2f}, as high as the lower quartile of answerable questions ({typical_gold:.2f})",
            )
        else:
            run.passed.append("dense-no-close-passage")
        return
    gold = _gold_passages(q, passages)
    if not gold:
        return
    # The model prefers a passage in the question's own language, so a cross-language pair is
    # ranked among the passages written in the evidence's language only.
    languages = {p.language for p in gold}
    pool = [p for p in passages if p.language in languages]
    order = sorted(pool, key=lambda p: scores[p.index], reverse=True)
    rank = {p.index: r for r, p in enumerate(order, start=1)}
    best_gold = min(rank[p.index] for p in gold)
    gold_score = max(scores[p.index] for p in gold)
    other = max((scores[p.index] for p in pool if p not in gold), default=0.0)
    run.metrics["dense_gold_rank"] = best_gold
    run.metrics["dense_pool"] = len(pool)
    run.metrics["dense_gold_score"] = round(gold_score, 3)
    run.metrics["dense_margin"] = round(gold_score - other, 3)
    if best_gold == 1 or (len(q.gold) > 1 and best_gold <= len(q.gold)):
        if gold_score - other < DENSE_MARGIN and len(q.gold) == 1:
            run.soft(
                "dense-close-call",
                f"another passage scores almost as high ({other:.3f} against {gold_score:.3f})",
            )
        else:
            run.passed.append("dense-gold-first")
    elif best_gold <= max(3, min(len({r.doc for r in q.gold}), 3)):
        run.soft(
            "dense-gold-not-first",
            f"the evidence ranks {best_gold} of {len(pool)} for this question",
        )
    else:
        run.soft(
            "dense-gold-far",
            f"the evidence ranks {best_gold} of {len(pool)}: the quote may not answer the question",
        )


def _finish(source: SourceDataset, run: _Run, *, semantic: bool) -> QuestionAudit:
    q = run.question
    linked = "lexical-link" in run.passed or "dense-gold-first" in run.passed
    if q.answerable and not linked and not any(f.severity != "info" for f in run.flags):
        why = (
            "no semantic layer was run"
            if not semantic
            else "the semantic ranking did not confirm it"
        )
        run.soft(
            "link-unverified",
            f"the link between question and evidence could not be shown automatically ({why})",
        )
    hard = [f for f in run.flags if f.severity == "hard"]
    soft = [f for f in run.flags if f.severity == "soft"]
    verdict = Verdict.INCORRECT if hard else Verdict.DOUBTFUL if soft else Verdict.CORRECT
    hint = ""
    if hard:
        hint = "; ".join(f.message for f in hard)
    elif soft:
        hint = "; ".join(f.message for f in soft[:3])
    return QuestionAudit(
        key=f"{source.name}:{q.id}", verdict=verdict, flags=tuple(run.flags),
        passed=tuple(run.passed), metrics=dict(run.metrics), hint=hint,
    )  # fmt: skip


def summarize(audits: Sequence[QuestionAudit]) -> dict[str, Any]:
    by_verdict = Counter(a.verdict.value for a in audits)
    flags = Counter(f.id for a in audits for f in a.flags if f.severity != "info")
    return {
        "questions": len(audits),
        "by_verdict": {v.value: by_verdict.get(v.value, 0) for v in Verdict},
        "flags": dict(flags.most_common()),
    }
