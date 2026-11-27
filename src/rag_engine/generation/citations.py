"""Citation validation for a generated answer.

Level 1 is structural: every ``[E<n>]`` marker must name evidence that was in the context, and an
answer must cite something. Level 2 is lexical overlap: a cited sentence must share enough of its
content terms with the evidence it cites. Level 2 is a cheap, deterministic screen, not entailment:
it catches a marker pasted on an unrelated passage, not a paraphrase that distorts the meaning.
A sentence that cannot be judged this way (too few terms, or written in another script than its
evidence, as when a Russian answer cites an English passage) is reported as unjudged and is never
counted as a failure.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from rag_engine.domain import Context, Language
from rag_engine.domain.answer import CitationCheck, CitationProblem, SentenceCheck
from rag_engine.lexical.analysis import analyze

_MARKER = re.compile(r"\[E([1-9][0-9]*)\]")
_SENTENCE = re.compile(r"[^\n]+?(?:[.!?…]+(?:\s*\[E[1-9][0-9]*\])*(?=\s|$)|$)")
_MIN_TERMS = 2
_MIN_TERM_LENGTH = 3


def extract_markers(text: str) -> tuple[str, ...]:
    """Evidence ids named in ``text``, in order of first appearance."""
    return tuple(dict.fromkeys(f"E{number}" for number in _MARKER.findall(text)))


def split_sentences(text: str) -> list[str]:
    """Sentences and list lines; a marker that follows the full stop stays with its sentence."""
    parts: list[str] = []
    for match in _SENTENCE.finditer(text):
        part = match.group().strip()
        if not part:
            continue
        if not _MARKER.sub("", part).strip() and parts:
            parts[-1] = f"{parts[-1]} {part}"
        else:
            parts.append(part)
    return parts


def _terms(text: str) -> set[str]:
    stems = analyze(_MARKER.sub(" ", text), Language.MIXED).split()
    return {s for s in stems if len(s) >= _MIN_TERM_LENGTH or any(c.isdigit() for c in s)}


def _script(text: str) -> str:
    cyrillic = latin = 0
    for char in unicodedata.normalize("NFC", text):
        if not char.isalpha():
            continue
        if "Ѐ" <= char <= "ԯ":
            cyrillic += 1
        elif char.isascii() or "À" <= char <= "ɏ":
            latin += 1
    return "cyrillic" if cyrillic > latin else "latin"


class CitationValidator:
    def __init__(self, min_coverage: float = 0.5) -> None:
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1]")
        self._min_coverage = min_coverage

    def validate(self, answer: str, context: Context) -> CitationCheck:
        by_id = {item.evidence_id: item.text for item in context.evidence}
        markers = extract_markers(answer)
        cited = tuple(m for m in markers if m in by_id)
        unknown = tuple(m for m in markers if m not in by_id)

        uncited: list[str] = []
        checks: list[SentenceCheck] = []
        for sentence in split_sentences(answer):
            ids = tuple(m for m in extract_markers(sentence) if m in by_id)
            if ids:
                checks.append(self._check(sentence, ids, by_id))
            elif not extract_markers(sentence) and len(_terms(sentence)) >= _MIN_TERMS:
                uncited.append(sentence)

        problems: list[CitationProblem] = []
        if unknown:
            problems.append(CitationProblem.UNKNOWN_MARKER)
        if not cited:
            problems.append(CitationProblem.NO_CITATION)
        if any(not item.overlap_ok for item in checks):
            problems.append(CitationProblem.LOW_OVERLAP)
        return CitationCheck(cited, unknown, tuple(uncited), tuple(checks), tuple(problems))

    def _check(self, sentence: str, ids: Sequence[str], by_id: dict[str, str]) -> SentenceCheck:
        wanted = _terms(sentence)
        script = _script(_MARKER.sub(" ", sentence))
        support: set[str] = set()
        comparable = False
        for evidence_id in ids:
            if _script(by_id[evidence_id]) == script:
                comparable = True
                support |= _terms(by_id[evidence_id])
        if len(wanted) < _MIN_TERMS or not comparable:
            return SentenceCheck(sentence, tuple(ids), None, True)
        coverage = len(wanted & support) / len(wanted)
        return SentenceCheck(sentence, tuple(ids), coverage, coverage >= self._min_coverage)
