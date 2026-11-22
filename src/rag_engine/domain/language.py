"""Language of a text from its script (no dependencies, deterministic).

The result drives the lexical analyser, so only the distinctions the product needs are made:
Cyrillic text is ``ru``, Latin-script text is ``en`` (English and other Latin-script
languages share the analyser), a substantial mix of both is ``mixed``, and anything else
(too few letters, other scripts, numbers, symbols) is ``und``.
"""

from __future__ import annotations

from rag_engine.domain.models import Language

MIN_LETTERS = 5  # Cyrillic + Latin letters needed before a decision is made
DOMINANCE = 0.85  # share of one script that makes a text ru or en rather than mixed
SCRIPT_SHARE = 0.5  # Cyrillic + Latin must be at least this share of all letters

_SAMPLE_THRESHOLD = 200_000  # longer texts are sampled instead of scanned completely
_SAMPLES = 50
_SAMPLE_SIZE = 4_000


def _sample(text: str) -> str:
    if len(text) <= _SAMPLE_THRESHOLD:
        return text
    step = (len(text) - _SAMPLE_SIZE) // (_SAMPLES - 1)
    return "".join(text[i * step : i * step + _SAMPLE_SIZE] for i in range(_SAMPLES))


def detect_language(text: str) -> Language:
    cyrillic = latin = other = 0
    for char in _sample(text):
        if not char.isalpha():
            continue
        code = ord(char)
        if 0x0400 <= code <= 0x052F:
            cyrillic += 1
        elif code < 0x0250 or 0x1E00 <= code <= 0x1EFF:
            latin += 1
        else:
            other += 1
    known = cyrillic + latin
    if known < MIN_LETTERS or known < SCRIPT_SHARE * (known + other):
        return Language.UND
    share = cyrillic / known
    if share >= DOMINANCE:
        return Language.RU
    if share <= 1 - DOMINANCE:
        return Language.EN
    return Language.MIXED
