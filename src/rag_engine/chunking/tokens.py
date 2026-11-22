"""Token counting that needs no model.

``HeuristicTokenCounter`` estimates tokens from the scripts present in the text, never fewer
than one per word. It is deterministic, fast and deliberately errs on the high side: a chunk
that the estimate calls "small enough" is small enough for any common tokenizer. Exact counting
with a model's own tokenizer is available through ``rag_engine.chunking.tokens_hf``.
"""

from __future__ import annotations

import math
import re

# Characters per token, chosen low (more tokens) so the estimate does not undercount.
_LATIN_PER_TOKEN = 3.2
_CYRILLIC_PER_TOKEN = 1.9
_DIGITS_PER_TOKEN = 2.0
_OTHER_LETTERS_PER_TOKEN = 2.0
_CJK_PER_TOKEN = 1.0
_SYMBOL_TOKENS = 1.0  # every punctuation mark or symbol counts as a token of its own
_MARGIN = 1.15

_LATIN = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]")
_CYRILLIC = re.compile(r"[Ѐ-ԯ]")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
_DIGIT = re.compile(r"\d")
_LETTER = re.compile(r"[^\W\d_]")
_SYMBOL = re.compile(r"[^\w\s]|_")
_WORD = re.compile(r"\w+")


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(text) - len(pattern.sub("", text))


class HeuristicTokenCounter:
    counter_id = "heuristic/1"

    def count(self, text: str) -> int:
        if not text:
            return 0
        cyrillic = _count(_CYRILLIC, text)
        latin = _count(_LATIN, text)
        cjk = _count(_CJK, text)
        letters = _count(_LETTER, text)
        other = max(letters - cyrillic - latin - cjk, 0)
        by_script = (
            latin / _LATIN_PER_TOKEN
            + cyrillic / _CYRILLIC_PER_TOKEN
            + cjk / _CJK_PER_TOKEN
            + other / _OTHER_LETTERS_PER_TOKEN
            + _count(_DIGIT, text) / _DIGITS_PER_TOKEN
        )
        words = _WORD.subn("", text)[1]  # no subword tokenizer gives a word fewer than one token
        raw = max(by_script, words) + _count(_SYMBOL, text) * _SYMBOL_TOKENS
        return math.ceil(raw * _MARGIN)
