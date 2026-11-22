"""Contract of ``TokenCounter`` and the specifics of the two implementations."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.chunking.tokens_hf import HfTokenCounter
from rag_engine.domain import ErrorCode, RagError
from rag_engine.ports import TokenCounter

RU = "Это достаточно длинный русский текст, в котором есть слова, числа 12345 и знаки!"
EN = "This is an ordinary English paragraph with a few words, numbers 12345 and symbols!"
CODE = "def add(a: int, b: int) -> int:\n    return a + b  # sum of {a, b}\n"
SAMPLES = [RU, EN, CODE, "Смесь mixed text с английским and русским 42.", "a b c d e f g"]


def build_hf(tmp_path: Path) -> TokenCounter:
    """A tiny word-level tokenizer built here, so nothing is downloaded."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = sorted({w for s in SAMPLES for w in re.findall(r"\w+|[^\w\s]", s)})
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace()])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return HfTokenCounter(path)


@pytest.fixture(params=["heuristic", "tokenizers"])
def counter(request: pytest.FixtureRequest, tmp_path: Path) -> TokenCounter:
    factories: dict[str, Callable[[], TokenCounter]] = {
        "heuristic": HeuristicTokenCounter,
        "tokenizers": lambda: build_hf(tmp_path),
    }
    return factories[request.param]()


def test_satisfies_the_protocol(counter: TokenCounter) -> None:
    assert isinstance(counter, TokenCounter)
    assert counter.counter_id and "/" in counter.counter_id


def test_empty_text_has_no_tokens_and_text_has_some(counter: TokenCounter) -> None:
    assert counter.count("") == 0
    for sample in SAMPLES:
        assert counter.count(sample) > 0


def test_counting_is_deterministic_and_non_negative(counter: TokenCounter) -> None:
    for sample in SAMPLES:
        assert counter.count(sample) == counter.count(sample) >= 0


@given(st.text(max_size=300), st.text(max_size=300))
def test_more_text_never_means_fewer_tokens_for_the_heuristic(a: str, b: str) -> None:
    counter = HeuristicTokenCounter()
    assert counter.count(a + b) >= counter.count(a)
    assert counter.count(a + b) >= counter.count(b)


@given(st.text(max_size=300), st.text(max_size=300))
def test_the_heuristic_never_exceeds_the_sum_of_its_parts_by_more_than_rounding(
    a: str, b: str
) -> None:
    counter = HeuristicTokenCounter()
    assert counter.count(a + b) <= counter.count(a) + counter.count(b) + 2


# ------------------------------------------------------------------ heuristic


@pytest.mark.parametrize("text", SAMPLES)
def test_the_heuristic_stays_between_the_word_count_and_the_character_count(text: str) -> None:
    words = len(re.findall(r"\w+|[^\w\s]", text))
    tokens = HeuristicTokenCounter().count(text)
    # no subword tokenizer produces fewer tokens than words, and none more than characters
    assert words <= tokens <= len(text)


def test_scripts_are_weighted_differently() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count("a" * 320) == pytest.approx(115, abs=1)  # 320 / 3.2 * 1.15
    assert counter.count("я" * 190) == pytest.approx(115, abs=1)  # 190 / 1.9 * 1.15
    assert counter.count("漢" * 100) == pytest.approx(115, abs=1)  # one token per character
    assert counter.count("1" * 200) == pytest.approx(115, abs=1)
    assert counter.count("!" * 100) == pytest.approx(115, abs=1)
    assert counter.count("   \n\t  ") == 0  # whitespace is free
    assert counter.count("a b c d e f g") >= 7  # at least one token per word


def test_the_heuristic_overestimates_a_word_tokenizer(tmp_path: Path) -> None:
    exact = build_hf(tmp_path)
    estimate = HeuristicTokenCounter()
    for sample in SAMPLES:
        assert estimate.count(sample) >= exact.count(sample)
        assert estimate.count(sample) <= 4 * exact.count(sample)


def test_the_heuristic_is_fast_enough_for_large_documents() -> None:
    text = (RU + "\n" + EN + "\n" + CODE) * 20_000  # several megabytes
    started = time.perf_counter()
    assert HeuristicTokenCounter().count(text) > 0
    assert time.perf_counter() - started < 5.0


# ------------------------------------------------------------------- tokenizers


def test_the_exact_counter_counts_with_the_model_tokenizer(tmp_path: Path) -> None:
    counter = build_hf(tmp_path)
    assert counter.count("a b c d e f g") == 7
    assert counter.count("Hello, world") == 3  # unknown words are still one token each


def test_the_exact_counter_id_follows_the_tokenizer_file(tmp_path: Path) -> None:
    first = build_hf(tmp_path)
    again = build_hf(tmp_path)
    assert first.counter_id == again.counter_id
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    from tokenizers import Tokenizer, models

    Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]")).save(
        str(other_dir / "tokenizer.json")
    )
    assert HfTokenCounter(other_dir / "tokenizer.json").counter_id != first.counter_id


def test_a_missing_or_broken_tokenizer_file_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(RagError) as missing:
        HfTokenCounter(tmp_path / "absent.json")
    assert missing.value.code is ErrorCode.E_CONFIG_INVALID
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(RagError) as invalid:
        HfTokenCounter(broken)
    assert invalid.value.code is ErrorCode.E_CONFIG_INVALID


def test_a_missing_optional_dependency_is_reported_with_a_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object) -> object:
        if name == "tokenizers":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(RagError) as info:
        HfTokenCounter(tmp_path / "tokenizer.json")
    assert info.value.code is ErrorCode.E_CONFIG_INVALID
    assert "rag[tokenizers]" in info.value.message
