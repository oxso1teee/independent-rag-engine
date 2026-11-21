from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_engine.domain import (
    ChunkId,
    ContentHash,
    DocumentVersionId,
    EmbeddingModelId,
    RagError,
    Sha256Hex,
    SourceKey,
    TextHash,
    chunker_fingerprint,
    normalize_source_key,
    text_hash,
)
from tests.factories import make_ids

IDS = make_ids()
VERSION_A = IDS.document_version_id()
VERSION_B = IDS.document_version_id()


def test_text_hash_is_sha256_of_utf8() -> None:
    assert text_hash("abc").hex == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert isinstance(text_hash("abc"), TextHash)
    assert ContentHash.of_bytes(b"abc").hex == text_hash("abc").hex
    assert ContentHash.of_bytes(b"abc") != text_hash("abc")  # type: ignore[comparison-overlap]


@pytest.mark.parametrize("bad", ["", "xyz", "A" * 64, "a" * 63, "g" * 64])
def test_sha256_rejects_malformed_digests(bad: str) -> None:
    with pytest.raises(RagError):
        Sha256Hex(bad)


@given(st.text())
def test_text_hash_is_deterministic(text: str) -> None:
    assert text_hash(text) == text_hash(text)


@given(st.text(), st.text())
def test_text_hash_separates_different_texts(a: str, b: str) -> None:
    assert (text_hash(a) == text_hash(b)) == (a == b)


@given(st.text(max_size=50), st.integers(0, 10_000))
def test_chunk_id_is_deterministic_and_well_formed(fingerprint: str, ordinal: int) -> None:
    first = ChunkId.derive(VERSION_A, fingerprint, ordinal)
    assert first == ChunkId.derive(VERSION_A, fingerprint, ordinal)
    assert len(first.value) == 26
    assert ChunkId(first.value) == first


@given(st.text(max_size=20), st.integers(0, 1000))
def test_chunk_id_depends_on_every_component(fingerprint: str, ordinal: int) -> None:
    base = ChunkId.derive(VERSION_A, fingerprint, ordinal)
    assert base != ChunkId.derive(VERSION_B, fingerprint, ordinal)
    assert base != ChunkId.derive(VERSION_A, fingerprint + "x", ordinal)
    assert base != ChunkId.derive(VERSION_A, fingerprint, ordinal + 1)


def test_chunk_id_encoding_is_unambiguous() -> None:
    # Boundary shifts between fingerprint and ordinal must not collide.
    assert ChunkId.derive(VERSION_A, "a", 1) != ChunkId.derive(VERSION_A, "a\x00", 0)


def test_chunk_id_golden_values() -> None:
    """Changing the derivation would invalidate every stored chunk id."""
    version = DocumentVersionId.parse("017f22e2-79b0-7cc3-98c4-dc0c0c07398f")
    assert ChunkId.derive(version, "fp-1", 0).value == "6gqjymy2c5h3abfzbpzdaegoce"
    assert ChunkId.derive(version, "fp-1", 7).value == "rjh64s7am7zgajaswcsqwhm2n4"


@pytest.mark.parametrize("bad", ["", "A" * 26, "a" * 25, "a" * 27, "1" * 26, "a" * 25 + "b"])
def test_chunk_id_rejects_malformed_values(bad: str) -> None:
    with pytest.raises(RagError):
        ChunkId(bad)


def test_chunk_id_rejects_negative_ordinal() -> None:
    with pytest.raises(RagError):
        ChunkId.derive(VERSION_A, "fp", -1)


def test_chunker_fingerprint_is_order_independent_and_sensitive() -> None:
    a = chunker_fingerprint("structure", "1", {"max": 384, "min": 48}, "n1")
    b = chunker_fingerprint("structure", "1", {"min": 48, "max": 384}, "n1")
    assert a == b
    assert a != chunker_fingerprint("structure", "2", {"max": 384, "min": 48}, "n1")
    assert a != chunker_fingerprint("structure", "1", {"max": 385, "min": 48}, "n1")
    assert a != chunker_fingerprint("structure", "1", {"max": 384, "min": 48}, "n2")


def _model(**overrides: object) -> EmbeddingModelId:
    params: dict[str, object] = {
        "provider": "local",
        "name": "embed-small:latest",
        "revision": "abc123",
        "dimension": 1024,
        "normalized": True,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    }
    params.update(overrides)
    return EmbeddingModelId.build(**params)  # type: ignore[arg-type]


def test_embedding_model_id_format_and_round_trip() -> None:
    model = _model()
    text = str(model)
    assert text.startswith("local/embed-small:latest@abc123#d1024:l2:")
    assert EmbeddingModelId.parse(text) == model


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "remote"},
        {"name": "other"},
        {"revision": "zzz"},
        {"dimension": 768},
        {"normalized": False},
        {"query_prefix": ""},
        {"document_prefix": ""},
    ],
)
def test_embedding_model_id_changes_with_every_component(change: dict[str, object]) -> None:
    assert _model(**change) != _model()
    assert str(_model(**change)) != str(_model())


@pytest.mark.parametrize("bad", ["", "no-slash", "p/n@r#d0:l2:abcdef12", "p/n@r#d8:l3:abcdef12"])
def test_embedding_model_id_rejects_malformed_text(bad: str) -> None:
    with pytest.raises(RagError):
        EmbeddingModelId.parse(bad)


def test_embedding_model_id_rejects_zero_dimension() -> None:
    with pytest.raises(RagError):
        _model(dimension=0)


def test_source_key_normalises_unicode_and_separators() -> None:
    key = normalize_source_key("Папка\\café.md")
    assert key.value == "Папка/café.md"
    assert str(key) == key.value


@pytest.mark.parametrize(
    "bad", ["", "/abs", "a//b", "a/./b", "a/../b", "a\\b", "trailing/", "café", "nul\x00"]
)
def test_source_key_rejects_unsafe_keys(bad: str) -> None:
    with pytest.raises(RagError):
        SourceKey(bad)
