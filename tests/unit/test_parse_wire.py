"""The messages between the parsing process and its isolated worker."""

from __future__ import annotations

import io
import struct

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rag_engine.domain import BlockKind, ErrorCode, ParseReason, RagError, parse_failed
from rag_engine.parsing.wire import (
    MAX_FRAME,
    decode_document,
    decode_error,
    decode_hints,
    dumps,
    encode_document,
    encode_error,
    encode_hints,
    frame,
    loads,
    read_frame,
)
from rag_engine.ports import ParseHints, ParseLimits
from tests.chunking_checks import document_of
from tests.isolation_fakes import Rich


def test_frames_round_trip_and_stop_at_the_end_of_the_stream() -> None:
    stream = io.BytesIO(frame(b"first") + frame(b"") + frame(b"x" * 100_000))
    assert read_frame(stream) == b"first"
    assert read_frame(stream) == b""
    assert read_frame(stream) == b"x" * 100_000
    with pytest.raises(EOFError):
        read_frame(stream)


def test_a_cut_frame_and_an_oversized_frame_are_refused() -> None:
    with pytest.raises(EOFError):
        read_frame(io.BytesIO(frame(b"hello")[:-2]))
    with pytest.raises(ValueError, match="too large"):
        read_frame(io.BytesIO(struct.pack(">Q", MAX_FRAME + 1)))


def test_a_document_survives_the_trip_with_every_field() -> None:
    document = Rich().parse(b"", ParseHints())
    again = decode_document(loads(dumps(encode_document(document))))
    assert again == document
    assert [dict(b.attrs) for b in again.blocks][-1] == {"rows": 2, "ok": True, "x": 1.5}
    assert again.created == document.created and again.language == document.language


def test_an_empty_document_survives_the_trip() -> None:
    document = document_of()
    assert decode_document(loads(dumps(encode_document(document)))) == document


@given(st.text(max_size=400))
@settings(max_examples=60, deadline=None)
def test_any_text_survives_the_trip_including_lone_surrogates(text: str) -> None:
    assert loads(dumps({"t": text}))["t"] == text


def test_a_rebuilt_document_is_checked_again() -> None:
    payload = encode_document(document_of((BlockKind.PARAGRAPH, "text")))
    payload["blocks"][0][2] = 999  # an offset beyond the text
    with pytest.raises(RagError):
        decode_document(payload)


def test_hints_round_trip_with_their_limits() -> None:
    hints = ParseHints(
        mime="application/pdf",
        extension="pdf",
        encoding="utf_8",
        limits=ParseLimits(max_pages=7, timeout_s=2.5),
    )
    assert decode_hints(loads(dumps(encode_hints(hints)))) == hints


def test_errors_round_trip_with_their_context() -> None:
    error = parse_failed(ParseReason.TOO_MANY_PAGES, "too long", pages=9, limit=5)
    again = decode_error(loads(dumps(encode_error(error))))
    assert (again.code, again.message) == (ErrorCode.E_PARSE_FAILED, "too long")
    assert dict(again.context) == {"reason": "too_many_pages", "pages": 9, "limit": 5}
