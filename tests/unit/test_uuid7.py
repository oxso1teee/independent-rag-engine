from __future__ import annotations

import uuid

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_engine.domain import (
    DocumentId,
    ErrorCode,
    RagError,
    RequestId,
    SourceId,
    Uuid7Generator,
    uuid7,
)

# RFC 9562 appendix A.6 test vector.
RFC_UNIX_MS = 0x017F22E279B0
RFC_RAND = bytes([0x0C, 0xC3]) + bytes.fromhex("18C4DC0C0C07398F")
RFC_EXPECTED = uuid.UUID("017f22e2-79b0-7cc3-98c4-dc0c0c07398f")


def test_matches_rfc_9562_vector() -> None:
    assert uuid7(RFC_UNIX_MS, RFC_RAND) == RFC_EXPECTED


@given(st.integers(0, (1 << 48) - 1), st.binary(min_size=10, max_size=10))
def test_layout_is_valid_for_any_input(unix_ms: int, rand: bytes) -> None:
    value = uuid7(unix_ms, rand)
    assert value.version == 7
    assert value.variant == uuid.RFC_4122
    assert value.int >> 80 == unix_ms


@given(
    st.integers(0, (1 << 48) - 2),
    st.integers(1, 1 << 20),
    st.binary(min_size=10, max_size=10),
    st.binary(min_size=10, max_size=10),
)
def test_later_timestamps_sort_later(
    unix_ms: int, delta: int, rand_a: bytes, rand_b: bytes
) -> None:
    later = min(unix_ms + delta, (1 << 48) - 1)
    if later == unix_ms:
        return
    assert uuid7(unix_ms, rand_a).bytes < uuid7(later, rand_b).bytes


@pytest.mark.parametrize(
    ("unix_ms", "rand"),
    [(-1, bytes(10)), (1 << 48, bytes(10)), (0, bytes(9)), (0, bytes(11))],
)
def test_rejects_out_of_range_input(unix_ms: int, rand: bytes) -> None:
    with pytest.raises(RagError) as info:
        uuid7(unix_ms, rand)
    assert info.value.code is ErrorCode.E_INTEGRITY


def _generator(times: list[int]) -> Uuid7Generator:
    iterator = iter(times)
    last = {"t": times[0]}

    def now() -> int:
        last["t"] = next(iterator, last["t"])
        return last["t"]

    return Uuid7Generator(now, lambda size: bytes(size))


def test_generator_is_strictly_increasing_within_one_millisecond() -> None:
    generator = _generator([1000] * 50)
    values = [generator.next() for _ in range(50)]
    assert [v.bytes for v in values] == sorted({v.bytes for v in values})


def test_generator_never_goes_backwards_with_a_regressing_clock() -> None:
    generator = _generator([2000, 1500, 1000, 3000])
    values = [generator.next() for _ in range(4)]
    assert [v.bytes for v in values] == sorted(v.bytes for v in values)
    assert values[1].int >> 80 == 2000


def test_generator_advances_the_timestamp_when_the_counter_overflows() -> None:
    generator = Uuid7Generator(lambda: 5000, lambda size: b"\xff" * size)
    values = [generator.next() for _ in range(0x1000)]
    assert len({v.bytes for v in values}) == len(values)
    assert [v.bytes for v in values] == sorted(v.bytes for v in values)
    assert values[-1].int >> 80 > 5000


def test_typed_ids_validate_and_round_trip() -> None:
    value = uuid7(RFC_UNIX_MS, RFC_RAND)
    for id_type in (SourceId, DocumentId, RequestId):
        identifier = id_type(value)
        assert id_type.parse(str(identifier)) == identifier
        assert identifier.unix_ms == RFC_UNIX_MS


def test_ids_of_different_kinds_are_never_equal() -> None:
    value = uuid7(RFC_UNIX_MS, RFC_RAND)
    assert SourceId(value) != DocumentId(value)  # type: ignore[comparison-overlap]


@pytest.mark.parametrize("bad", ["", "not-a-uuid", str(uuid.UUID(int=0)), str(uuid.uuid4())])
def test_typed_ids_reject_non_uuid7(bad: str) -> None:
    with pytest.raises(RagError):
        SourceId.parse(bad)
