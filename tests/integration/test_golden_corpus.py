"""Exact parse-and-chunk output of every ``corpus_mini`` document against the stored snapshot.

A difference means normalisation, parsing, chunk boundaries, token counts or identifiers
changed. If that is intended, run ``uv run python tests/fixtures/update_golden.py`` and review the
diff of ``tests/fixtures/golden/corpus_mini.json``.
"""

from __future__ import annotations

import json

import pytest

from tests.corpus_snapshot import CORPUS, GOLDEN, dump_snapshot, snapshot

STORED = json.loads(GOLDEN.read_text(encoding="utf-8"))
HINT = "run `uv run python tests/fixtures/update_golden.py` if the change is intended"


def test_the_snapshot_covers_exactly_the_files_of_the_corpus() -> None:
    assert sorted(STORED) == sorted(p.name for p in CORPUS.iterdir()), HINT


@pytest.mark.parametrize("name", sorted(STORED))
def test_output_matches_the_stored_snapshot(name: str) -> None:
    current = json.loads(json.dumps(snapshot(name), ensure_ascii=False))
    stored = STORED[name]
    for field in sorted(stored):
        assert current[field] == stored[field], f"{name}: {field} differs; {HINT}"
    assert current == stored


def test_the_stored_file_is_in_the_canonical_format() -> None:
    assert GOLDEN.read_text(encoding="utf-8") == dump_snapshot(STORED)
