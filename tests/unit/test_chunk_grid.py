"""The chunk-grid report script: grid points, shape statistics and output formats."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[2] / "eval" / "scripts" / "chunk_grid.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("chunk_grid", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["chunk_grid"] = module
    spec.loader.exec_module(module)
    return module


grid = load()


@pytest.fixture(scope="module")
def rows() -> list[Any]:
    result: list[Any] = grid.run(grid.DEFAULT_CORPUS)
    return result


def test_the_grid_points_derive_their_bounds_from_the_target() -> None:
    policy = grid.grid_policy(256)
    assert (policy.target_tokens, policy.max_tokens, policy.min_tokens) == (256, 384, 48)
    assert grid.grid_policy(192).max_tokens == 288 and grid.grid_policy(384).max_tokens == 576
    assert grid.DEFAULT_TARGETS == (192, 256, 384)


def test_every_grid_point_describes_the_whole_corpus(rows: list[Any]) -> None:
    assert [r.target for r in rows] == [192, 256, 384]
    documents = {r.documents for r in rows}
    assert documents == {len(list(grid.DEFAULT_CORPUS.iterdir()))}
    for row in rows:
        assert 0 < row.min <= row.median <= row.p95 <= row.max <= row.max_tokens
        assert row.leaves > 0 and row.leaf_tokens >= row.leaves
        assert 0 <= row.overlap_share < 0.5 and 0 <= row.small_share <= 1


def test_larger_targets_give_fewer_and_longer_chunks(rows: list[Any]) -> None:
    leaves = [r.leaves for r in rows]
    medians = [r.median for r in rows]
    assert leaves == sorted(leaves, reverse=True) and leaves[0] > leaves[-1]
    assert medians == sorted(medians) and medians[0] < medians[-1]


def test_markdown_and_json_reports_say_they_rank_nothing(rows: list[Any]) -> None:
    markdown = grid.render_markdown(rows, "corpus_mini")
    assert markdown.startswith("# Chunk grid: corpus_mini") and grid.NOTE in markdown
    assert markdown.count("\n| 192 |") == 1 and markdown.count("\n| 384 |") == 1
    assert "best" not in markdown.lower() and "recommend" not in markdown.lower()
    payload = json.loads(grid.render_json(rows, "corpus_mini"))
    assert payload["note"] == grid.NOTE and [r["target"] for r in payload["rows"]] == [
        192,
        256,
        384,
    ]


def test_the_command_writes_a_file_or_standard_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "grid.json"
    assert grid.main(["--format", "json", "--targets", "256", "--output", str(target)]) == 0
    assert [r["target"] for r in json.loads(target.read_text())["rows"]] == [256]
    assert grid.main(["--targets", "192", "384", "--strategy", "recursive"]) == 0
    out = capsys.readouterr().out
    assert "| 192 |" in out and "| 384 |" in out and "| 256 |" not in out


def test_an_empty_corpus_reports_zeroes(tmp_path: Path) -> None:
    (row,) = grid.run(tmp_path, [256])
    assert (row.documents, row.leaves, row.leaf_tokens, row.max) == (0, 0, 0, 0)
