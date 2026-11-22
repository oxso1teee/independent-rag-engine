"""The configured counter, limits and strategy reach the processor built by the container."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_engine.chunking.registry import build_counter
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.container import DefaultContainer
from rag_engine.domain import ErrorCode, RagError, SourceKey
from rag_engine.ports import TokenCounter

DOC = b"# Title\n\nFirst paragraph.\n\nSecond paragraph.\n"


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "rag.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_heuristic_counter_is_the_default() -> None:
    counter: TokenCounter = build_counter("heuristic")
    assert isinstance(counter, HeuristicTokenCounter) and counter.counter_id == "heuristic/1"


@pytest.mark.parametrize(("kind", "path"), [("tokenizers", None), ("magic", None)])
def test_a_counter_that_cannot_be_built_is_a_configuration_error(
    kind: str, path: Path | None
) -> None:
    with pytest.raises(RagError) as info:
        build_counter(kind, path)
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


def test_the_container_builds_processing_from_the_configuration(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        "[chunking]\ntarget_tokens = 64\nmax_tokens = 96\nmin_tokens = 16\n"
        'strategy = "recursive"\n\n'
        "[parsing]\nmax_blocks = 500\nmax_archive_mb = 8\ntimeout_s = 5\n",
    )
    processing = DefaultContainer().processing(config)
    assert (processing.policy.target_tokens, processing.policy.max_tokens) == (64, 96)
    assert processing.limits.max_blocks == 500
    assert processing.limits.max_archive_bytes == 8 * 1024 * 1024
    assert processing.limits.timeout_s == 5
    done = processing.build((), None).process(
        DOC,
        SourceKey("a.md"),
        processing.limits,
        processing.policy,
        processing.ids.document_version_id(),
    )
    assert done.chunker_id == "recursive"  # the configured strategy beats the structure
    other = processing.build((), "fixed").process(
        DOC,
        SourceKey("a.md"),
        processing.limits,
        processing.policy,
        processing.ids.document_version_id(),
    )
    assert other.chunker_id == "fixed"


def test_cleaner_patterns_are_validated_when_the_processor_is_built(tmp_path: Path) -> None:
    processing = DefaultContainer().processing(None)
    assert processing.build(("^Footer.*$",), None) is not None
    with pytest.raises(RagError) as info:
        processing.build(("(a+)+$",), None)
    assert info.value.code is ErrorCode.E_CONFIG_INVALID


def test_a_broken_configuration_is_reported_before_anything_is_processed(tmp_path: Path) -> None:
    config = write_config(tmp_path, "[chunking]\nmin_tokens = 900\n")
    with pytest.raises(RagError) as info:
        DefaultContainer().processing(config)
    assert info.value.code is ErrorCode.E_CONFIG_INVALID
