"""Choosing a chunking strategy for a document."""

from __future__ import annotations

import pytest

from rag_engine.chunking.registry import STRATEGIES, ChunkerRegistry
from rag_engine.chunking.tokens import HeuristicTokenCounter
from rag_engine.domain import BlockKind, ErrorCode, RagError
from rag_engine.ports import ChunkerSelector
from tests.chunking_checks import document_of, prose

COUNTER = HeuristicTokenCounter()
P, H = BlockKind.PARAGRAPH, BlockKind.HEADING


def chosen(
    registry: ChunkerRegistry, *blocks: tuple[BlockKind, str] | tuple[BlockKind, str, int]
) -> str:
    return registry.select(document_of(*blocks)).chunker_id


def test_the_registry_is_a_selector() -> None:
    assert isinstance(ChunkerRegistry(COUNTER), ChunkerSelector)


def test_auto_follows_the_structure_of_the_document() -> None:
    registry = ChunkerRegistry(COUNTER)
    assert chosen(registry, (P, "intro"), (H, "Title", 1), (P, "text")) == "structure_aware"
    assert chosen(registry, (P, prose(2)), (P, prose(2, start=5))) == "recursive"
    assert chosen(registry, (P, prose(8))) == "fixed"
    assert chosen(registry) == "fixed"


@pytest.mark.parametrize("strategy", [s for s in STRATEGIES if s != "auto"])
def test_a_configured_strategy_is_used_for_every_document(strategy: str) -> None:
    registry = ChunkerRegistry(COUNTER, strategy)
    for blocks in ([(H, "T", 1), (P, "x")], [(P, "a"), (P, "b")], [(P, "a")]):
        assert chosen(registry, *blocks) == strategy


def test_every_strategy_can_be_fetched_by_id() -> None:
    registry = ChunkerRegistry(COUNTER)
    for strategy in STRATEGIES[1:]:
        assert registry.get(strategy).chunker_id == strategy


def test_an_unknown_strategy_is_a_configuration_error() -> None:
    with pytest.raises(RagError) as info:
        ChunkerRegistry(COUNTER, "semantic")
    assert info.value.code is ErrorCode.E_CONFIG_INVALID
    assert info.value.context["strategy"] == "semantic"
