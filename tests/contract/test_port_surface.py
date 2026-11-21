"""Ports are pure contracts: Protocols only, no behaviour."""

from __future__ import annotations

import inspect
from typing import Protocol

import pytest

import rag_engine.ports as ports

EXPECTED = {
    "Chunker",
    "Clock",
    "EmbeddingProvider",
    "LLMProvider",
    "LexicalIndex",
    "Parser",
    "Reranker",
    "SourceAdapter",
    "Store",
    "TokenCounter",
    "VectorIndex",
}


def test_all_required_ports_are_exported() -> None:
    assert set(ports.__all__) >= EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_port_is_a_runtime_checkable_protocol(name: str) -> None:
    port = getattr(ports, name)
    assert issubclass(port, Protocol)
    assert getattr(port, "_is_runtime_protocol", False)
    assert inspect.isclass(port)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_port_methods_have_no_implementation(name: str) -> None:
    port = getattr(ports, name)
    for member_name, member in vars(port).items():
        func = member.fget if isinstance(member, property) else member
        if member_name.startswith("_") or not inspect.isfunction(func):
            continue
        source = inspect.getsource(func)
        body = source.split(":\n", 1)[-1] if ":\n" in source else source
        assert "..." in body or '"""' in body, f"{name}.{member_name} must be abstract"
