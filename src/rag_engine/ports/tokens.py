"""Token counting port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    @property
    def counter_id(self) -> str:
        """Stable identifier recorded next to every stored token count."""
        ...

    def count(self, text: str) -> int:
        """Number of tokens in ``text`` (always >= 0, deterministic)."""
        ...
