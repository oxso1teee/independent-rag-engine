"""Exact token counting with a model's own ``tokenizer.json`` (optional extra ``tokenizers``)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rag_engine.domain import ErrorCode, RagError


class HfTokenCounter:
    """Counts tokens with a local tokenizer file; nothing is downloaded."""

    def __init__(self, tokenizer_json: Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise RagError(
                ErrorCode.E_CONFIG_INVALID,
                "exact token counting needs the optional dependency: install rag[tokenizers]",
            ) from None
        try:
            payload = tokenizer_json.read_bytes()
        except OSError:
            raise RagError(
                ErrorCode.E_CONFIG_INVALID,
                "the tokenizer file cannot be read",
                context={"path": str(tokenizer_json)},
            ) from None
        try:
            self._tokenizer: Any = Tokenizer.from_str(payload.decode("utf-8"))
        except Exception as exc:
            raise RagError(
                ErrorCode.E_CONFIG_INVALID,
                "the tokenizer file is not a valid tokenizer.json",
                context={"path": str(tokenizer_json)},
            ) from exc
        self.counter_id = f"tokenizers/{hashlib.sha256(payload).hexdigest()[:12]}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
