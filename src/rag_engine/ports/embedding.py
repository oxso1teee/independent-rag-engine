"""Embedding provider port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from rag_engine.domain import EmbeddingModelId

type Vectors = NDArray[np.float32]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors.

    Query/document prefixes belong to the model description, not to callers.
    Implementations raise ``E_EMBED_DIM_MISMATCH`` rather than return vectors of the
    wrong size, and fail instead of silently truncating over-long input.
    """

    @property
    def model_id(self) -> EmbeddingModelId: ...

    @property
    def dimension(self) -> int: ...

    @property
    def max_input_tokens(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Array of shape ``(len(texts), dimension)``, one row per text, in order."""
        ...

    async def embed_query(self, text: str) -> Vectors:
        """Array of shape ``(1, dimension)``."""
        ...
