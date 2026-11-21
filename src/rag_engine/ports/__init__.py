"""Ports: Protocol contracts between the core and its adapters. No implementations."""

from rag_engine.ports.chunker import Chunker
from rag_engine.ports.clock import Clock
from rag_engine.ports.embedding import EmbeddingProvider, Vectors
from rag_engine.ports.lexical import LexicalDocument, LexicalIndex
from rag_engine.ports.llm import (
    Delta,
    FinishReason,
    GenerationParams,
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ModelLimits,
    TokenUsage,
)
from rag_engine.ports.parser import ParseHints, Parser
from rag_engine.ports.rerank import Candidate, RerankBudget, RerankedCandidate, Reranker
from rag_engine.ports.source import (
    SourceAdapter,
    SourceEntry,
    SourceIdentity,
    SourceRevision,
)
from rag_engine.ports.store import Store
from rag_engine.ports.tokens import TokenCounter
from rag_engine.ports.vector import VectorIndex

__all__ = [
    "Candidate",
    "Chunker",
    "Clock",
    "Delta",
    "EmbeddingProvider",
    "FinishReason",
    "GenerationParams",
    "GenerationRequest",
    "GenerationResult",
    "LLMProvider",
    "LexicalDocument",
    "LexicalIndex",
    "ModelLimits",
    "ParseHints",
    "Parser",
    "RerankBudget",
    "RerankedCandidate",
    "Reranker",
    "SourceAdapter",
    "SourceEntry",
    "SourceIdentity",
    "SourceRevision",
    "Store",
    "TokenCounter",
    "TokenUsage",
    "VectorIndex",
    "Vectors",
]
