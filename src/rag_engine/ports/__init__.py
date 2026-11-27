"""Ports: Protocol contracts between the core and its adapters. No implementations."""

from rag_engine.ports.cache import EmbeddingCache, EmbeddingCacheWriter, StoredEmbedding
from rag_engine.ports.chunker import Chunker, ChunkerSelector
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
from rag_engine.ports.parser import ParseHints, ParseLimits, Parser
from rag_engine.ports.rerank import Candidate, RerankBudget, RerankedCandidate, Reranker
from rag_engine.ports.retrieval import (
    ChunkCatalog,
    ChunkContent,
    ChunkContentReader,
    ChunkFilter,
    ChunkInfo,
)
from rag_engine.ports.source import (
    ScanIssue,
    ScanReport,
    SourceAdapter,
    SourceEntry,
    SourceIdentity,
    SourceRevision,
)
from rag_engine.ports.store import MetadataReader, MetadataWriter, Store, StoredChunk, UnitOfWork
from rag_engine.ports.tokens import TokenCounter
from rag_engine.ports.vector import VectorIndex

__all__ = [
    "Candidate",
    "ChunkCatalog",
    "ChunkContent",
    "ChunkContentReader",
    "ChunkFilter",
    "ChunkInfo",
    "Chunker",
    "ChunkerSelector",
    "Clock",
    "Delta",
    "EmbeddingCache",
    "EmbeddingCacheWriter",
    "EmbeddingProvider",
    "FinishReason",
    "GenerationParams",
    "GenerationRequest",
    "GenerationResult",
    "LLMProvider",
    "LexicalDocument",
    "LexicalIndex",
    "MetadataReader",
    "MetadataWriter",
    "ModelLimits",
    "ParseHints",
    "ParseLimits",
    "Parser",
    "RerankBudget",
    "RerankedCandidate",
    "Reranker",
    "ScanIssue",
    "ScanReport",
    "SourceAdapter",
    "SourceEntry",
    "SourceIdentity",
    "SourceRevision",
    "Store",
    "StoredChunk",
    "StoredEmbedding",
    "TokenCounter",
    "TokenUsage",
    "UnitOfWork",
    "VectorIndex",
    "Vectors",
]
