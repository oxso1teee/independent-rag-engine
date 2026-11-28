"""Request and response models of the API; they are the source of the OpenAPI document."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SourceName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
DocumentKey = Annotated[
    str, StringConstraints(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Problem(BaseModel):
    """``application/problem+json`` (RFC 9457) with the engine's stable error code."""

    type: str
    title: str
    status: int
    detail: str
    code: str
    category: str
    retryable: bool
    instance: str | None = None
    request_id: str | None = None
    context: dict[str, Any] | None = None


class Health(BaseModel):
    status: Literal["ok"]
    version: str


class Readiness(BaseModel):
    ready: bool
    checks: dict[str, bool]
    detail: dict[str, str] = Field(default_factory=dict)


class SourceOut(BaseModel):
    id: str
    name: str
    kind: str
    managed: bool
    created_at: str


class SourceCreate(_Model):
    name: SourceName


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None = None


class DocumentOut(BaseModel):
    id: str
    source: str
    key: str
    status: str
    title: str | None
    tags: list[str]
    current_version: str | None
    indexed: bool
    updated_at: str


class VersionOut(BaseModel):
    id: str
    seq: int
    content_hash: str
    size_bytes: int
    mime: str
    created_at: str
    parse_warnings: list[str]


class DocumentDetail(DocumentOut):
    versions: list[VersionOut]
    chunks: int | None = None


class DocumentCreate(_Model):
    source: SourceName
    key: DocumentKey
    text: str | None = None
    content_base64: str | None = None

    @model_validator(mode="after")
    def _exactly_one_content(self) -> DocumentCreate:
        if (self.text is None) == (self.content_base64 is None):
            raise ValueError("give exactly one of text and content_base64")
        return self


class JobOut(BaseModel):
    id: str
    kind: str
    source: str | None
    document_id: str | None
    stage: str
    status: str
    attempts: int
    error_code: str | None
    created_at: str
    finished_at: str | None


class Accepted(BaseModel):
    document: DocumentOut | None = None
    ingest: dict[str, Any]
    indexing: Literal["scheduled"]


class Stats(BaseModel):
    schema_version: int | None
    index_generation: int | None
    sources: list[dict[str, Any]]
    jobs: dict[str, int]


class FilterIn(_Model):
    sources: list[str] = Field(default_factory=list, max_length=50)
    path_prefix: str | None = Field(default=None, max_length=500)
    path_glob: str | None = Field(default=None, max_length=500)
    tags_any: list[str] = Field(default_factory=list, max_length=50)
    tags_all: list[str] = Field(default_factory=list, max_length=50)
    language: str | None = None
    kinds: list[str] = Field(default_factory=list, max_length=10)
    modified_after: str | None = None
    modified_before: str | None = None
    doc_ids: list[str] = Field(default_factory=list, max_length=100)
    exclude_doc_ids: list[str] = Field(default_factory=list, max_length=100)


class SearchIn(_Model):
    query: str = Field(min_length=1)
    mode: Literal["dense", "lexical", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    filters: FilterIn = Field(default_factory=FilterIn)
    rerank: bool | None = None
    debug: bool = False


class SearchOut(BaseModel):
    request_id: str
    mode: str
    index_generation: int
    embedding_model: str | None
    fallbacks: list[str]
    hits: list[dict[str, Any]]
    diagnostics: dict[str, Any] | None = None


class AnswerIn(_Model):
    question: str = Field(min_length=1)
    mode: Literal["dense", "lexical", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    filters: FilterIn = Field(default_factory=FilterIn)
    rerank: bool | None = None
    stream: bool = False
    live: bool = False
    include_sources: bool = False
    include_unverified_text: bool = False


class AnswerOut(BaseModel):
    request_id: str
    status: Literal["answered", "insufficient_evidence", "unverified", "error"]
    text: str
    reasons: list[str]
    citations: list[dict[str, Any]]
    sources: list[dict[str, Any]] | None = None
    unverified_text: str | None = None
    error: dict[str, Any] | None = None
    diagnostics: dict[str, Any]
