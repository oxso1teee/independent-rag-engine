"""Application settings.

Sources, highest priority first: explicit constructor arguments, ``RAG_*``
environment variables (nested keys use ``__``, e.g. ``RAG_LLM__CONTEXT_WINDOW``),
an optional TOML file, defaults. Secrets are never stored in settings: a
provider names the environment variable that holds its key (``api_key_env``).
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from rag_engine.config.profiles import PROFILE_LIMITS, ProfileLimits, ResourceProfile
from rag_engine.domain import ErrorCode, RagError

ProviderName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,31}$")]
EnvVarName = Annotated[str, StringConstraints(pattern=r"^[A-Z_][A-Z0-9_]{0,63}$")]
ModelName = Annotated[str, StringConstraints(min_length=1, max_length=200, strip_whitespace=True)]

_SECRET_LIKE = re.compile(r"key|token|secret|password|credential", re.IGNORECASE)


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LimitOverrides(_Section):
    """Optional explicit overrides of the selected profile's limits."""

    llm_concurrency: int | None = Field(default=None, ge=1, le=64)
    embed_concurrency: int | None = Field(default=None, ge=1, le=64)
    embed_batch: int | None = Field(default=None, ge=1, le=1024)
    ingest_workers: int | None = Field(default=None, ge=1, le=64)
    max_loaded_vectors_mb: int | None = Field(default=None, ge=64, le=1_048_576)
    reranker_enabled: bool | None = None


class StorageSettings(_Section):
    db_path: Path = Path("data/rag.db")


class EmbeddingSettings(_Section):
    enabled: bool = False
    provider: ProviderName | None = None
    model: ModelName | None = None
    revision: str | None = Field(default=None, pattern=r"^[^#\s]+$")
    dimension: int | None = Field(default=None, ge=1, le=65536)
    max_input_tokens: int = Field(default=8192, ge=1)
    query_prefix: str = ""
    document_prefix: str = ""
    base_url: str = "http://127.0.0.1:11434"
    model_dir: Path | None = None
    timeout_s: float = Field(default=180.0, gt=0, le=3600)
    batch_tokens: int = Field(default=4096, ge=1)
    api_key_env: EnvVarName | None = None

    @model_validator(mode="after")
    def _require_model_when_enabled(self) -> EmbeddingSettings:
        if self.enabled and (self.provider is None or self.model is None):
            raise ValueError("embedding.provider and embedding.model are required when enabled")
        return self


class RetrievalSettings(_Section):
    """Search defaults; ``dense`` stays the default until an evaluation clears hybrid."""

    mode: Literal["dense", "lexical", "hybrid"] = "dense"
    top_k: int = Field(default=8, ge=1, le=100)
    candidate_k: int = Field(default=40, ge=1, le=500)
    max_per_document: int | None = Field(default=None, ge=1)
    query_max_tokens: int = Field(default=256, ge=1, le=8192)
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = Field(default=60, ge=1, le=1000)
    dense_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    lexical_weight: float = Field(default=0.5, ge=0.0, le=100.0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _top_k_fits_candidates(self) -> RetrievalSettings:
        if self.top_k > self.candidate_k:
            raise ValueError("retrieval.top_k must not exceed retrieval.candidate_k")
        return self


class RerankSettings(_Section):
    """Optional cross-encoder reranking; off until a measured gain justifies it (see ADR 0014)."""

    enabled: bool = False
    model: ModelName | None = None
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_dir: Path | None = None
    pool_size: int = Field(default=30, ge=1, le=100)
    timeout_s: float = Field(default=10.0, gt=0, le=600)
    batch_size: int = Field(default=8, ge=1, le=64)
    max_pair_tokens: int = Field(default=512, ge=16, le=8192)
    threads: int = Field(default=1, ge=1, le=64)

    @model_validator(mode="after")
    def _require_a_pinned_model_when_enabled(self) -> RerankSettings:
        if self.enabled and (self.model is None or self.model_dir is None or self.revision is None):
            raise ValueError("rerank.model, rerank.model_dir and rerank.revision are required")
        return self


class ContextSettings(_Section):
    """How retrieved chunks become evidence."""

    max_chunks_per_document: int | None = Field(default=3, ge=1)
    order: Literal["by_score", "by_document_order"] = "by_score"
    small_to_big: bool = False
    min_evidence_items: int = Field(default=1, ge=1, le=100)
    item_overhead_tokens: int = Field(default=8, ge=0, le=256)


class LlmSettings(_Section):
    enabled: bool = False
    provider: ProviderName | None = None
    model: ModelName | None = None
    context_window: int | None = Field(default=None, ge=256, le=4_194_304)
    max_new_tokens: int = Field(default=512, ge=1, le=65_536)
    request_timeout_s: float = Field(default=180.0, gt=0, le=3600)
    api_key_env: EnvVarName | None = None

    @model_validator(mode="after")
    def _require_window_when_enabled(self) -> LlmSettings:
        if self.enabled:
            if self.provider is None or self.model is None or self.context_window is None:
                raise ValueError(
                    "llm.provider, llm.model and llm.context_window are required when enabled"
                )
            if self.max_new_tokens >= self.context_window:
                raise ValueError("llm.max_new_tokens must be smaller than llm.context_window")
        return self


class IngestSettings(_Section):
    """Retry and lease behaviour of the ingestion job queue."""

    max_attempts: int = Field(default=5, ge=1, le=100)
    retry_base_delay_s: float = Field(default=30.0, ge=0, le=86_400)
    retry_factor: float = Field(default=2.0, ge=1, le=10)
    retry_max_delay_s: float = Field(default=3600.0, ge=0, le=604_800)
    retry_jitter: float = Field(default=0.2, ge=0, le=1)
    lease_s: float = Field(default=300.0, gt=0, le=86_400)
    keep_versions: int = Field(default=3, ge=1, le=1000)


class ParsingSettings(_Section):
    """Limits applied to every document and the formats parsed in a separate process."""

    max_text_chars: int = Field(default=20_000_000, ge=1_000, le=500_000_000)
    max_blocks: int = Field(default=200_000, ge=10, le=10_000_000)
    max_pages: int = Field(default=2_000, ge=1, le=100_000)
    max_archive_mb: int = Field(default=200, ge=1, le=10_240)
    max_archive_ratio: int = Field(default=100, ge=2, le=10_000)
    max_archive_entries: int = Field(default=10_000, ge=1, le=1_000_000)
    max_nesting: int = Field(default=200, ge=2, le=10_000)
    timeout_s: float = Field(default=60.0, gt=0, le=3600)
    isolate: tuple[Literal["pdf", "docx", "html", "md", "txt"], ...] = ("pdf", "docx", "html")
    isolation_memory_mb: int = Field(default=1024, ge=128, le=65_536)


class ChunkingSettings(_Section):
    """Chunk size policy, the strategy and the token counter."""

    target_tokens: int = Field(default=256, ge=16, le=8192)
    max_tokens: int = Field(default=384, ge=16, le=8192)
    min_tokens: int = Field(default=48, ge=1, le=8192)
    overlap_ratio: float = Field(default=0.12, ge=0, lt=0.5)
    strategy: Literal["auto", "structure_aware", "recursive", "fixed"] = "auto"
    counter: Literal["heuristic", "tokenizers"] = "heuristic"
    tokenizer_json: Path | None = None

    @model_validator(mode="after")
    def _check_sizes(self) -> ChunkingSettings:
        if not self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("chunking sizes need min_tokens <= target_tokens <= max_tokens")
        if self.counter == "tokenizers" and self.tokenizer_json is None:
            raise ValueError("chunking.tokenizer_json is required for the 'tokenizers' counter")
        return self


class LoggingSettings(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_output: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    profile: ResourceProfile = ResourceProfile.LOW
    limits: LimitOverrides = Field(default_factory=LimitOverrides)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    parsing: ParsingSettings = Field(default_factory=ParsingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Dotenv and secret files are intentionally not sources: secrets come from
        # the process environment only.
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls))

    def effective_limits(self) -> ProfileLimits:
        """Profile defaults with explicit overrides applied."""
        overrides = {k: v for k, v in self.limits.model_dump().items() if v is not None}
        return replace(PROFILE_LIMITS[self.profile], **overrides)


def _describe(error: ValidationError) -> tuple[str, str]:
    lines: list[str] = []
    fields: list[str] = []
    for item in error.errors(include_input=False, include_url=False):
        path = ".".join(str(part) for part in item["loc"])
        message = item["msg"].removeprefix("Value error, ")
        if item["type"] == "extra_forbidden" and _SECRET_LIKE.search(path):
            message += " (secrets must be provided through environment variables)"
        lines.append(f"{path or '<root>'}: {message}")
        fields.append(path)
    return "; ".join(lines), ",".join(fields)


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate settings; any problem becomes ``E_CONFIG_INVALID``.

    Error messages name fields and reasons only, never the offending values.
    """
    settings_cls: type[Settings] = Settings
    if config_path is not None:
        if not config_path.is_file():
            raise RagError(
                ErrorCode.E_CONFIG_INVALID,
                "configuration file not found",
                context={"path": str(config_path)},
            )

        class _FileSettings(Settings):
            model_config = SettingsConfigDict(toml_file=config_path)

        settings_cls = _FileSettings
    try:
        return settings_cls()
    except ValidationError as exc:
        message, fields = _describe(exc)
        raise RagError(
            ErrorCode.E_CONFIG_INVALID,
            f"invalid configuration: {message}",
            context={"fields": fields},
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise RagError(
            ErrorCode.E_CONFIG_INVALID,
            f"configuration file is not valid TOML: {exc}",
            context={"path": str(config_path)},
        ) from exc


def read_secret(env_name: str, environ: Mapping[str, str]) -> str:
    """Return the secret stored in environment variable ``env_name``."""
    value = environ.get(env_name)
    if not value:
        raise RagError(
            ErrorCode.E_CONFIG_INVALID,
            f"environment variable {env_name} is not set",
            context={"env": env_name},
        )
    return value
