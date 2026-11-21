"""Resource profiles: explicit, validated limits for the target machine class."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class ResourceProfile(StrEnum):
    LOW = "low"
    STANDARD = "standard"


@dataclass(frozen=True, slots=True)
class ProfileLimits:
    """Concurrency and memory limits that bound the engine's resource use."""

    llm_concurrency: int
    embed_concurrency: int
    embed_batch: int
    ingest_workers: int
    max_loaded_vectors_mb: int
    reranker_enabled: bool


PROFILE_LIMITS: Mapping[ResourceProfile, ProfileLimits] = MappingProxyType(
    {
        ResourceProfile.LOW: ProfileLimits(
            llm_concurrency=1,
            embed_concurrency=1,
            embed_batch=16,
            ingest_workers=1,
            max_loaded_vectors_mb=512,
            reranker_enabled=False,
        ),
        ResourceProfile.STANDARD: ProfileLimits(
            llm_concurrency=1,
            embed_concurrency=2,
            embed_batch=32,
            ingest_workers=2,
            max_loaded_vectors_mb=2048,
            reranker_enabled=False,
        ),
    }
)
