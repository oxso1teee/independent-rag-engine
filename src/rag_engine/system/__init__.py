"""Adapters of the real environment: the system clock and identifier entropy."""

from rag_engine.system.memory import trim_heap
from rag_engine.system.runtime import SystemClock, system_id_factory

__all__ = ["SystemClock", "system_id_factory", "trim_heap"]
