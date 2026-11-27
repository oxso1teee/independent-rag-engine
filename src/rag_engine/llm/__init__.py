"""Chat adapters: Ollama's native API and OpenAI-compatible servers behind ``LLMProvider``."""

from rag_engine.llm.ollama import OllamaChat
from rag_engine.llm.openai_compat import OpenAICompatChat

__all__ = ["OllamaChat", "OpenAICompatChat"]
