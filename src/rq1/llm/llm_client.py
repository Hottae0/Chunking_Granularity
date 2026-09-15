from __future__ import annotations

from .base import LLMClient
from .mock import MockClient
from .openai_compatible import OpenAICompatibleClient


def make_client(settings) -> LLMClient:
    if settings.llm_backend == "mock" or settings.backend == "mock":
        return MockClient()
    if settings.llm_backend == "openai_compatible":
        return OpenAICompatibleClient(settings.base_url, settings.api_key,
                                      settings.model, settings.embedding_model,
                                      settings.retries, settings.retry_delay_seconds)
    raise ValueError("Unknown LLM backend. Add a new LLMClient implementation.")
