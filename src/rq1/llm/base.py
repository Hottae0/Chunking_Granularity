from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> Completion:
        """Return answer and measured usage. Implement for non-OpenAI lab servers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
