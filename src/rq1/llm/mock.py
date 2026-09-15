from __future__ import annotations

import hashlib
import re
from .base import Completion, LLMClient


class MockClient(LLMClient):
    """Deterministic fixture client; never use its scores as research results."""

    def complete(self, prompt: str) -> Completion:
        text = prompt.lower()
        if "who gave bob" in text and "alice gave bob" in text:
            answer = "Alice"
        elif "who returned" in text and "bob returned" in text:
            answer = "Bob"
        else:
            answer = "unknown"
        return Completion(answer, len(prompt.split()), len(answer.split()), 0.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            words = re.findall(r"\w+", text.lower())
            vector = [0.0] * 32
            for word in words:
                i = int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % 32
                vector[i] += 1.0
            vectors.append(vector)
        return vectors
