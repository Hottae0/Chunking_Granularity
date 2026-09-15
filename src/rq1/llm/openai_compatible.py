from __future__ import annotations

import time
from .base import Completion, LLMClient


class OpenAICompatibleClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str, embedding_model: str,
                 retries: int = 3, delay: float = 2):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.embedding_model = model, embedding_model
        self.retries, self.delay = retries, delay

    def _call(self, fn):
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except Exception:
                if attempt >= self.retries:
                    raise
                time.sleep(self.delay * 2 ** attempt)

    def complete(self, prompt: str) -> Completion:
        start = time.perf_counter()
        response = self._call(lambda: self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0))
        usage = response.usage
        return Completion(response.choices[0].message.content or "",
                          int(usage.prompt_tokens) if usage else 0,
                          int(usage.completion_tokens) if usage else 0,
                          time.perf_counter() - start)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._call(lambda: self.client.embeddings.create(
            model=self.embedding_model, input=texts))
        return [x.embedding for x in sorted(response.data, key=lambda x: x.index)]
