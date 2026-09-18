from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path


def _context_sources(context_data) -> list[int]:
    """Read ranked source human-readable IDs from official Local Search context."""
    if not isinstance(context_data, dict):
        return []
    frame = context_data.get("sources")
    if frame is None:
        frame = context_data.get("text_units")
    if frame is None:
        return []
    if hasattr(frame, "to_dict"):
        rows = frame.to_dict("records")
    elif isinstance(frame, list):
        rows = frame
    else:
        return []
    result = []
    for row in rows:
        value = row.get("id", row.get("human_readable_id"))
        try:
            result.append(int(value))
        except (ValueError, TypeError):
            pass
    return result


class OfficialLocalSearch:
    def __init__(self, graph_root: Path, tables, community_level: int,
                 context_tokens: int, community_prop: float):
        from graphrag.config.load_config import load_config
        self.config = load_config(root_dir=graph_root)
        # Query settings belong to the experiment, not to the index that happens
        # to be reused. This keeps E comparisons on the same retrieval budget.
        self.config.local_search.max_context_tokens = context_tokens
        self.config.local_search.text_unit_prop = 0.5
        self.config.local_search.community_prop = community_prop
        self.tables = tables
        self.community_level = community_level

    def query(self, question: str) -> tuple[str, list[int], float]:
        import graphrag.api as api
        start = time.perf_counter()
        response, context = asyncio.run(api.local_search(
            config=self.config, entities=self.tables["entities"],
            communities=self.tables["communities"],
            community_reports=self.tables["community_reports"],
            text_units=self.tables["text_units"],
            relationships=self.tables["relationships"],
            covariates=None, community_level=self.community_level,
            response_type="Single Sentence", query=question))
        return str(response), _context_sources(context), time.perf_counter() - start


class MockLocalSearch:
    def __init__(self, chunks, client, evidence_budget: int):
        self.chunks, self.client, self.evidence_budget = chunks, client, evidence_budget

    def query(self, question: str, source: str) -> tuple[str, list[int], float, int, int]:
        start = time.perf_counter()
        qwords = set(re.findall(r"\w+", question.lower()))
        ranked = sorted(enumerate(self.chunks), key=lambda pair: (
            -len(qwords & set(re.findall(r"\w+", pair[1].text.lower()))), pair[0]))
        selected, used = [], 0
        for i, chunk in ranked:
            if used + chunk.n_tokens > self.evidence_budget:
                continue
            selected.append(i)
            used += chunk.n_tokens
        prompt = question + "\n" + "\n".join(self.chunks[i].text for i in selected)
        completion = self.client.complete(prompt)
        return completion.text, selected, time.perf_counter() - start, completion.input_tokens, completion.output_tokens
