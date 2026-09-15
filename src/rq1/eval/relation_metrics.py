from __future__ import annotations

import re


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def relation_metrics(gold_relations: tuple[str, ...], graph_relationships) -> dict[str, float | None]:
    """Lexical diagnostic proxy; descriptions are LM summaries, not benchmark triples."""
    if not gold_relations:
        return {"relation_recall_proxy": None, "path_coverage_proxy": None}
    graph = [(str(r.get("source", "")), str(r.get("target", "")),
              str(r.get("description", ""))) for r in graph_relationships]
    matched = []
    for gold in gold_relations:
        words = _words(gold)
        hit = any(len(words & _words(" ".join(edge))) / max(1, len(words)) >= 0.5
                  for edge in graph)
        matched.append(hit)
    recall = sum(matched) / len(matched)
    return {"relation_recall_proxy": recall,
            "path_coverage_proxy": float(all(matched))}
