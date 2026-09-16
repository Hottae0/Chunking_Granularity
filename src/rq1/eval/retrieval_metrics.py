from __future__ import annotations

from rq1.alignment.provenance import overlap
import re


def evidence_statement_proxy(evidence: tuple[str, ...], ranked_chunks,
                             token_budget: int) -> dict[str, float | None]:
    if not evidence:
        return {f"evidence_recall_at_{k}_proxy": None for k in (1, 5, 10)} | {
            "fixed_budget_evidence_recall_proxy": None}
    words = [set(re.findall(r"\w+", x.casefold())) for x in evidence]
    def recall(chunks):
        return sum(any(len(g & set(re.findall(r"\w+", c.text.casefold()))) / max(1, len(g)) >= 0.5
                       for c in chunks) for g in words) / len(words)
    result = {f"evidence_recall_at_{k}_proxy": recall(ranked_chunks[:k]) for k in (1, 5, 10)}
    selected, used = [], 0
    for chunk in ranked_chunks:
        if used + chunk.n_tokens <= token_budget:
            selected.append(chunk)
            used += chunk.n_tokens
    result["fixed_budget_evidence_recall_proxy"] = recall(selected)
    return result


def evidence_metrics(evidence: tuple[str, ...], ranked_chunks,
                     source: str, source_text: str, token_budget: int) -> dict[str, float | None]:
    """Evidence is a textual statement, not an annotated span.

    Exact statements found in source are evaluated by source overlap. Unlocatable
    statements yield null, never a fabricated zero.
    """
    spans = []
    for statement in evidence:
        at = source_text.find(statement)
        if at < 0:
            return {f"evidence_recall_at_{k}": None for k in (1, 5, 10)} | {
                "fixed_budget_evidence_recall": None,
                "evidence_span_precision": None, "evidence_span_recall": None,
                "evidence_span_f1": None}
        spans.append((at, at + len(statement)))
    if not spans:
        return {f"evidence_recall_at_{k}": None for k in (1, 5, 10)} | {
            "fixed_budget_evidence_recall": None,
            "evidence_span_precision": None, "evidence_span_recall": None,
            "evidence_span_f1": None}
    chunks = list(ranked_chunks)
    def recall(selected):
        return sum(any(c.document == source and overlap(s, e, c.start, c.end) > 0 for c in selected) for s, e in spans) / len(spans)
    results = {f"evidence_recall_at_{k}": recall(chunks[:k]) for k in (1, 5, 10)}
    selected, used = [], 0
    for chunk in chunks:
        if used + chunk.n_tokens > token_budget:
            continue
        used += chunk.n_tokens
        selected.append(chunk)
    results["fixed_budget_evidence_recall"] = recall(selected)
    gold_chars = set().union(*(set((source, i) for i in range(s, e)) for s, e in spans))
    found_chars = set().union(*(set((c.document, i) for i in range(c.start, c.end)) for c in selected))
    hit = len(gold_chars & found_chars)
    precision = hit / len(found_chars) if found_chars else 0.0
    gold_recall = hit / len(gold_chars) if gold_chars else 0.0
    results.update({"evidence_span_precision": precision,
                    "evidence_span_recall": gold_recall,
                    "evidence_span_f1": 2 * precision * gold_recall / (precision + gold_recall)
                    if precision + gold_recall else 0.0})
    return results
