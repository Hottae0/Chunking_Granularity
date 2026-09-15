from __future__ import annotations

import re
from collections import Counter


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def qa_proxy(predicted: str, gold: str) -> dict[str, float]:
    """Transparent local EM/F1 proxy. Official judge Accuracy is separate."""
    prediction, answer = normalize(predicted), normalize(gold)
    em = float(prediction == answer)
    p, g = Counter(prediction.split()), Counter(answer.split())
    common = sum((p & g).values())
    precision = common / sum(p.values()) if p else 0.0
    recall = common / sum(g.values()) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"qa_em": em, "answer_f1": f1, "qa_accuracy_proxy": em}
