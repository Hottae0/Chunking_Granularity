"""Strict loader for the official Novel schema and local text fixtures."""
from dataclasses import dataclass
from pathlib import Path
import json
import random

@dataclass(frozen=True)
class Document:
    name: str
    text: str

@dataclass(frozen=True)
class Question:
    id: str
    source: str
    question: str
    answer: str
    question_type: str
    evidence: tuple[str, ...]
    evidence_relations: tuple[str, ...]

def statements(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return tuple(x for x in value if x.strip())
    raise ValueError("Annotations must be a string or list of strings")

def load_novel(corpus, questions, max_documents=None, max_questions_per_document=None,
               document_selection="first", seed=42):
    corpus, questions = Path(corpus), Path(questions)
    if corpus.is_dir():
        docs = [Document(p.stem, p.read_text(encoding="utf-8")) for p in sorted(corpus.glob("*.txt"))]
    else:
        docs = [Document(x["corpus_name"], x["context"]) for x in json.loads(corpus.read_text(encoding="utf-8"))]
    if not docs or len({d.name for d in docs}) != len(docs) or any(not d.text.strip() for d in docs):
        raise ValueError("Corpus must contain nonempty, uniquely named documents")
    docs.sort(key=lambda d:d.name)
    known = {d.name for d in docs}
    if document_selection not in ("first", "seeded_random"):
        raise ValueError("Unknown document selection")
    if max_documents is not None:
        if max_documents <= 0: raise ValueError("max_documents must be positive")
        docs = (random.Random(seed).sample(docs, min(max_documents, len(docs)))
                if document_selection == "seeded_random" else docs[:max_documents])
    grouped = {d.name: [] for d in docs}
    seen = set()
    for row in json.loads(questions.read_text(encoding="utf-8")):
        qid = str(row["source"]) + "::" + str(row["id"])
        if qid in seen: raise ValueError(f"Duplicate question ID: {qid}")
        seen.add(qid)
        if row["source"] not in known: raise ValueError(f"Unknown source: {row['source']}")
        if row["source"] not in grouped: continue
        grouped[row["source"]].append(Question(qid, row["source"], row["question"], row["answer"],
            row["question_type"], statements(row.get("evidence")),
            statements(row.get("evidence_relations", row.get("evidence_triple")))))
    for name, group in grouped.items():
        if max_questions_per_document is not None:
            if max_questions_per_document <= 0: raise ValueError("Question limit must be positive")
            limit = min(max_questions_per_document, len(group))
            if document_selection == "seeded_random":
                rng = random.Random(f"{seed}:{name}")
                indices = sorted(rng.sample(range(len(group)), limit))
                grouped[name] = [group[index] for index in indices]
            else:
                grouped[name] = group[:limit]
        if not grouped[name]: raise ValueError(f"No questions for {name}")
    return docs, grouped
