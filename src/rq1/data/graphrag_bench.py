"""Strict loader for the official Novel schema and local text fixtures."""
from dataclasses import dataclass
import ast
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
    raise ValueError("Evidence annotations must be a string or list of strings")


def relation_statements(value):
    """Normalize official evidence_triple variants without discarding structure."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text[0] in "[{(" and text[-1] in "]})":
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
                if not isinstance(parsed, str):
                    return relation_statements(parsed)
        return (text,)
    if isinstance(value, dict):
        return (json.dumps(value, ensure_ascii=False, sort_keys=True),)
    if isinstance(value, (list, tuple)):
        normalized = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (list, tuple)):
                if all(not isinstance(part, (list, tuple, dict)) for part in item):
                    parts = [str(part).strip() for part in item if str(part).strip()]
                    if parts:
                        normalized.append("(" + ", ".join(parts) + ")")
                else:
                    normalized.extend(relation_statements(item))
            elif isinstance(item, dict):
                normalized.extend(relation_statements(item))
            elif isinstance(item, str):
                if item.strip():
                    normalized.append(item.strip())
            else:
                normalized.append(str(item))
        return tuple(normalized)
    raise ValueError("Relation annotations must be strings, triples, or mappings")

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
            relation_statements(row.get("evidence_relations", row.get("evidence_triple")))))
    for name, group in grouped.items():
        if max_questions_per_document is not None:
            if max_questions_per_document <= 0: raise ValueError("Question limit must be positive")
            grouped[name] = group[:max_questions_per_document]
        if not grouped[name]: raise ValueError(f"No questions for {name}")
    return docs, grouped
