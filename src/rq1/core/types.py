"""Shared immutable records used across datasets and backends."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    text: str
    start: int
    end: int
    n_tokens: int


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
