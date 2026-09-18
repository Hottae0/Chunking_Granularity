"""Dataset loader contract."""
from typing import Protocol

from rq1.core.types import Document, Question


class DatasetLoader(Protocol):
    def __call__(self, corpus, questions, max_documents=None,
                 max_questions_per_document=None, document_selection="first",
                 seed=42) -> tuple[list[Document], dict[str, list[Question]]]: ...
