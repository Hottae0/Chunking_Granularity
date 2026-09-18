"""Dataset registry."""
from rq1.datasets.graphrag_bench import load_novel


def get_loader(name: str):
    if name in {"graphrag_bench", "synthetic"}:
        return load_novel
    if name == "musique":
        from rq1.datasets.musique import load
        return load
    if name == "2wiki":
        from rq1.datasets.two_wiki import load
        return load
    if name == "qasper":
        from rq1.datasets.qasper import load
        return load
    raise ValueError(f"Unknown dataset: {name}")


def load_dataset(settings):
    return get_loader(settings.dataset)(
        settings.corpus, settings.questions, settings.max_documents,
        settings.max_questions_per_document, settings.document_selection, settings.seed,
    )
