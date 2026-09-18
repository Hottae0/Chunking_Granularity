from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project: Path
    name: str
    dataset: str
    backend: str
    sizes: tuple[int, ...]
    overlap_ratio: float
    context_tokens: int
    evidence_tokens: int
    community_level: int
    community_prop: float
    seed: int
    max_documents: int | None
    document_selection: str
    max_questions_per_document: int | None
    retries: int
    retry_delay_seconds: float
    workers: int
    data_root: Path
    questions: Path
    corpus: Path
    output: Path
    index_store: Path | None
    reuse_cells_from: tuple[Path, ...]
    require_reuse_cells: tuple[str, ...]
    base_url: str
    api_key: str
    model: str
    embedding_model: str
    llm_backend: str
    chat_input_usd_per_million: float
    chat_output_usd_per_million: float
    embedding_base_url: str
    embedding_api_key: str
    embedding_dimensions: int
    indexing_concurrency: int


def _read(path: Path) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install project dependencies: python -m pip install -e .") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _project_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise ValueError(f"Cannot find project root above {path}")


def _load_dotenv(project: Path) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(project / ".env", override=False)
    except ImportError:
        dotenv = project / ".env"
        if dotenv.exists():
            for line in dotenv.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def _resolve(project: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def load_settings(config_path: str | Path) -> Settings:
    config_path = Path(config_path).resolve()
    project = _project_root(config_path)
    _load_dotenv(project)
    raw = _read(config_path)
    e = raw["experiment"]

    # New configs compose experiment, dataset, and backend files. Legacy combined
    # configs remain readable for synthetic fixtures and old run manifests.
    dataset_name = str(e.get("dataset", raw.get("dataset", {}).get("name", "synthetic")))
    backend_name = str(e.get("backend", raw.get("backend", {}).get("name", "mock")))
    if backend_name == "official":
        backend_name = "ms_graphrag"
    dataset = raw.get("data") or raw.get("dataset") or {}
    backend = raw.get("backend") if isinstance(raw.get("backend"), dict) else {}
    if isinstance(e.get("dataset"), str):
        dataset = _read(project / "configs" / "datasets" / f"{dataset_name}.yaml")["dataset"]
    if isinstance(e.get("backend"), str):
        backend = _read(project / "configs" / "backends" / f"{backend_name}.yaml")["backend"]

    sizes = tuple(int(x) for x in e["sizes"])
    if not sizes or len(set(sizes)) != len(sizes) or any(x <= 0 for x in sizes):
        raise ValueError("sizes must be unique positive integers")
    overlap = float(e.get("overlap_ratio", backend.get("overlap_ratio", 0.0)))
    if not 0 <= overlap < 1:
        raise ValueError("overlap_ratio must be in [0,1)")
    if backend_name not in ("ms_graphrag", "hipporag", "mock"):
        raise ValueError("backend must be ms_graphrag, hipporag, or mock")

    def option(key, default):
        return e.get(key, backend.get(key, default))

    for key in ("context_tokens", "evidence_tokens", "workers", "indexing_concurrency"):
        if int(option(key, 1)) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(option("retries", 3)) < 0:
        raise ValueError("retries must be nonnegative")

    root = _resolve(project, dataset["root"])
    output_value = e.get("output", dataset.get("output"))
    if not output_value:
        raise ValueError("experiment.output is required")
    index_value = e.get("index_store", dataset.get("index_store"))
    reuse = tuple(_resolve(project, path) for path in e.get("reuse_cells_from", ()))
    return Settings(
        project=project, name=str(e["name"]), dataset=dataset_name, backend=backend_name,
        sizes=sizes, overlap_ratio=overlap,
        context_tokens=int(option("context_tokens", 4096)),
        evidence_tokens=int(option("evidence_tokens", 2048)),
        community_level=int(option("community_level", 2)),
        community_prop=float(option("community_prop", 0.15, seed=int(e.get("seed", 42)),
        max_documents=e.get("max_documents"),
        document_selection=str(e.get("document_selection", "first")),
        max_questions_per_document=e.get("max_questions_per_document"),
        retries=int(option("retries", 3)), retry_delay_seconds=float(option("retry_delay_seconds", 2)),
        workers=int(option("workers", 1)), data_root=root,
        questions=root / dataset["questions"], corpus=root / dataset["corpus"],
        output=_resolve(project, output_value),
        index_store=_resolve(project, index_value) if index_value else None,
        reuse_cells_from=reuse,
        require_reuse_cells=tuple(str(cell) for cell in e.get("require_reuse_cells", ())),
        base_url=os.getenv("BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("API_KEY", "local-key"), model=os.getenv("MODEL", "unset"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "unset"),
        llm_backend=os.getenv("LLM_BACKEND", "openai_compatible"),
        chat_input_usd_per_million=float(os.getenv("CHAT_INPUT_USD_PER_MILLION", "0")),
        chat_output_usd_per_million=float(os.getenv("CHAT_OUTPUT_USD_PER_MILLION", "0")),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("BASE_URL", "http://127.0.0.1:8000/v1"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("API_KEY", "local-key"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "3072")),
        indexing_concurrency=int(option("indexing_concurrency", 2)),
    )
