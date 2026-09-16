from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project: Path
    name: str
    sizes: tuple[int, ...]
    overlap_ratio: float
    context_tokens: int
    evidence_tokens: int
    community_level: int
    seed: int
    max_documents: int | None
    document_selection: str
    max_questions_per_document: int | None
    retries: int
    retry_delay_seconds: float
    workers: int
    backend: str
    data_root: Path
    questions: Path
    corpus: Path
    output: Path
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


def load_settings(config_path: str | Path) -> Settings:
    import json
    config_path = Path(config_path).resolve()
    project = config_path.parent.parent
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
    if config_path.suffix == ".json":
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install project dependencies: python -m pip install -e .") from exc
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    e, d = raw["experiment"], raw["data"]
    sizes = tuple(int(x) for x in e["sizes"])
    if not sizes or len(set(sizes)) != len(sizes) or any(x <= 0 for x in sizes):
        raise ValueError("sizes must be unique positive integers")
    overlap = float(e.get("overlap_ratio", 0.0))
    if not 0 <= overlap < 1:
        raise ValueError("overlap_ratio must be in [0,1)")
    if e.get("backend", "official") not in ("official", "mock"):
        raise ValueError("backend must be official or mock")
    for key in ("context_tokens", "evidence_tokens", "workers", "indexing_concurrency"):
        if int(e.get(key, 1)) <= 0: raise ValueError(f"{key} must be positive")
    if int(e.get("retries", 3)) < 0: raise ValueError("retries must be nonnegative")
    root = project / d["root"]
    return Settings(project, str(e["name"]), sizes, overlap,
                    int(e["context_tokens"]), int(e["evidence_tokens"]),
                    int(e["community_level"]), int(e["seed"]),
                    e.get("max_documents"), str(e.get("document_selection", "first")),
                    e.get("max_questions_per_document"),
                    int(e.get("retries", 3)), float(e.get("retry_delay_seconds", 2)),
                    int(e.get("workers", 1)), str(e.get("backend", "official")),
                    root, root / d["questions"], root / d["corpus"],
                    project / d["output"], os.getenv("BASE_URL", "http://127.0.0.1:8000/v1"),
                    os.getenv("API_KEY", "local-key"), os.getenv("MODEL", "unset"),
                    os.getenv("EMBEDDING_MODEL", "unset"),
                    os.getenv("LLM_BACKEND", "openai_compatible"),
                    float(os.getenv("CHAT_INPUT_USD_PER_MILLION", "0")),
                    float(os.getenv("CHAT_OUTPUT_USD_PER_MILLION", "0")),
                    os.getenv("EMBEDDING_BASE_URL") or os.getenv("BASE_URL", "http://127.0.0.1:8000/v1"),
                    os.getenv("EMBEDDING_API_KEY") or os.getenv("API_KEY", "local-key"),
                    int(os.getenv("EMBEDDING_DIMENSIONS", "3072")),
                    int(e.get("indexing_concurrency", 2)))
