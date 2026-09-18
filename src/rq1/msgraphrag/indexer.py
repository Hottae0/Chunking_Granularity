from __future__ import annotations

import hashlib
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REQUIRED_TABLES = ("entities", "relationships", "text_units", "documents",
                   "communities", "community_reports")


def graph_dir(output: Path, extraction_size: int, graph_store: Path | None = None) -> Path:
    """Resolve a graph from the shared store, falling back to a run-local store."""
    return (graph_store if graph_store is not None else output / "graphs") / f"e{extraction_size}"


def _input_name(source_name: str) -> str:
    return hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:16] + ".txt"


def _configured_models(raw: dict, section: str) -> set[str]:
    entries = raw.get(section) or {}
    if not isinstance(entries, dict):
        raise ValueError(f"Invalid {section} in graph settings")
    return {
        str(value["model"])
        for value in entries.values()
        if isinstance(value, dict) and value.get("model")
    }


def validate_graph(root: Path, extraction_size: int, documents, settings) -> bool:
    """Reject an incompatible shared graph and report whether it is complete."""
    import yaml

    settings_path = root / "settings.yaml"
    titles_path = root / "input_titles.json"
    input_dir = root / "input"
    present = [path.exists() for path in (settings_path, titles_path, input_dir)]
    if not any(present):
        return False
    if not all(present):
        raise ValueError(f"Cannot reuse partial graph {root}: metadata is incomplete")

    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    chunking = raw.get("chunking") or {}
    expected_overlap = int(extraction_size * settings.overlap_ratio)
    actual_size = int(chunking.get("size", -1))
    actual_overlap = int(chunking.get("overlap", -1))
    if (actual_size, actual_overlap) != (extraction_size, expected_overlap):
        raise ValueError(
            f"Cannot reuse {root}: chunking is size={actual_size}, overlap={actual_overlap}; "
            f"expected size={extraction_size}, overlap={expected_overlap}"
        )
    if settings.model not in _configured_models(raw, "completion_models"):
        raise ValueError(f"Cannot reuse {root}: chat model differs")
    if settings.embedding_model not in _configured_models(raw, "embedding_models"):
        raise ValueError(f"Cannot reuse {root}: embedding model differs")
    vector_size = int((raw.get("vector_store") or {}).get("vector_size", -1))
    if vector_size != settings.embedding_dimensions:
        raise ValueError(
            f"Cannot reuse {root}: embedding dimensions are {vector_size}, "
            f"expected {settings.embedding_dimensions}"
        )

    expected_titles = {_input_name(doc.name): doc.name for doc in documents}
    if json.loads(titles_path.read_text(encoding="utf-8")) != expected_titles:
        raise ValueError(f"Cannot reuse {root}: selected documents differ")
    for doc in documents:
        input_path = input_dir / _input_name(doc.name)
        if not input_path.exists() or input_path.read_text(encoding="utf-8") != doc.text:
            raise ValueError(f"Cannot reuse {root}: input text differs for {doc.name}")
    output = root / "output"
    return (
        (root / "graph_complete.json").exists()
        and all((output / f"{name}.parquet").exists() for name in REQUIRED_TABLES)
    )


def _patch_settings(path: Path, settings, extraction_size: int) -> None:
    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section, model_name in (("completion_models", settings.model),
                                ("embedding_models", settings.embedding_model)):
        for model in raw[section].values():
            model["model_provider"] = "openai"
            model["model"] = model_name
            model["api_base"] = settings.embedding_base_url if section == "embedding_models" else settings.base_url
            model["api_key"] = "${GRAPHRAG_EMBEDDING_API_KEY}" if section == "embedding_models" else "${GRAPHRAG_API_KEY}"
            model["retry"] = {"type": "exponential_backoff", "max_retries": settings.retries}
    raw["concurrent_requests"] = settings.indexing_concurrency
    raw["vector_store"]["vector_size"] = settings.embedding_dimensions
    for schema in (raw["vector_store"].get("index_schema") or {}).values():
        schema["vector_size"] = settings.embedding_dimensions
    raw["chunking"]["type"] = "tokens"
    raw["chunking"]["size"] = extraction_size
    raw["chunking"]["overlap"] = int(extraction_size * settings.overlap_ratio)
    raw["chunking"]["encoding_model"] = "cl100k_base"
    raw["local_search"]["max_context_tokens"] = settings.context_tokens
    raw["local_search"]["text_unit_prop"] = 0.5
    raw["local_search"]["community_prop"] = 0.0
    raw["cluster_graph"]["seed"] = settings.seed
    raw["output_storage"]["base_dir"] = "output"
    raw["cache"]["storage"]["base_dir"] = "cache"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def build_graph(settings, documents, extraction_size: int, logger) -> Path:
    """Official standard index, once per g_E across the entire selected corpus."""
    root = graph_dir(settings.output, extraction_size, settings.graph_store)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".index.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is indexing {root}") from None
        return _build_graph(settings, documents, extraction_size, logger)


def _build_graph(settings, documents, extraction_size: int, logger) -> Path:
    root = graph_dir(settings.output, extraction_size, settings.graph_store)
    output = root / "output"
    ready = root / "graph_complete.json"
    complete = validate_graph(root, extraction_size, documents, settings)
    if complete:
        logger.info("Reuse graph %s", root)
        return root
    if settings.model == "unset" or settings.embedding_model == "unset":
        raise ValueError("Set MODEL and EMBEDDING_MODEL in .env")
    root.mkdir(parents=True, exist_ok=True)
    input_dir = root / "input"
    input_dir.mkdir(exist_ok=True)
    titles = {}
    for doc in documents:
        name = _input_name(doc.name)
        (input_dir / name).write_text(doc.text, encoding="utf-8")
        titles[name] = doc.name
    (root / "input_titles.json").write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
    config = root / "settings.yaml"
    env = os.environ.copy()
    env["GRAPHRAG_API_KEY"] = settings.api_key
    env["GRAPHRAG_EMBEDDING_API_KEY"] = settings.embedding_api_key
    if not config.exists():
        # Official init creates compatible prompts for the installed GraphRAG version.
        subprocess.run([sys.executable, "-m", "graphrag", "init", "--root", str(root)],
                       input=f"{settings.model}\n{settings.embedding_model}\n", text=True,
                       check=True, env=env, stdout=(root / "init.log").open("w", encoding="utf-8"))
        _patch_settings(config, settings, extraction_size)
    started = time.perf_counter()
    for attempt in range(settings.retries + 1):
        with (root / "index.log").open("a", encoding="utf-8") as log:
            log.write(f"\nAttempt {attempt + 1}\n")
            result = subprocess.run([sys.executable, "-m", "graphrag", "index", "--root", str(root)],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0 and all((output / f"{x}.parquet").exists() for x in REQUIRED_TABLES):
            ready.write_text(json.dumps({"extraction_size": extraction_size,
                                         "elapsed_seconds": time.perf_counter() - started,
                                         "tables": list(REQUIRED_TABLES)}), encoding="utf-8")
            return root
        if attempt < settings.retries:
            time.sleep(settings.retry_delay_seconds * 2 ** attempt)
    raise RuntimeError(f"GraphRAG indexing failed for e{extraction_size}; see {root / 'index.log'}")
