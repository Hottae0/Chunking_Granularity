from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REQUIRED_TABLES = ("entities", "relationships", "text_units", "documents",
                   "communities", "community_reports")


def graph_dir(output: Path, extraction_size: int) -> Path:
    return output / "graphs" / f"e{extraction_size}"


def _input_name(source_name: str) -> str:
    return hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:16] + ".txt"


def _patch_settings(path: Path, settings, extraction_size: int) -> None:
    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section, model_name in (("completion_models", settings.model),
                                ("embedding_models", settings.embedding_model)):
        for model in raw[section].values():
            model["model_provider"] = "openai"
            model["model"] = model_name
            model["api_base"] = settings.base_url
            model["api_key"] = "${GRAPHRAG_API_KEY}"
            model["retry"] = {"type": "exponential_backoff", "max_retries": settings.retries}
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
    root = graph_dir(settings.output, extraction_size)
    output = root / "output"
    ready = root / "graph_complete.json"
    if ready.exists() and all((output / f"{x}.parquet").exists() for x in REQUIRED_TABLES):
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
