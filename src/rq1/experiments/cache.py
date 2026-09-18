"""Prevent stale results when data, implementation, models or settings change."""
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path


def _code_hashes(settings):
    package = Path(__file__).resolve().parents[1]
    roots = [package / "core", package / "eval", package / "llm"]
    roots += [package / "datasets" / f"{settings.dataset}.py"]
    if settings.backend in ("ms_graphrag", "mock"):
        roots += [package / "backends" / "ms_graphrag"]
    else:
        roots += [package / "backends" / settings.backend]
    roots += [package / "experiments" / "run_grid.py"]
    files = []
    for root in roots:
        files.extend(root.rglob("*.py") if root.is_dir() else [root])
    return {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(files)) if path.exists()}


def cache_identity(settings, docs, questions):
    config = asdict(settings)
    for key in ("api_key", "embedding_api_key", "project", "output", "index_store",
                "reuse_cells_from", "require_reuse_cells", "data_root", "questions", "corpus", "workers",
                "retries", "retry_delay_seconds"):
        config.pop(key, None)
    code = _code_hashes(settings)
    versions = {}
    for name in ("graphrag", "tiktoken", "pandas", "pyarrow"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    payload = dict(config=config, documents=[asdict(x) for x in docs],
                   questions=[asdict(q) for qs in questions.values() for q in qs], code=code, versions=versions)
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    scientific_config = dict(config)
    scientific_config.pop("name", None)
    scientific_config.pop("sizes", None)
    scientific = hashlib.sha256(json.dumps(dict(
        config=scientific_config, documents=payload["documents"],
        questions=payload["questions"], code=code, versions=versions,
    ), sort_keys=True, default=str).encode()).hexdigest()
    return fingerprint, scientific, config, versions


def guard_run(settings, docs, questions):
    fingerprint, scientific, config, versions = cache_identity(settings, docs, questions)
    path = settings.output / "cache_identity.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Cache identity changed; choose a new data.output directory")
    else:
        if (settings.output / "indexes").exists() or (settings.output / "cells").exists():
            raise ValueError("Unversioned existing cache; choose a new data.output directory")
        path.write_text(json.dumps(dict(
            fingerprint=fingerprint,
            scientific_fingerprint=scientific,
            config=config,
            storage={"index_store": str(settings.index_store) if settings.index_store else None},
            versions=versions,
        ), indent=2))
    return fingerprint


def scientific_fingerprint(settings, docs, questions):
    return cache_identity(settings, docs, questions)[1]
