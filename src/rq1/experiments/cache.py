"""Prevent stale results when data, implementation, models or settings change."""
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

def guard_run(settings, docs, questions):
    config = asdict(settings)
    for key in ("api_key", "project", "output", "data_root", "questions", "corpus", "workers", "retries", "retry_delay_seconds"):
        config.pop(key, None)
    package = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob("*.py"))}
    versions = {}
    for name in ("graphrag", "tiktoken", "pandas", "pyarrow"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    payload = dict(config=config, documents=[asdict(x) for x in docs],
                   questions=[asdict(q) for qs in questions.values() for q in qs], code=code, versions=versions)
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    path = settings.output / "cache_identity.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Cache identity changed; choose a new data.output directory")
    else:
        if (settings.output / "graphs").exists() or (settings.output / "cells").exists():
            raise ValueError("Unversioned existing cache; choose a new data.output directory")
        path.write_text(json.dumps(dict(fingerprint=fingerprint, config=config, versions=versions), indent=2))
    return fingerprint
